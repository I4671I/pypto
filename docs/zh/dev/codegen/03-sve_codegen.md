# SVE Codegen（鲲鹏 CPU）

`pypto.ir.sve_codegen` 把早期 Tile IR lower 成面向带 SVE（可伸缩向量扩展，Scalable Vector
Extension）的 Arm CPU（如鲲鹏 920F）的 C 内核。PyPTO 只输出普通的逐元素 C 循环和目标编译参数，
由 C 编译器的循环向量化器把循环翻译成 SVE 指令。

## 流水线

```text
@pl.program with InCore functions
  → OptimizationStrategy.CPUSVE   (generic prefix passes + ConvertTensorToTileOps)
  → sve_codegen.generate()
  → kernels/sve/<func>.c  +  kernels/sve/<func>.json
  → clang -O3 -march=armv8.2-a+sve -msve-vector-bits=512
```

`CPUSVE` 先运行与 `Default` 相同的前缀 Pass（`InlineFunctions` 到 `FlattenCallExpr`），再运行
`ConvertTensorToTileOps`，然后停止。`Default` 中此后的每个 Pass 都在做昇腾相关的决策：2-D Tile
布局、片上存储空间与地址、AIC/AIV 拆分、运行时任务依赖。CPU 的 C 内核不需要这些决策。即使启用了
Buffer IR，也会跳过 `LowerTileToBuffer`。

`sve_codegen.lower()` 运行这条流水线时不配置任何 NPU 后端。它继承当前 `PassContext` 的设置，并关闭
仅适用于昇腾的 `TileInnermostDimGranularity` 性能提示，否则这项检查会查询 NPU 后端。

## 用法

```python
import pypto.language as pl
from pypto.ir import sve_codegen


@pl.program
class VectorAdd:
    @pl.function(type=pl.FunctionType.InCore)
    def vector_add(
        self,
        a: pl.Tensor[[1024], pl.FP32],
        b: pl.Tensor[[1024], pl.FP32],
        c: pl.Out[pl.Tensor[[1024], pl.FP32]],
    ) -> pl.Tensor[[1024], pl.FP32]:
        a_part = pl.tensor.slice(a, [1024], [0])
        b_part = pl.tensor.slice(b, [1024], [0])
        c = pl.tensor.assemble(c, pl.add(a_part, b_part), [0])
        return c


files = sve_codegen.compile_to_files(VectorAdd, "build_output/vector_add")
```

| 函数 | 作用 |
| ---- | ---- |
| `lower(program, *, dump_passes, passes_dump_dir)` | 运行 `CPUSVE`，返回早期 Tile IR |
| `generate(program, target)` | 为每个 InCore 函数返回 `{相对路径: 内容}` |
| `compile_to_files(program, output_dir, *, target, dump_passes)` | 依次调用 `lower` 和 `generate`，再写出文件 |

CPU 后端接受 `@pl.program` 中的 InCore 函数，暂不支持 `@pl.jit` 入口函数，因为后者会 lower 成一个
Orchestration 函数加若干外提的内核。

## Lowering 规则

函数体必须是直线代码，`ConvertTensorToTileOps` 处理逐元素张量程序时产生的正是这种形式。所有 Tile
共用同一个一维长度，因此整个函数体融合成一个 `i < n` 的循环。每个 Tile 被内联成关于 `i` 的 C 表达式。
这样做是正确的，因为 Tile 是 SSA 值，且所有受支持的算子都没有副作用。

| IR | C |
| -- | - |
| 参数 `pl.Tensor[[N], pl.FP32]` | `float* <name>`（去掉 SSA 后缀） |
| `tile.load(a, [off], [N], [N])` | `a[i + off]` |
| `tile.add` / `tile.sub` / `tile.mul` / `tile.div` | `+` / `-` / `*` / `/` |
| `tile.store(t, [off], c)` | `c[i + off] = <t>;` |
| `return c` | 不生成代码，输出原地写回 |

上例生成的内核：

```c
#include <stdint.h>

void vector_add_kernel(float* a, float* b, float* c, int64_t n) {
    #pragma clang loop vectorize(enable) interleave(disable)
    for (int64_t i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}
```

循环边界 `n` 是运行时参数，因此内核从不假设 Tile 长度是 SVE 向量长度的整数倍，尾部由 SVE 的谓词循环
处理。调用方传入的 `n` 不超过编译配置中记录的 Tile 长度。

## 目标配置

`SVETargetConfig`（frozen dataclass）：

| 字段 | 默认值 | 含义 |
| ---- | ------ | ---- |
| `vector_bits` | `512` | `-msve-vector-bits`；`None` 表示生成与向量长度无关的代码 |
| `compiler` | `"clang"` | 写入编译配置的编译器 |
| `march` | `"armv8.2-a+sve"` | `-march`；`streaming` 时需包含 `+sme` |
| `opt_level` | `"-O3"` | 优化参数 |
| `interleave` | `False` | 为真时 pragma 使用 `interleave(enable)`，否则为 `interleave(disable)` |
| `streaming` | `False` | 把内核标记为 `__arm_streaming`（SME 接口） |

用固定 `vector_bits` 编译的程序，只有在执行线程的向量长度恰好等于该值时才正确。Linux 的
`prctl(PR_SVE_SET_VL)` 可以按线程调低向量长度，用于 128/256 位实验。

`streaming=True` 是为后续 SME 矩阵内核预留的接口。内核会带上 `__arm_streaming` 属性，编译器在其前后
插入 `smstart`/`smstop`。逐元素内核使用非流式模式。

## 编译配置

`kernels/sve/<func>.json` 描述如何编译和调用内核：

| 键 | 内容 |
| -- | ---- |
| `kernel` / `symbol` / `source` | IR 函数名、C 符号（`<func>_kernel`）、源文件 |
| `params` | 按调用顺序列出每个张量参数的名称、方向、dtype 和形状 |
| `n` | Tile 长度，即末尾 `int64_t n` 参数应传入的值 |
| `compiler` / `flags` | 来自 `SVETargetConfig` 的编译器和参数 |
| `streaming` | 内核是否为 `__arm_streaming` |

## 当前限制

遇到不支持的 IR 时抛出 `ValueError`，错误信息包含函数名和具体结构：

- FP32 以外的 dtype、非一维张量、动态形状
- 部分有效的 Tile（`valid_shape != shape`）以及超出张量范围的窗口
- load、store 和上述四个二元算子以外的算子
- 控制流（`for`/`if`）、标量参数、非 InCore 函数
- 同一函数中，从此前已被 store 写过的张量再次 load
- 向同一输出写入偏移不同且相互重叠的两个窗口，因为融合后的循环会打乱它们的先后顺序

## 测试

`tests/ut/codegen/test_sve_codegen.py` 检查生成的代码文本、编译配置、目标配置校验和报错信息。它还会用
主机 C 编译器编译内核，并在 512 位向量附近的长度（1、15、16、17、1000）上与 `numpy` float32 结果逐位
比对。SVE 指令选择和性能需要在目标 CPU 上验证。
