# SVE Codegen (Kunpeng CPU)

`pypto.ir.sve_codegen` lowers early Tile IR to C kernels for Arm CPUs with SVE,
such as Kunpeng 920F. PyPTO emits a plain element-wise C loop plus the target
flags; the C compiler's loop vectorizer turns the loop into SVE instructions.

## Pipeline

```text
@pl.program with InCore functions
  → OptimizationStrategy.CPUSVE   (generic prefix passes + ConvertTensorToTileOps)
  → sve_codegen.generate()
  → kernels/sve/<func>.c  +  kernels/sve/<func>.json
  → clang -O3 -march=armv8.2-a+sve -msve-vector-bits=512
```

`CPUSVE` runs the same prefix passes as `Default` (`InlineFunctions` through
`FlattenCallExpr`) followed by `ConvertTensorToTileOps`, and stops there. Every
later `Default` pass encodes an Ascend decision — 2-D tile layouts, on-chip
memory spaces and addresses, AIC/AIV splitting, runtime task dependencies — that
a CPU C kernel does not need. `LowerTileToBuffer` is skipped even when Buffer IR
is enabled.

`sve_codegen.lower()` runs the pipeline without configuring an NPU backend. It
inherits the active `PassContext` settings and disables the Ascend-only
`TileInnermostDimGranularity` performance hint, which would otherwise query the
NPU backend.

## Usage

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

| Function | Purpose |
| -------- | ------- |
| `lower(program, *, dump_passes, passes_dump_dir)` | Run `CPUSVE`; return early Tile IR |
| `generate(program, target)` | Return `{relative_path: content}` for every InCore function |
| `compile_to_files(program, output_dir, *, target, dump_passes)` | `lower` + `generate`, then write the files |

The CPU backend accepts `@pl.program` InCore functions. `@pl.jit` entry
functions are not supported yet, because they lower to an Orchestration function
plus outlined kernels.

## Lowering Rules

The body must be straight-line code, which is what `ConvertTensorToTileOps`
produces for element-wise tensor programs. All tiles share one 1-D extent, so the
whole body fuses into one loop over `i < n`. Each tile is inlined as a C
expression of `i`. This is valid because tiles are SSA values and every
supported op is pure.

| IR | C |
| -- | - |
| Parameter `pl.Tensor[[N], pl.FP32]` | `float* <name>` (the SSA suffix is removed) |
| `tile.load(a, [off], [N], [N])` | `a[i + off]` |
| `tile.add` / `tile.sub` / `tile.mul` / `tile.div` | `+` / `-` / `*` / `/` |
| `tile.store(t, [off], c)` | `c[i + off] = <t>;` |
| `return c` | nothing — outputs are written in place |

Generated kernel for the example above:

```c
#include <stdint.h>

void vector_add_kernel(float* a, float* b, float* c, int64_t n) {
    #pragma clang loop vectorize(enable) interleave(disable)
    for (int64_t i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}
```

The loop bound `n` is a runtime argument, so the kernel never assumes the tile
length is a multiple of the SVE vector length. The SVE predicated loop covers
the tail. Callers pass `n` less than or equal to the tile length that the build
config records.

## Target Configuration

`SVETargetConfig` (frozen dataclass):

| Field | Default | Meaning |
| ----- | ------- | ------- |
| `vector_bits` | `512` | `-msve-vector-bits`; `None` builds vector-length-agnostic code |
| `compiler` | `"clang"` | Driver recorded in the build config |
| `march` | `"armv8.2-a+sve"` | `-march`; needs `+sme` when `streaming` |
| `opt_level` | `"-O3"` | Optimization flag |
| `interleave` | `False` | `interleave(enable)` instead of `interleave(disable)` in the pragma |
| `streaming` | `False` | Mark kernels `__arm_streaming` (SME hook) |

A binary built with a fixed `vector_bits` is only correct when the executing
thread runs at exactly that vector length. Linux `prctl(PR_SVE_SET_VL)` can
lower the length per thread for 128/256-bit experiments.

`streaming=True` is the hook for future SME matrix kernels. The kernel gets the
`__arm_streaming` attribute, and the compiler brackets it with
`smstart`/`smstop`. Element-wise kernels use non-streaming mode.

## Build Config

`kernels/sve/<func>.json` describes how to build and call the kernel:

| Key | Content |
| --- | ------- |
| `kernel` / `symbol` / `source` | IR function name, C symbol (`<func>_kernel`), source file |
| `params` | Name, direction, dtype, and shape of each tensor parameter, in call order |
| `n` | Tile length; the value to pass as the trailing `int64_t n` argument |
| `compiler` / `flags` | Compiler driver and flags from `SVETargetConfig` |
| `streaming` | Whether the kernel is `__arm_streaming` |

## Current Limits

Unsupported IR raises `ValueError` that names the function and the construct:

- dtype other than FP32, tensors of rank other than 1, or dynamic shapes
- partially valid tiles (`valid_shape != shape`) and windows outside the tensor
- operators other than load, store, and the four binary ops above
- control flow (`for`/`if`), scalar parameters, and non-InCore functions
- a load from a tensor that an earlier store in the same function wrote
- two stores to one output at different, overlapping windows, because the fused
  loop would reorder them

## Testing

`tests/ut/codegen/test_sve_codegen.py` checks the generated text, the build
config, the target validation, and the rejection messages. It also compiles the
kernel with the host C compiler and compares the results bit for bit against
`numpy` float32 for lengths around one 512-bit vector (1, 15, 16, 17, 1000).
SVE instruction selection and performance must be verified on the target CPU.
