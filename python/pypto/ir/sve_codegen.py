# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""C code generation for Kunpeng 920F SVE/SME from early Tile IR.

The CPU backend reuses the generic frontend passes and stops at early Tile IR
(``OptimizationStrategy.CPUSVE``). Each InCore function becomes one C kernel
whose element-wise loop the C compiler auto-vectorizes for SVE. See
``docs/en/dev/codegen/03-sve_codegen.md``.
"""

import json
import os
import re
from dataclasses import dataclass

from pypto.pypto_core import DataType
from pypto.pypto_core import ir as _ir_core
from pypto.pypto_core import passes as _passes

from .pass_manager import OptimizationStrategy, PassDumpLevel, PassManager

# Output layout mirrors the PTO backend's kernels/aiv/ directory.
SVE_KERNEL_DIR = "kernels/sve"

# Validated at import: a renamed operator fails here instead of silently
# matching nothing (see operator-identity rules).
_LOAD_OP = _ir_core.get_op("tile.load").name
_STORE_OP = _ir_core.get_op("tile.store").name
# Element-wise binary tile ops and their C operators. All supported dtypes are
# IEEE-754 floating point, where C's operators have the same per-element
# rounding as the tile op, so CPU results stay bit-exact against a float32
# reference.
_BINARY_C_OPERATORS: dict[str, str] = {
    _ir_core.get_op("tile.add").name: "+",
    _ir_core.get_op("tile.sub").name: "-",
    _ir_core.get_op("tile.mul").name: "*",
    _ir_core.get_op("tile.div").name: "/",
}

# -msve-vector-bits accepts the power-of-two SVE lengths only.
_SVE_VECTOR_BITS = (128, 256, 512, 1024, 2048)

# Names the kernel body itself uses; parameters must not shadow them.
_LOOP_INDEX = "i"
_LOOP_BOUND = "n"
_SSA_SUFFIX_RE = re.compile(r"__ssa_v\d+$")
_NON_IDENTIFIER_RE = re.compile(r"\W")


@dataclass(frozen=True)
class SVETargetConfig:
    """Compiler target for generated SVE kernels.

    Attributes:
        vector_bits: Fixed SVE vector length, passed as ``-msve-vector-bits``.
            The default 512 is the Kunpeng 920F hardware length. A fixed-length
            binary is only correct when the executing thread runs at exactly
            that length (``prctl(PR_SVE_SET_VL)`` can lower it). ``None`` emits
            vector-length-agnostic code.
        compiler: C compiler driver recorded in the build config.
        march: ``-march`` value. It must enable SVE, and SME when
            ``streaming`` is set.
        opt_level: Optimization flag.
        interleave: Emit ``interleave(enable)`` in the loop pragma. Disabled by
            default: the SVE predicated loop already covers the tail without a
            scalar epilogue, and interleaving changes the code the vectorizer
            picks for it.
        streaming: Mark kernels ``__arm_streaming`` so the compiler brackets
            them with ``smstart``/``smstop``. This is the SME hook for future
            matrix kernels; element-wise kernels run in non-streaming mode.
    """

    vector_bits: int | None = 512
    compiler: str = "clang"
    march: str = "armv8.2-a+sve"
    opt_level: str = "-O3"
    interleave: bool = False
    streaming: bool = False

    def __post_init__(self) -> None:
        if self.vector_bits is not None and self.vector_bits not in _SVE_VECTOR_BITS:
            raise ValueError(f"vector_bits must be one of {_SVE_VECTOR_BITS} or None, got {self.vector_bits}")
        if self.streaming and "+sme" not in self.march:
            raise ValueError(
                f"streaming kernels need an SME-enabled march such as 'armv9-a+sme', got '{self.march}'"
            )

    def compile_flags(self) -> list[str]:
        """Return the compiler flags that select this target."""
        flags = [self.opt_level, f"-march={self.march}"]
        if self.vector_bits is not None:
            flags.append(f"-msve-vector-bits={self.vector_bits}")
        return flags


@dataclass(frozen=True)
class _TileElement:
    """C expression for element ``i`` of a tile.

    Attributes:
        expr: Expression over the loop index, e.g. ``a[i + 16]``.
        extent: Number of elements in the 1-D tile.
        compound: True when ``expr`` is a binary expression, which needs
            parentheses when nested inside another operator.
    """

    expr: str
    extent: int
    compound: bool = False

    def as_operand(self) -> str:
        return f"({self.expr})" if self.compound else self.expr


def _const_int_list(expr: _ir_core.Expr, what: str, func_name: str) -> list[int]:
    """Read a MakeTuple of ConstInt (static offsets / shapes) as Python ints."""
    if not isinstance(expr, _ir_core.MakeTuple):
        raise ValueError(f"SVE codegen in '{func_name}': {what} must be a static tuple, got {expr}")
    values: list[int] = []
    for element in expr.elements:
        if not isinstance(element, _ir_core.ConstInt):
            raise ValueError(
                f"SVE codegen in '{func_name}': dynamic {what} {expr} is not supported yet; use static sizes"
            )
        values.append(element.value)
    return values


def _static_extent(shape: list[_ir_core.Expr], what: str, func_name: str) -> int:
    """Return the length of a 1-D static shape."""
    if len(shape) != 1:
        raise ValueError(
            f"SVE codegen in '{func_name}': only 1-D data is supported, {what} has rank {len(shape)}"
        )
    if not isinstance(shape[0], _ir_core.ConstInt):
        raise ValueError(
            f"SVE codegen in '{func_name}': {what} has dynamic length {shape[0]}; use a static shape"
        )
    return shape[0].value


class _KernelEmitter:
    """Translate one InCore function of early Tile IR into a C kernel.

    The body must be straight-line ``tile.load`` / element-wise binary tile ops /
    ``tile.store`` followed by a return, which is exactly what
    ``ConvertTensorToTileOps`` produces for element-wise tensor code. All tiles
    share one 1-D extent, so the body fuses into a single loop over
    ``i < n``: each tile becomes a C expression of ``i`` and each store becomes
    one assignment. Inlining is valid because tiles are SSA values and every
    supported op is pure; a load from a tensor that an earlier store wrote is
    rejected, because the fused loop would read it before the store finished.
    """

    def __init__(self, func: _ir_core.Function, target: SVETargetConfig):
        self._func = func
        self._target = target
        # Keyed by Var.unique_id: name hints are cosmetic and may repeat.
        self._param_names: dict[int, str] = {}
        self._param_directions: dict[int, _ir_core.ParamDirection] = {}
        self._tiles: dict[int, _TileElement] = {}
        # Tensor var id -> id of the parameter whose memory it names. Each
        # parameter maps to itself; the tensor a tile.store returns maps to the
        # parameter it wrote, so a later store or the return can name it.
        self._tensor_aliases: dict[int, int] = {}
        self._param_extents: dict[int, int] = {}
        # Store offsets per output parameter; all windows share self._extent.
        self._store_offsets: dict[str, list[int]] = {}
        self._assignments: list[str] = []
        self._extent: int | None = None

    def _error(self, message: str) -> ValueError:
        return ValueError(f"SVE codegen in '{self._func.name}': {message}")

    # ------------------------------------------------------------------ params

    def _bind_params(self) -> list[str]:
        """Map tensor parameters to unique C identifiers and return the C parameter list."""
        used = {_LOOP_INDEX, _LOOP_BOUND}
        c_params: list[str] = []
        for param, direction in zip(self._func.params, self._func.param_directions):
            param_type = param.type
            if not isinstance(param_type, _ir_core.TensorType):
                raise self._error(
                    f"parameter '{param.name_hint}' has type {param_type}; only tensors are supported"
                )
            if param_type.dtype != DataType.FP32:
                raise self._error(
                    f"parameter '{param.name_hint}' has dtype {param_type.dtype.to_string()}; "
                    "only fp32 is supported"
                )
            extent = _static_extent(list(param_type.shape), f"parameter '{param.name_hint}'", self._func.name)
            base = _NON_IDENTIFIER_RE.sub("_", _SSA_SUFFIX_RE.sub("", param.name_hint)) or "arg"
            name = base
            suffix = 1
            while name in used:
                name = f"{base}_{suffix}"
                suffix += 1
            used.add(name)
            self._param_names[param.unique_id] = name
            self._param_directions[param.unique_id] = direction
            self._param_extents[param.unique_id] = extent
            self._tensor_aliases[param.unique_id] = param.unique_id
            c_params.append(f"{param_type.dtype.to_c_type_string()}* {name}")
        return c_params

    def _param_operand(self, expr: _ir_core.Expr, role: str) -> tuple[int, str, int]:
        """Return (parameter id, C name, extent) of the parameter a tensor operand names."""
        if not isinstance(expr, _ir_core.Var) or expr.unique_id not in self._tensor_aliases:
            raise self._error(f"{role} must be a function parameter, got {expr}")
        param_id = self._tensor_aliases[expr.unique_id]
        return param_id, self._param_names[param_id], self._param_extents[param_id]

    # --------------------------------------------------------------- body ops

    def _use_extent(self, extent: int) -> None:
        if self._extent is None:
            self._extent = extent
        elif extent != self._extent:
            raise self._error(f"all tiles must have the same length, got {self._extent} and {extent}")

    def _tile_operand(self, expr: _ir_core.Expr) -> _TileElement:
        if not isinstance(expr, _ir_core.Var) or expr.unique_id not in self._tiles:
            raise self._error(f"expected a tile produced earlier in the function, got {expr}")
        return self._tiles[expr.unique_id]

    @staticmethod
    def _element(base: str, offset: int) -> str:
        return f"{base}[{_LOOP_INDEX} + {offset}]" if offset else f"{base}[{_LOOP_INDEX}]"

    def _emit_load(self, var: _ir_core.Var, call: _ir_core.Call) -> None:
        # tile.load(tensor, offsets, shapes, valid_shapes)
        tensor, offsets, shapes, valid_shapes = call.args
        _, name, tensor_extent = self._param_operand(tensor, "tile.load source")
        if name in self._store_offsets:
            raise self._error(
                f"tile.load reads '{name}' after a tile.store wrote it; the fused loop cannot order that"
            )
        (offset,) = _const_int_list(offsets, "tile.load offset", self._func.name)
        (extent,) = _const_int_list(shapes, "tile.load shape", self._func.name)
        if _const_int_list(valid_shapes, "tile.load valid shape", self._func.name) != [extent]:
            raise self._error("partially valid tiles are not supported yet; load full tiles")
        if offset < 0 or offset + extent > tensor_extent:
            raise self._error(
                f"tile.load window [{offset}, {offset + extent}) exceeds '{name}' of length {tensor_extent}"
            )
        self._use_extent(extent)
        self._tiles[var.unique_id] = _TileElement(self._element(name, offset), extent)

    def _emit_store(self, var: _ir_core.Var, call: _ir_core.Call) -> None:
        # tile.store(tile, offsets, output_tensor) -> the updated output tensor
        tile, offsets, tensor = call.args
        value = self._tile_operand(tile)
        param_id, name, tensor_extent = self._param_operand(tensor, "tile.store destination")
        if self._param_directions[param_id] == _ir_core.ParamDirection.In:
            raise self._error(f"tile.store writes '{name}', which is an input parameter; declare it pl.Out")
        (offset,) = _const_int_list(offsets, "tile.store offset", self._func.name)
        if offset < 0 or offset + value.extent > tensor_extent:
            raise self._error(
                f"tile.store window [{offset}, {offset + value.extent}) exceeds '{name}' "
                f"of length {tensor_extent}"
            )
        # Fusing reorders stores across iterations: with windows [0, 16) and
        # [8, 24), iteration i + 8 rewrites element i + 8 with the first tile
        # after iteration i stored the second one there. Equal offsets keep
        # program order within each iteration; disjoint windows never meet.
        for previous in self._store_offsets.setdefault(name, []):
            if previous != offset and abs(previous - offset) < value.extent:
                raise self._error(
                    f"tile.store windows at offsets {previous} and {offset} of '{name}' overlap; "
                    "the fused loop cannot keep their order"
                )
        self._store_offsets[name].append(offset)
        self._tensor_aliases[var.unique_id] = param_id
        self._assignments.append(f"{self._element(name, offset)} = {value.expr};")

    def _emit_binary(self, var: _ir_core.Var, call: _ir_core.Call, c_operator: str) -> None:
        lhs, rhs = (self._tile_operand(arg) for arg in call.args)
        if lhs.extent != rhs.extent:
            raise self._error(f"{call.op.name} operands have lengths {lhs.extent} and {rhs.extent}")
        expr = f"{lhs.as_operand()} {c_operator} {rhs.as_operand()}"
        self._tiles[var.unique_id] = _TileElement(expr, lhs.extent, compound=True)

    def _emit_assign(self, stmt: _ir_core.AssignStmt) -> None:
        call = stmt.value
        if not isinstance(call, _ir_core.Call):
            raise self._error(f"unsupported assignment to '{stmt.var.name_hint}': {call}")
        op_name = call.op.name
        if op_name == _LOAD_OP:
            self._emit_load(stmt.var, call)
        elif op_name == _STORE_OP:
            self._emit_store(stmt.var, call)
        elif op_name in _BINARY_C_OPERATORS:
            self._emit_binary(stmt.var, call, _BINARY_C_OPERATORS[op_name])
        else:
            raise self._error(f"operator {op_name} is not supported yet")

    def _check_return(self, stmt: _ir_core.ReturnStmt) -> None:
        # Outputs are written in place, so a return only has to name them.
        for value in stmt.value:
            is_output = isinstance(value, _ir_core.Var) and value.unique_id in self._tensor_aliases
            if not is_output:
                raise self._error(f"return value {value} is not an output tensor")

    def _visit_body(self) -> None:
        body = self._func.body
        stmts = list(body) if isinstance(body, _ir_core.SeqStmts) else [body]
        for stmt in stmts:
            if isinstance(stmt, _ir_core.AssignStmt):
                self._emit_assign(stmt)
            elif isinstance(stmt, _ir_core.ReturnStmt):
                self._check_return(stmt)
            else:
                raise self._error(
                    f"{type(stmt).__name__} is not supported yet; only straight-line tile code is"
                )
        if not self._assignments:
            raise self._error("the function stores no tile, so there is nothing to generate")

    # ------------------------------------------------------------------ output

    @property
    def symbol(self) -> str:
        return f"{self._func.name}_kernel"

    def emit(self) -> str:
        """Return the C source of the kernel."""
        c_params = self._bind_params()
        self._visit_body()
        interleave = "enable" if self._target.interleave else "disable"
        attributes = " __arm_streaming" if self._target.streaming else ""
        lines = [
            f"// Generated by pypto.ir.sve_codegen from InCore function '{self._func.name}'.",
            f"// Processes elements [0, {_LOOP_BOUND}); "
            f"callers pass {_LOOP_BOUND} <= {self._extent}, the tile length.",
            "#include <stdint.h>",
            "",
            f"void {self.symbol}({', '.join([*c_params, f'int64_t {_LOOP_BOUND}'])}){attributes} {{",
            f"    #pragma clang loop vectorize(enable) interleave({interleave})",
            f"    for (int64_t {_LOOP_INDEX} = 0; {_LOOP_INDEX} < {_LOOP_BOUND}; {_LOOP_INDEX}++) {{",
            *(f"        {assignment}" for assignment in self._assignments),
            "    }",
            "}",
        ]
        return "\n".join(lines) + "\n"

    def build_config(self) -> dict[str, object]:
        """Describe how to compile and call the kernel. Call after :meth:`emit`."""
        params: list[dict[str, object]] = []
        for param, direction in zip(self._func.params, self._func.param_directions):
            tensor_type = param.type
            assert isinstance(tensor_type, _ir_core.TensorType)  # checked in _bind_params
            params.append(
                {
                    "name": self._param_names[param.unique_id],
                    "direction": direction.name,
                    "dtype": tensor_type.dtype.to_string(),
                    "shape": [_static_extent(list(tensor_type.shape), "parameter", self._func.name)],
                }
            )
        return {
            "kernel": self._func.name,
            "symbol": self.symbol,
            "source": f"{self._func.name}.c",
            "params": params,
            "n": self._extent,
            "compiler": self._target.compiler,
            "flags": self._target.compile_flags(),
            "streaming": self._target.streaming,
        }


def _cpu_pass_context() -> _passes.PassContext:
    """Build the PassContext for the CPUSVE pipeline.

    Settings come from the active context when there is one (as in
    ``ir.compile``), so callers keep their instruments and verification level.
    ``TileInnermostDimGranularity`` is always disabled: it is an Ascend
    performance hint that queries the configured NPU backend, and the CPU path
    deliberately configures none.
    """
    outer = _passes.PassContext.current()
    cpu_disabled = _passes.DiagnosticCheckSet()
    cpu_disabled.insert(_passes.DiagnosticCheck.TileInnermostDimGranularity)
    if outer is not None:
        return _passes.PassContext(
            list(outer.get_instruments()),
            outer.get_verification_level(),
            outer.get_diagnostic_phase(),
            outer.get_disabled_diagnostics().union_with(cpu_disabled),
            outer.get_memory_planner(),
            outer.get_enable_pypto_l0c_double_buffer(),
            outer.get_runtime(),
            outer.get_enable_buffer_ir(),
        )
    cpu_disabled.insert(_passes.DiagnosticCheck.UnusedControlFlowResult)  # PassContext default
    return _passes.PassContext(
        [],
        _passes.get_default_verification_level(),
        _passes.get_default_diagnostic_phase(),
        cpu_disabled,
    )


def lower(
    program: _ir_core.Program,
    *,
    dump_passes: bool | PassDumpLevel = False,
    passes_dump_dir: str | None = None,
) -> _ir_core.Program:
    """Run the ``OptimizationStrategy.CPUSVE`` pipeline without an NPU backend.

    Args:
        program: Frontend program whose functions are InCore
        dump_passes: Per-pass IR dump control, as in :func:`pypto.ir.compile`
        passes_dump_dir: Dump directory; required when dumping

    Returns:
        The early Tile IR program that :func:`generate` consumes
    """
    with _cpu_pass_context():
        pm = PassManager.get_strategy(OptimizationStrategy.CPUSVE)
        return pm.run_passes(program, dump_ir=dump_passes, output_dir=passes_dump_dir)


def generate(program: _ir_core.Program, target: SVETargetConfig | None = None) -> dict[str, str]:
    """Generate SVE C kernels for every InCore function of an early Tile IR program.

    Args:
        program: Program after the ``OptimizationStrategy.CPUSVE`` pipeline
        target: Compiler target; defaults to 512-bit SVE with clang

    Returns:
        Mapping from output-relative path to file content: one
        ``kernels/sve/<func>.c`` and one ``kernels/sve/<func>.json`` build
        config per InCore function.

    Raises:
        ValueError: If the program contains IR the CPU backend cannot lower yet
    """
    target = target or SVETargetConfig()
    files: dict[str, str] = {}
    for func in program.functions.values():
        if func.func_type != _ir_core.FunctionType.InCore:
            raise ValueError(
                f"SVE codegen: function '{func.name}' is {func.func_type.name}; "
                "only InCore functions are supported"
            )
        if func.ir_stage != _ir_core.FunctionIRStage.Functional:
            raise ValueError(
                f"SVE codegen: function '{func.name}' is at the {func.ir_stage.name} stage; "
                "lower it with OptimizationStrategy.CPUSVE"
            )
        emitter = _KernelEmitter(func, target)
        files[f"{SVE_KERNEL_DIR}/{func.name}.c"] = emitter.emit()
        files[f"{SVE_KERNEL_DIR}/{func.name}.json"] = json.dumps(emitter.build_config(), indent=2) + "\n"
    return files


def compile_to_files(
    program: _ir_core.Program,
    output_dir: str,
    *,
    target: SVETargetConfig | None = None,
    dump_passes: bool | PassDumpLevel = False,
) -> dict[str, str]:
    """Lower a program with the CPUSVE pipeline and write its SVE kernels.

    Args:
        program: Frontend program whose functions are InCore
        output_dir: Directory that receives ``kernels/sve/`` (and
            ``passes_dump/`` when dumping)
        target: Compiler target; defaults to 512-bit SVE with clang
        dump_passes: Per-pass IR dump control, as in :func:`pypto.ir.compile`

    Returns:
        The generated files, keyed by path relative to ``output_dir``
    """
    lowered = lower(program, dump_passes=dump_passes, passes_dump_dir=os.path.join(output_dir, "passes_dump"))
    files = generate(lowered, target)
    for relative_path, content in files.items():
        path = os.path.join(output_dir, relative_path)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
    return files
