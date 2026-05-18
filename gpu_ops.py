#!/usr/bin/env python3
"""
Validate CUDA architecture feature families exposed by the visible GPUs.

This maps compute capability to the CUDA operations that should be available on
that architecture, runs practical runtime tests for library-exposed operations,
and optionally compiles architecture-specific PTX/intrinsic probes that exercise
the documented opcode families introduced for sm_80 and newer targets.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
from dataclasses import asdict, dataclass
from typing import Callable, Optional, Tuple

try:
    import torch
except ImportError as exc:
    torch = None
    TORCH_IMPORT_ERROR = exc
else:
    TORCH_IMPORT_ERROR = None


Capability = Tuple[int, int]


@dataclass(frozen=True)
class Feature:
    feature_id: str
    label: str
    min_cc: Capability
    validator: Optional[str]
    detail: str


@dataclass
class Result:
    device: int
    gpu: str
    cc: str
    feature_id: str
    label: str
    status: str
    detail: str


@dataclass
class CompileResult:
    feature_id: str
    label: str
    validator: str
    arch: str
    status: str
    detail: str


FEATURES = [
    Feature(
        "cuda_runtime",
        "CUDA runtime context",
        (1, 0),
        "runtime_context",
        "Creates a CUDA context and synchronizes the device.",
    ),
    Feature(
        "global_memory_copy",
        "Global memory load/store copy",
        (1, 0),
        "memory_copy",
        "Copies GPU tensor data device-to-device and validates it on GPU.",
    ),
    Feature(
        "cuda_graph",
        "CUDA graph capture/replay",
        (7, 0),
        "cuda_graph",
        "Captures and replays a GPU-only tensor operation.",
    ),
    Feature(
        "cuda_core_fp32",
        "CUDA core FP32 arithmetic",
        (1, 0),
        "fp32_alu",
        "Runs elementwise FP32 fused arithmetic.",
    ),
    Feature(
        "cuda_core_fp64",
        "CUDA core FP64 arithmetic",
        (1, 3),
        "fp64_alu",
        "Runs elementwise FP64 fused arithmetic.",
    ),
    Feature(
        "cuda_core_int32",
        "CUDA core INT32 arithmetic",
        (1, 0),
        "int32_alu",
        "Runs elementwise INT32 arithmetic.",
    ),
    Feature(
        "global_atomic_add",
        "Global atomic add",
        (2, 0),
        "atomic_add",
        "Uses repeated-index index_add, which lowers to CUDA atomics.",
    ),
    Feature(
        "tensor_core_fp16",
        "FP16 Tensor Core GEMM",
        (7, 0),
        "gemm_fp16",
        "Runs FP16 matrix multiply through torch/cuBLAS.",
    ),
    Feature(
        "tensor_core_int8",
        "INT8 Tensor Core GEMM",
        (7, 5),
        "gemm_int8",
        "Runs int8 x int8 -> int32 matrix multiply when PyTorch exposes it.",
    ),
    Feature(
        "ampere_minmax_xorsign_abs",
        "Ampere half/bfloat min/max xorsign.abs",
        (8, 6),
        "compile_minmax_xorsign_abs",
        "Compiles min/max.xorsign.abs half-precision PTX instructions.",
    ),
    Feature(
        "tensor_core_ldmatrix",
        "Tensor Core ldmatrix matrix load",
        (7, 5),
        "compile_ldmatrix",
        "Compiles a warp-level ldmatrix shared-memory load probe.",
    ),
    Feature(
        "tensor_core_tf32",
        "TF32 Tensor Core GEMM",
        (8, 0),
        "gemm_tf32",
        "Runs FP32 GEMM with TF32 enabled.",
    ),
    Feature(
        "tensor_core_fp64_dmma",
        "FP64 Tensor Core DMMA",
        (8, 0),
        "compile_mma_f64",
        "Compiles an sm_80 FP64 mma.sync DMMA probe.",
    ),
    Feature(
        "tensor_core_bf16",
        "BF16 Tensor Core GEMM",
        (8, 0),
        "gemm_bf16",
        "Runs BF16 matrix multiply through torch/cuBLAS.",
    ),
    Feature(
        "tensor_core_sparse_mma",
        "Sparse Tensor Core MMA",
        (8, 0),
        "compile_mma_sparse",
        "Compiles an sm_80 sparse mma.sp::ordered_metadata probe.",
    ),
    Feature(
        "tensor_core_subbyte_mma",
        "Sub-byte INT4 Tensor Core MMA",
        (8, 0),
        "compile_mma_u4",
        "Compiles an sm_80 u4 x u4 MMA probe.",
    ),
    Feature(
        "tensor_core_binary_mma",
        "Binary Tensor Core BMMA",
        (8, 0),
        "compile_mma_b1",
        "Compiles an sm_80 b1 x b1 BMMA probe.",
    ),
    Feature(
        "ampere_cp_async",
        "Ampere cp.async shared-memory copy",
        (8, 0),
        "compile_cp_async",
        "Compiles an sm_80+ inline-PTX cp.async probe.",
    ),
    Feature(
        "ampere_l2_priority_discard",
        "Ampere L2 applypriority/discard",
        (8, 0),
        "compile_l2_priority_discard",
        "Compiles sm_80 L2 applypriority and discard instructions.",
    ),
    Feature(
        "ampere_mbarrier",
        "Ampere shared-memory mbarrier",
        (8, 0),
        "compile_mbarrier",
        "Compiles sm_80 mbarrier init/invalidate instructions.",
    ),
    Feature(
        "ampere_mbarrier_arrive_wait",
        "Ampere mbarrier arrive/test/pending count",
        (8, 0),
        "compile_mbarrier_arrive_wait",
        "Compiles mbarrier arrive, test_wait, and pending_count instructions.",
    ),
    Feature(
        "ampere_warp_redux",
        "Ampere warp-wide redux.sync",
        (8, 0),
        "compile_redux_sync",
        "Compiles sm_80 warp-wide integer reduction instructions.",
    ),
    Feature(
        "ampere_l2_cache_hint",
        "Ampere L2 cache hint / residency policy",
        (8, 0),
        "compile_l2_cache_hint",
        "Compiles sm_80 createpolicy and L2 cache-hint load instructions.",
    ),
    Feature(
        "tensor_core_fp8",
        "FP8 Tensor Core GEMM",
        (8, 9),
        "gemm_fp8",
        "Runs FP8 E4M3 cuBLASLt GEMM when hardware and cuBLASLt support it.",
    ),
    Feature(
        "tensor_core_fp8_mma",
        "FP8 PTX mma.sync",
        (8, 9),
        "compile_mma_fp8",
        "Compiles e4m3/e5m2 mma.sync PTX for sm_89+.",
    ),
    Feature(
        "fp8_conversion",
        "FP8 conversion instructions",
        (8, 9),
        "compile_cvt_fp8",
        "Compiles e4m3/e5m2 conversion instructions for sm_89+.",
    ),
    Feature(
        "hopper_dpx",
        "Hopper DPX intrinsics",
        (9, 0),
        "compile_dpx",
        "Compiles CUDA DPX intrinsic probes such as __vimax3_s32.",
    ),
    Feature(
        "hopper_cluster_addressing",
        "Hopper cluster shared-memory addressing",
        (9, 0),
        "compile_cluster_map_rank",
        "Compiles mapa.shared::cluster and getctarank probes.",
    ),
    Feature(
        "hopper_cluster_registers",
        "Hopper cluster special registers",
        (9, 0),
        "compile_cluster_registers",
        "Compiles reads of cluster id/rank and aggregate shared-memory registers.",
    ),
    Feature(
        "hopper_cluster_barrier",
        "Hopper cluster barrier",
        (9, 0),
        "compile_cluster_barrier",
        "Compiles barrier.cluster arrive/wait instructions.",
    ),
    Feature(
        "hopper_elect_sync",
        "Hopper elect.sync",
        (9, 0),
        "compile_elect_sync",
        "Compiles warp leader election instruction.",
    ),
    Feature(
        "hopper_mbarrier_tx",
        "Hopper mbarrier transaction accounting",
        (9, 0),
        "compile_mbarrier_tx",
        "Compiles mbarrier expect_tx, complete_tx, and try_wait instructions.",
    ),
    Feature(
        "hopper_stmatrix",
        "Hopper stmatrix matrix store",
        (9, 0),
        "compile_stmatrix",
        "Compiles a warp-level stmatrix shared-memory store probe.",
    ),
    Feature(
        "hopper_wgmma",
        "Hopper warp-group MMA",
        (9, 0),
        "compile_wgmma",
        "Compiles sm_90a wgmma.mma_async plus group-control instructions.",
    ),
    Feature(
        "hopper_tma_cp_async_bulk",
        "Hopper TMA / cp.async.bulk",
        (9, 0),
        "compile_cp_async_bulk",
        "Compiles sm_90 cp.async.bulk copy and prefetch probes.",
    ),
    Feature(
        "hopper_cp_async_bulk_tensor",
        "Hopper tensor-map cp.async.bulk.tensor",
        (9, 0),
        "compile_cp_async_bulk_tensor",
        "Compiles tensor-map bulk async copy and prefetch instructions.",
    ),
    Feature(
        "hopper_cp_reduce_async_bulk",
        "Hopper cp.reduce.async.bulk",
        (9, 0),
        "compile_cp_reduce_async_bulk",
        "Compiles sm_90 async bulk reduce-copy instruction.",
    ),
    Feature(
        "hopper_cp_reduce_async_bulk_tensor",
        "Hopper tensor-map cp.reduce.async.bulk.tensor",
        (9, 0),
        "compile_cp_reduce_async_bulk_tensor",
        "Compiles tensor-map async bulk reduce-copy instructions.",
    ),
    Feature(
        "hopper_tensormap_replace",
        "Hopper tensor-map replace",
        (9, 0),
        "compile_tensormap_replace",
        "Compiles architecture-specific tensormap.replace instructions.",
    ),
    Feature(
        "hopper_tensormap_proxy",
        "Hopper tensor-map proxy fence",
        (9, 0),
        "compile_tensormap_proxy",
        "Compiles tensormap.cp_fenceproxy and fence.proxy.tensormap instructions.",
    ),
    Feature(
        "hopper_multimem",
        "Hopper multimem operations",
        (9, 0),
        "compile_multimem",
        "Compiles multimem load-reduce, store, and reduction instructions.",
    ),
    Feature(
        "hopper_grid_dependency_control",
        "Hopper grid dependency control",
        (9, 0),
        "compile_griddepcontrol",
        "Compiles griddepcontrol.wait and launch_dependents instructions.",
    ),
    Feature(
        "hopper_thread_block_cluster",
        "Hopper thread-block clusters / DSM",
        (9, 0),
        None,
        "Mapped to sm_90. Runtime validation needs a cluster-launched kernel.",
    ),
    Feature(
        "blackwell_tcgen05",
        "Blackwell tcgen05 alloc/dealloc",
        (10, 0),
        "compile_tcgen05",
        "Compiles tcgen05.alloc, dealloc, and relinquish_alloc_permit.",
    ),
    Feature(
        "blackwell_tcgen05_ldst_wait",
        "Blackwell tcgen05 load/store/wait",
        (10, 0),
        "compile_tcgen05_ldst_wait",
        "Compiles tcgen05.ld, tcgen05.st, and tcgen05.wait instructions.",
    ),
    Feature(
        "blackwell_tcgen05_cp_shift",
        "Blackwell tcgen05 copy/shift",
        (10, 0),
        "compile_tcgen05_cp_shift",
        "Compiles tcgen05.cp and tcgen05.shift instructions.",
    ),
    Feature(
        "blackwell_tcgen05_mma",
        "Blackwell tcgen05 MMA",
        (10, 0),
        "compile_tcgen05_mma",
        "Compiles dense and sparse tcgen05.mma instructions.",
    ),
    Feature(
        "blackwell_tcgen05_mma_ws",
        "Blackwell tcgen05 weight-stationary MMA",
        (10, 0),
        "compile_tcgen05_mma_ws",
        "Compiles dense and sparse tcgen05.mma.ws instructions.",
    ),
    Feature(
        "blackwell_tcgen05_fence_commit",
        "Blackwell tcgen05 fence/commit",
        (10, 0),
        "compile_tcgen05_fence_commit",
        "Compiles tcgen05.fence and tcgen05.commit instructions.",
    ),
    Feature(
        "blackwell_st_bulk",
        "Blackwell st.bulk",
        (10, 0),
        "compile_st_bulk",
        "Compiles sm_100 bulk shared-memory store instruction.",
    ),
    Feature(
        "blackwell_stmatrix_b8",
        "Blackwell stmatrix b8 shape",
        (10, 0),
        "compile_stmatrix_b8",
        "Compiles sm_100a stmatrix m16n8 b8 matrix-store instructions.",
    ),
    Feature(
        "blackwell_tensormap_replace_ext",
        "Blackwell tensor-map replace extensions",
        (10, 0),
        "compile_tensormap_replace_ext",
        "Compiles sm_100a tensormap.replace field extensions.",
    ),
    Feature(
        "blackwell_small_float_conversion",
        "Blackwell small-float conversion instructions",
        (10, 0),
        "compile_cvt_small_float",
        "Compiles e2m3/e3m2/ue8m0 conversion instructions.",
    ),
    Feature(
        "blackwell_cluster_launch_control",
        "Blackwell cluster launch control",
        (10, 0),
        "compile_clusterlaunchcontrol",
        "Compiles clusterlaunchcontrol.try_cancel and query_cancel instructions.",
    ),
    Feature(
        "blackwell_tensor_memory",
        "Blackwell tensor memory",
        (10, 0),
        None,
        "Mapped to sm_100a-family features. Runtime validation needs a tcgen05 kernel.",
    ),
    Feature(
        "blackwell_f8f6f4_mma",
        "Blackwell f8/f6/f4 MMA",
        (12, 0),
        "compile_mma_f8f6f4",
        "Compiles sm_120a dense and sparse f8/f6/f4 MMA probes.",
    ),
]


def cc_at_least(actual: Capability, required: Capability) -> bool:
    return actual[0] > required[0] or (
        actual[0] == required[0] and actual[1] >= required[1])


def require_torch():
    if torch is not None:
        return
    message = str(TORCH_IMPORT_ERROR)
    if "ncclCommResume" in message:
        raise SystemExit(
            "PyTorch failed to import because libtorch_cuda.so is loading an "
            "NCCL runtime without ncclCommResume. This is a PyTorch/NCCL "
            "install or LD_LIBRARY_PATH mismatch, not a gpu_ops.py failure."
        ) from TORCH_IMPORT_ERROR
    raise SystemExit(f"PyTorch failed to import: {message}") from TORCH_IMPORT_ERROR


def status_result(device, props, feature, status, detail):
    return Result(
        device=device,
        gpu=props.name,
        cc=f"{props.major}.{props.minor}",
        feature_id=feature.feature_id,
        label=feature.label,
        status=status,
        detail=detail,
    )


def make_input(size, dtype, device):
    values = torch.arange(size * size, device=device, dtype=torch.float32)
    values = (values.reshape(size, size) % 17 - 8) / 64.0
    return values.to(dtype)


def validate_close(value, reference, tol):
    diff = (value.float() - reference.float()).abs()
    max_diff = float(diff.max().item())
    if max_diff > tol:
        raise RuntimeError(f"max_diff={max_diff:.6g} > tol={tol:.6g}")
    return max_diff


def test_runtime_context(device, args):
    torch.cuda.synchronize(device)
    return "context synchronized"


def test_memory_copy(device, args):
    src = torch.arange(args.elements, device=device, dtype=torch.uint8)
    dst = torch.empty_like(src)
    dst.copy_(src)
    bad = torch.count_nonzero(src != dst)
    torch.cuda.synchronize(device)
    errors = int(bad.item())
    if errors:
        raise RuntimeError(f"errors={errors}")
    return f"{args.elements} bytes copied, errors=0"


def test_cuda_graph(device, args):
    x = torch.ones(args.elements, device=device, dtype=torch.float32)
    y = torch.empty_like(x)
    stream = torch.cuda.Stream(device=device)
    with torch.cuda.stream(stream):
        y.copy_(x)
        y.add_(1.0)
    stream.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        y.copy_(x)
        y.add_(1.0)
    graph.replay()
    bad = torch.count_nonzero(y != 2.0)
    torch.cuda.synchronize(device)
    errors = int(bad.item())
    if errors:
        raise RuntimeError(f"errors={errors}")
    return "capture/replay ok"


def test_fp32_alu(device, args):
    x = torch.linspace(-1.0, 1.0, args.elements, device=device)
    y = x.mul(1.25).add(0.5)
    ref = x.cpu().float().mul(1.25).add(0.5).to(device)
    max_diff = validate_close(y, ref, 1e-6)
    return f"max_diff={max_diff:.3g}"


def test_fp64_alu(device, args):
    x = torch.linspace(-1.0, 1.0, args.elements, device=device,
                       dtype=torch.float64)
    y = x.mul(1.25).add(0.5)
    ref = x.cpu().mul(1.25).add(0.5).to(device)
    max_diff = float((y - ref).abs().max().item())
    if max_diff > 1e-12:
        raise RuntimeError(f"max_diff={max_diff:.6g}")
    return f"max_diff={max_diff:.3g}"


def test_int32_alu(device, args):
    x = torch.arange(args.elements, device=device, dtype=torch.int32)
    y = x.mul(3).add(7)
    ref = torch.arange(args.elements, device=device, dtype=torch.int32).mul(3).add(7)
    bad = torch.count_nonzero(y != ref)
    torch.cuda.synchronize(device)
    errors = int(bad.item())
    if errors:
        raise RuntimeError(f"errors={errors}")
    return "errors=0"


def test_atomic_add(device, args):
    count = max(1024, args.elements)
    buckets = 64
    dst = torch.zeros(buckets, device=device, dtype=torch.float32)
    index = torch.arange(count, device=device, dtype=torch.int64) % buckets
    src = torch.ones(count, device=device, dtype=torch.float32)
    dst.index_add_(0, index, src)
    expected = torch.bincount(index, minlength=buckets).float()
    max_diff = validate_close(dst, expected, 0.0)
    return f"max_diff={max_diff:.3g}"


def test_gemm(dtype, device, args, tol, allow_tf32=False):
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    previous_precision = torch.get_float32_matmul_precision()
    try:
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        torch.set_float32_matmul_precision("high" if allow_tf32 else "highest")
        a = make_input(args.size, dtype, device)
        b = make_input(args.size, dtype, device).t().contiguous()
        out = torch.mm(a, b)
        ref = torch.mm(a.float(), b.float())
        torch.cuda.synchronize(device)
        max_diff = validate_close(out, ref, tol)
        return f"n={args.size}, max_diff={max_diff:.3g}"
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32
        torch.set_float32_matmul_precision(previous_precision)


def test_gemm_fp16(device, args):
    return test_gemm(torch.float16, device, args, 0.25)


def test_gemm_bf16(device, args):
    return test_gemm(torch.bfloat16, device, args, 0.75)


def test_gemm_tf32(device, args):
    return test_gemm(torch.float32, device, args, 0.25, allow_tf32=True)


def test_gemm_int8(device, args):
    if not hasattr(torch, "_int_mm"):
        raise RuntimeError("torch._int_mm is not available")
    n = max(32, args.size)
    n = ((n + 31) // 32) * 32
    a = (torch.arange(n * n, device=device, dtype=torch.int32)
         .reshape(n, n).remainder(7).sub(3).to(torch.int8))
    b = torch.eye(n, device=device, dtype=torch.int8)
    out = torch._int_mm(a, b)
    ref = a.to(torch.int32)
    bad = torch.count_nonzero(out != ref)
    torch.cuda.synchronize(device)
    errors = int(bad.item())
    if errors:
        raise RuntimeError(f"errors={errors}")
    return f"n={n}, errors=0"


def test_gemm_fp8(device, args):
    try:
        from gpu_burn import CUDA_R_8F_E4M3, Fp8LtMatmul
    except Exception as exc:
        raise RuntimeError(f"could not import gpu_burn FP8 helper: {exc}") from exc
    if not hasattr(torch, "float8_e4m3fn"):
        raise RuntimeError("torch.float8_e4m3fn is not available")

    n = max(128, args.size)
    n = ((n + 127) // 128) * 128
    a = make_input(n, torch.float32, device).to(torch.float8_e4m3fn)
    b = make_input(n, torch.float32, device).to(torch.float8_e4m3fn)
    out = torch.empty((n, n), device=device, dtype=torch.float32)
    ref = torch.empty_like(out)
    op = Fp8LtMatmul(n, CUDA_R_8F_E4M3, device)
    try:
        op(a, b, out)
        op(a, b, ref)
        torch.cuda.synchronize(device)
        max_diff = validate_close(out, ref, 0.0)
        nonfinite = int(torch.count_nonzero(~torch.isfinite(out)).item())
        if nonfinite:
            raise RuntimeError(f"nonfinite={nonfinite}")
        return f"n={n}, deterministic_max_diff={max_diff:.3g}"
    finally:
        op.close()


VALIDATORS: dict[str, Callable] = {
    "runtime_context": test_runtime_context,
    "memory_copy": test_memory_copy,
    "cuda_graph": test_cuda_graph,
    "fp32_alu": test_fp32_alu,
    "fp64_alu": test_fp64_alu,
    "int32_alu": test_int32_alu,
    "atomic_add": test_atomic_add,
    "gemm_fp16": test_gemm_fp16,
    "gemm_int8": test_gemm_int8,
    "gemm_tf32": test_gemm_tf32,
    "gemm_bf16": test_gemm_bf16,
    "gemm_fp8": test_gemm_fp8,
}


COMPILE_CACHE: dict[str, str] = {}


COMPILE_PROBES = {
    "compile_minmax_xorsign_abs": (
        "sm_86",
        r"""
        #include <cuda_runtime.h>
        extern "C" __global__ void probe(unsigned short* out) {
            unsigned short a = 0x3c00u, b = 0xbc00u, d0, d1;
            asm volatile("min.xorsign.abs.f16 %0, %1, %2;"
                         : "=h"(d0) : "h"(a), "h"(b));
            asm volatile("max.xorsign.abs.f16 %0, %1, %2;"
                         : "=h"(d1) : "h"(a), "h"(b));
            if (threadIdx.x == 0) out[0] = d0 + d1;
        }
        """,
    ),
    "compile_ldmatrix": (
        "sm_80",
        r"""
        #include <cuda_runtime.h>
        extern "C" __global__ void probe(int* out) {
            __shared__ int smem[64];
            unsigned addr = static_cast<unsigned>(__cvta_generic_to_shared(smem));
            unsigned d;
            asm volatile("ldmatrix.sync.aligned.m8n8.x1.shared::cta.b16 {%0}, [%1];"
                         : "=r"(d) : "r"(addr));
            if (threadIdx.x == 0) out[0] = d;
        }
        """,
    ),
    "compile_mma_f64": (
        "sm_80",
        r"""
        #include <cuda_runtime.h>
        extern "C" __global__ void probe(double* out) {
            double a = 1.0, b = 2.0, c0 = 0.0, c1 = 0.0, d0, d1;
            asm volatile(
                "mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 "
                "{%0, %1}, {%2}, {%3}, {%4, %5};"
                : "=d"(d0), "=d"(d1)
                : "d"(a), "d"(b), "d"(c0), "d"(c1));
            if (threadIdx.x == 0) out[0] = d0 + d1;
        }
        """,
    ),
    "compile_mma_sparse": (
        "sm_80",
        r"""
        #include <cuda_runtime.h>
        extern "C" __global__ void probe(int* out) {
            unsigned a0 = 0, a1 = 0, b0 = 0, b1 = 0, e = 0;
            int c0 = 0, c1 = 0, c2 = 0, c3 = 0, d0, d1, d2, d3;
            asm volatile(
                "mma.sp::ordered_metadata.sync.aligned.m16n8k32.row.col"
                ".satfinite.s32.u8.u8.s32 "
                "{%0,%1,%2,%3}, {%4,%5}, {%6,%7}, {%8,%9,%10,%11}, %12, 0x1;"
                : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
                : "r"(a0), "r"(a1), "r"(b0), "r"(b1),
                  "r"(c0), "r"(c1), "r"(c2), "r"(c3), "r"(e));
            if (threadIdx.x == 0) out[0] = d0 + d1 + d2 + d3;
        }
        """,
    ),
    "compile_mma_u4": (
        "sm_80",
        r"""
        #include <cuda_runtime.h>
        extern "C" __global__ void probe(int* out) {
            unsigned a = 0x11111111u, b = 0x22222222u;
            int c0 = 0, c1 = 0, d0, d1;
            asm volatile(
                "mma.sync.aligned.m8n8k32.row.col.satfinite.s32.u4.u4.s32 "
                "{%0, %1}, {%2}, {%3}, {%4, %5};"
                : "=r"(d0), "=r"(d1)
                : "r"(a), "r"(b), "r"(c0), "r"(c1));
            if (threadIdx.x == 0) out[0] = d0 + d1;
        }
        """,
    ),
    "compile_mma_b1": (
        "sm_80",
        r"""
        #include <cuda_runtime.h>
        extern "C" __global__ void probe(int* out) {
            unsigned a = 0xffffffffu, b = 0x0f0f0f0fu;
            int c0 = 0, c1 = 0, d0, d1;
            asm volatile(
                "mma.sync.aligned.m8n8k128.row.col.s32.b1.b1.s32.and.popc "
                "{%0, %1}, {%2}, {%3}, {%4, %5};"
                : "=r"(d0), "=r"(d1)
                : "r"(a), "r"(b), "r"(c0), "r"(c1));
            if (threadIdx.x == 0) out[0] = d0 + d1;
        }
        """,
    ),
    "compile_mma_fp8": (
        "sm_89",
        r"""
        #include <cuda_runtime.h>
        extern "C" __global__ void probe(int* out) {
            unsigned a0 = 0, a1 = 0, a2 = 0, a3 = 0, b0 = 0, b1 = 0;
            float c0 = 0, c1 = 0, c2 = 0, c3 = 0, d0, d1, d2, d3;
            asm volatile(
                "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e5m2.f32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};"
                : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
                : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
                  "f"(c0), "f"(c1), "f"(c2), "f"(c3));
            if (threadIdx.x == 0) out[0] = static_cast<int>(d0 + d1 + d2 + d3);
        }
        """,
    ),
    "compile_cvt_fp8": (
        "sm_89",
        r"""
        #include <cuda_runtime.h>
        extern "C" __global__ void probe(unsigned short* out) {
            float a = 1.0f, b = -1.0f;
            unsigned short e4, e5;
            asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;"
                         : "=h"(e4) : "f"(a), "f"(b));
            asm volatile("cvt.rn.satfinite.e5m2x2.f32 %0, %1, %2;"
                         : "=h"(e5) : "f"(a), "f"(b));
            if (threadIdx.x == 0) out[0] = e4 + e5;
        }
        """,
    ),
    "compile_cp_async": (
        "sm_80",
        r"""
        #include <cuda_runtime.h>
        extern "C" __global__ void probe(const int* src, int* dst) {
            extern __shared__ int smem[];
            unsigned smem_addr = static_cast<unsigned>(__cvta_generic_to_shared(smem));
            const void* gmem_addr = src;
            asm volatile("cp.async.ca.shared.global [%0], [%1], 16;"
                         :: "r"(smem_addr), "l"(gmem_addr));
            asm volatile("cp.async.commit_group;");
            asm volatile("cp.async.wait_group 0;");
            __syncthreads();
            if (threadIdx.x == 0) dst[0] = smem[0];
        }
        """,
    ),
    "compile_l2_priority_discard": (
        "sm_80",
        r"""
        #include <cuda_runtime.h>
        extern "C" __global__ void probe(int* ptr) {
            asm volatile("applypriority.global.L2::evict_normal [%0], 128;"
                         :: "l"(ptr));
            asm volatile("discard.global.L2 [%0], 128;" :: "l"(ptr));
        }
        """,
    ),
    "compile_mbarrier": (
        "sm_80",
        r"""
        #include <cuda_runtime.h>
        extern "C" __global__ void probe(int* out) {
            __shared__ unsigned long long barrier;
            unsigned addr = static_cast<unsigned>(__cvta_generic_to_shared(&barrier));
            if (threadIdx.x == 0) {
                asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;"
                             :: "r"(addr), "r"(1));
                asm volatile("mbarrier.inval.shared::cta.b64 [%0];"
                             :: "r"(addr));
                out[0] = 1;
            }
        }
        """,
    ),
    "compile_mbarrier_arrive_wait": (
        "sm_80",
        r"""
        #include <cuda_runtime.h>
        extern "C" __global__ void probe(int* out) {
            __shared__ unsigned long long barrier;
            unsigned addr = static_cast<unsigned>(__cvta_generic_to_shared(&barrier));
            unsigned long long state;
            unsigned count, done;
            if (threadIdx.x == 0) {
                asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;"
                             :: "r"(addr), "r"(1));
            }
            __syncthreads();
            asm volatile("mbarrier.arrive.shared::cta.b64 %0, [%1];"
                         : "=l"(state) : "r"(addr));
            asm volatile(
                "{ .reg .pred p; "
                "mbarrier.test_wait.shared::cta.b64 p, [%1], %2; "
                "selp.u32 %0, 1, 0, p; }"
                : "=r"(done) : "r"(addr), "l"(state));
            asm volatile("mbarrier.pending_count.b64 %0, %1;"
                         : "=r"(count) : "l"(state));
            if (threadIdx.x == 0) out[0] = done + count;
        }
        """,
    ),
    "compile_redux_sync": (
        "sm_80",
        r"""
        #include <cuda_runtime.h>
        extern "C" __global__ void probe(int* out) {
            int src = threadIdx.x;
            int dst;
            asm volatile("redux.sync.add.s32 %0, %1, 0xffffffff;"
                         : "=r"(dst) : "r"(src));
            if (threadIdx.x == 0) out[0] = dst;
        }
        """,
    ),
    "compile_l2_cache_hint": (
        "sm_80",
        r"""
        #include <cuda_runtime.h>
        extern "C" __global__ void probe(const int* src, int* dst) {
            unsigned long long policy;
            int value;
            asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;"
                         : "=l"(policy));
            asm volatile("ld.global.L2::cache_hint.b32 %0, [%1], %2;"
                         : "=r"(value) : "l"(src), "l"(policy));
            if (threadIdx.x == 0) dst[0] = value;
        }
        """,
    ),
    "compile_dpx": (
        "sm_90",
        r"""
        #include <cuda_runtime.h>
        #include <cuda_fp16.h>
        extern "C" __global__ void probe(int* out, int a, int b, int c) {
            if (threadIdx.x == 0) {
                int x = __vimax3_s32(a, b, c);
                int y = __viaddmax_s32(a, b, c);
                out[0] = x + y;
            }
        }
        """,
    ),
    "compile_cluster_map_rank": (
        "sm_90",
        r"""
        #include <cuda_runtime.h>
        extern "C" __global__ void probe(unsigned* out) {
            extern __shared__ int smem[];
            unsigned addr = static_cast<unsigned>(__cvta_generic_to_shared(smem));
            unsigned mapped, rank;
            asm volatile("mapa.shared::cluster.u32 %0, %1, 0;"
                         : "=r"(mapped) : "r"(addr));
            asm volatile("getctarank.shared::cluster.u32 %0, %1;"
                         : "=r"(rank) : "r"(mapped));
            if (threadIdx.x == 0) out[0] = mapped + rank;
        }
        """,
    ),
    "compile_cluster_registers": (
        "sm_90",
        r"""
        #include <cuda_runtime.h>
        extern "C" __global__ void probe(int* out) {
            unsigned a, b, c, d, e, f, g;
            asm volatile("mov.u32 %0, %%clusterid.x;" : "=r"(a));
            asm volatile("mov.u32 %0, %%nclusterid.x;" : "=r"(b));
            asm volatile("mov.u32 %0, %%cluster_ctaid.x;" : "=r"(c));
            asm volatile("mov.u32 %0, %%cluster_nctaid.x;" : "=r"(d));
            asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(e));
            asm volatile("mov.u32 %0, %%cluster_nctarank;" : "=r"(f));
            asm volatile("mov.u32 %0, %%aggr_smem_size;" : "=r"(g));
            if (threadIdx.x == 0) out[0] = a + b + c + d + e + f + g;
        }
        """,
    ),
    "compile_cluster_barrier": (
        "sm_90",
        r"""
        #include <cuda_runtime.h>
        extern "C" __global__ void probe(int* out) {
            asm volatile("barrier.cluster.arrive;\n\tbarrier.cluster.wait;");
            if (threadIdx.x == 0) out[0] = 1;
        }
        """,
    ),
    "compile_elect_sync": (
        "sm_90",
        r"""
        #include <cuda_runtime.h>
        extern "C" __global__ void probe(int* out) {
            unsigned r, p;
            asm volatile(
                "{ .reg .pred q; elect.sync %0|q, 0xffffffff; "
                "selp.u32 %1, 1, 0, q; }"
                : "=r"(r), "=r"(p));
            if (threadIdx.x == 0) out[0] = r + p;
        }
        """,
    ),
    "compile_mbarrier_tx": (
        "sm_90",
        r"""
        #include <cuda_runtime.h>
        extern "C" __global__ void probe(int* out) {
            __shared__ unsigned long long barrier;
            unsigned addr = static_cast<unsigned>(__cvta_generic_to_shared(&barrier));
            unsigned long long state;
            unsigned done;
            if (threadIdx.x == 0) {
                asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;"
                             :: "r"(addr), "r"(1));
                asm volatile("mbarrier.expect_tx.shared::cta.b64 [%0], %1;"
                             :: "r"(addr), "r"(16));
                asm volatile("mbarrier.complete_tx.shared::cta.b64 [%0], %1;"
                             :: "r"(addr), "r"(16));
                out[0] = 1;
            }
            __syncthreads();
            asm volatile("mbarrier.arrive.shared::cta.b64 %0, [%1];"
                         : "=l"(state) : "r"(addr));
            asm volatile(
                "{ .reg .pred p; "
                "mbarrier.try_wait.shared::cta.b64 p, [%1], %2; "
                "selp.u32 %0, 1, 0, p; }"
                : "=r"(done) : "r"(addr), "l"(state));
            if (threadIdx.x == 0) out[0] += done;
        }
        """,
    ),
    "compile_stmatrix": (
        "sm_90",
        r"""
        #include <cuda_runtime.h>
        extern "C" __global__ void probe(int* out) {
            __shared__ int smem[64];
            unsigned addr = static_cast<unsigned>(__cvta_generic_to_shared(smem));
            unsigned r = 0;
            asm volatile("stmatrix.sync.aligned.m8n8.x1.shared.b16 [%0], {%1};"
                         :: "r"(addr), "r"(r));
            if (threadIdx.x == 0) out[0] = 1;
        }
        """,
    ),
    "compile_cp_async_bulk": (
        "sm_90",
        r"""
        #include <cuda_runtime.h>
        extern "C" __global__ void probe(const int* src, int* out) {
            __shared__ unsigned long long barrier;
            __shared__ int smem[64];
            unsigned barrier_addr =
                static_cast<unsigned>(__cvta_generic_to_shared(&barrier));
            unsigned smem_addr =
                static_cast<unsigned>(__cvta_generic_to_shared(smem));
            if (threadIdx.x == 0) {
                asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;"
                             :: "r"(barrier_addr), "r"(1));
            }
            __syncthreads();
            asm volatile(
                "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes "
                "[%0], [%1], %2, [%3];"
                :: "r"(smem_addr), "l"(src), "r"(128), "r"(barrier_addr));
            asm volatile("cp.async.bulk.prefetch.L2.global [%0], 128;" :: "l"(src));
            if (threadIdx.x == 0) out[0] = smem[0];
        }
        """,
    ),
    "compile_cp_async_bulk_tensor": (
        "sm_90",
        r"""
        #include <cuda_runtime.h>
        extern "C" __global__ void probe(const void* tensor_map, int* out) {
            __shared__ unsigned long long barrier;
            __shared__ int smem[64];
            unsigned barrier_addr =
                static_cast<unsigned>(__cvta_generic_to_shared(&barrier));
            unsigned smem_addr =
                static_cast<unsigned>(__cvta_generic_to_shared(smem));
            int coord = 0;
            if (threadIdx.x == 0) {
                asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;"
                             :: "r"(barrier_addr), "r"(1));
            }
            __syncthreads();
            asm volatile(
                "cp.async.bulk.tensor.1d.shared::cta.global"
                ".mbarrier::complete_tx::bytes.tile "
                "[%0], [%1, {%2}], [%3];"
                :: "r"(smem_addr), "l"(tensor_map), "r"(coord),
                   "r"(barrier_addr));
            asm volatile(
                "cp.async.bulk.prefetch.tensor.1d.L2.global.tile [%0, {%1}];"
                :: "l"(tensor_map), "r"(coord));
            if (threadIdx.x == 0) out[0] = smem[0];
        }
        """,
    ),
    "compile_cp_reduce_async_bulk": (
        "sm_90",
        r"""
        #include <cuda_runtime.h>
        extern "C" __global__ void probe(int* dst) {
            extern __shared__ int smem[];
            unsigned smem_addr =
                static_cast<unsigned>(__cvta_generic_to_shared(smem));
            asm volatile(
                "cp.reduce.async.bulk.global.shared::cta.bulk_group.add.u32 "
                "[%0], [%1], %2;"
                :: "l"(dst), "r"(smem_addr), "r"(128));
        }
        """,
    ),
    "compile_cp_reduce_async_bulk_tensor": (
        "sm_90",
        r"""
        #include <cuda_runtime.h>
        extern "C" __global__ void probe(const void* tensor_map) {
            __shared__ int smem[64];
            unsigned smem_addr =
                static_cast<unsigned>(__cvta_generic_to_shared(smem));
            int coord = 0;
            asm volatile(
                "cp.reduce.async.bulk.tensor.1d.global.shared::cta"
                ".add.tile.bulk_group [%0, {%1}], [%2];"
                :: "l"(tensor_map), "r"(coord), "r"(smem_addr));
            asm volatile("cp.async.bulk.commit_group;");
            asm volatile("cp.async.bulk.wait_group 0;");
        }
        """,
    ),
    "compile_tensormap_replace": (
        "sm_90a",
        r"""
        #include <cuda_runtime.h>
        extern "C" __global__ void probe(int* out) {
            __shared__ unsigned tensor_map[32];
            unsigned addr =
                static_cast<unsigned>(__cvta_generic_to_shared(tensor_map));
            unsigned long long new_val = 0;
            asm volatile(
                "tensormap.replace.tile.global_address.shared::cta.b1024.b64 "
                "[%0], %1;"
                :: "r"(addr), "l"(new_val));
            if (threadIdx.x == 0) out[0] = 1;
        }
        """,
    ),
    "compile_tensormap_proxy": (
        "sm_90",
        r"""
        #include <cuda_runtime.h>
        extern "C" __global__ void probe(unsigned* dst) {
            __shared__ unsigned tensor_map[32];
            unsigned src =
                static_cast<unsigned>(__cvta_generic_to_shared(tensor_map));
            asm volatile(
                "tensormap.cp_fenceproxy.global.shared::cta"
                ".tensormap::generic.release.gpu.sync.aligned "
                "[%0], [%1], 128;"
                :: "l"(dst), "r"(src));
            asm volatile(
                "fence.proxy.tensormap::generic.acquire.gpu [%0], 128;"
                :: "l"(dst));
        }
        """,
    ),
    "compile_multimem": (
        "sm_90",
        r"""
        #include <cuda_runtime.h>
        extern "C" __global__ void probe(int* src, int* dst) {
            int v;
            asm volatile("multimem.ld_reduce.relaxed.sys.global.add.u32 %0, [%1];"
                         : "=r"(v) : "l"(src));
            asm volatile("multimem.st.relaxed.sys.global.u32 [%0], %1;"
                         :: "l"(dst), "r"(v));
            asm volatile("multimem.red.relaxed.sys.global.add.u32 [%0], %1;"
                         :: "l"(dst), "r"(v));
        }
        """,
    ),
    "compile_griddepcontrol": (
        "sm_90",
        r"""
        #include <cuda_runtime.h>
        extern "C" __global__ void probe(int* out) {
            asm volatile("griddepcontrol.wait;");
            asm volatile("griddepcontrol.launch_dependents;");
            if (threadIdx.x == 0) out[0] = 1;
        }
        """,
    ),
    "compile_wgmma": (
        "sm_90a",
        r"""
        #include <cuda_runtime.h>
        extern "C" __global__ void probe() {
            unsigned a0 = 0, a1 = 0, a2 = 0, a3 = 0;
            unsigned long long desc_b = 0;
            float d0 = 0, d1 = 0, d2 = 0, d3 = 0;
            asm volatile("wgmma.fence.sync.aligned;");
            asm volatile(
                "wgmma.mma_async.sync.aligned.m64n8k16.f32.f16.f16 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, %8, 1, -1, -1, 1;"
                : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
                : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc_b));
            asm volatile("wgmma.commit_group.sync.aligned;");
            asm volatile("wgmma.wait_group.sync.aligned 0;");
        }
        """,
    ),
    "compile_tcgen05": (
        "sm_100a",
        r"""
        #include <cuda_runtime.h>
        extern "C" __global__ void probe(int* out) {
            __shared__ unsigned taddr_slot;
            unsigned addr =
                static_cast<unsigned>(__cvta_generic_to_shared(&taddr_slot));
            unsigned taddr;
            asm volatile(
                "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 "
                "[%0], 32;"
                :: "r"(addr));
            asm volatile("ld.shared.b32 %0, [%1];"
                         : "=r"(taddr) : "r"(addr));
            asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 32;"
                         :: "r"(taddr));
            asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
            if (threadIdx.x == 0) out[0] = static_cast<int>(taddr);
        }
        """,
    ),
    "compile_tcgen05_ldst_wait": (
        "sm_100a",
        r"""
        #include <cuda_runtime.h>
        extern "C" __global__ void probe(int* out) {
            unsigned taddr = 0;
            unsigned r0 = 0, r1 = 0;
            asm volatile(
                "tcgen05.ld.sync.aligned.32x32b.x2.b32 {%0, %1}, [%2];"
                : "=r"(r0), "=r"(r1) : "r"(taddr));
            asm volatile("tcgen05.wait::ld.sync.aligned;");
            asm volatile(
                "tcgen05.st.sync.aligned.32x32b.x2.b32 [%0], {%1, %2};"
                :: "r"(taddr), "r"(r0), "r"(r1));
            asm volatile("tcgen05.wait::st.sync.aligned;");
            if (threadIdx.x == 0) out[0] = static_cast<int>(r0 + r1);
        }
        """,
    ),
    "compile_tcgen05_cp_shift": (
        "sm_100a",
        r"""
        #include <cuda_runtime.h>
        extern "C" __global__ void probe(int* out) {
            unsigned taddr = 0;
            unsigned long long sdesc = 0;
            asm volatile("tcgen05.cp.cta_group::1.128x256b [%0], %1;"
                         :: "r"(taddr), "l"(sdesc));
            asm volatile("tcgen05.shift.cta_group::1.down [%0];"
                         :: "r"(taddr));
            if (threadIdx.x == 0) out[0] = 1;
        }
        """,
    ),
    "compile_tcgen05_mma": (
        "sm_100a",
        r"""
        #include <cuda_runtime.h>
        extern "C" __global__ void probe(int* out) {
            unsigned daddr = 0, sp_meta = 0, idesc = 0;
            unsigned m0 = 0, m1 = 0, m2 = 0, m3 = 0;
            unsigned long long adesc = 0, bdesc = 0;
            asm volatile(
                "{ .reg .pred p; setp.ne.u32 p, 0, 1; "
                "tcgen05.mma.cta_group::1.kind::tf32 "
                "[%0], %1, %2, %3, {%4, %5, %6, %7}, p; }"
                :: "r"(daddr), "l"(adesc), "l"(bdesc), "r"(idesc),
                   "r"(m0), "r"(m1), "r"(m2), "r"(m3));
            asm volatile(
                "{ .reg .pred p; setp.ne.u32 p, 0, 1; "
                "tcgen05.mma.sp.cta_group::1.kind::f16 "
                "[%0], %1, %2, [%3], %4, {%5, %6, %7, %8}, p; }"
                :: "r"(daddr), "l"(adesc), "l"(bdesc), "r"(sp_meta),
                   "r"(idesc), "r"(m0), "r"(m1), "r"(m2), "r"(m3));
            if (threadIdx.x == 0) out[0] = 1;
        }
        """,
    ),
    "compile_tcgen05_mma_ws": (
        "sm_100a",
        r"""
        #include <cuda_runtime.h>
        extern "C" __global__ void probe(int* out) {
            unsigned daddr = 0, aaddr = 0, sp_meta = 0, idesc = 0;
            unsigned long long bdesc = 0;
            asm volatile(
                "{ .reg .pred p; setp.ne.u32 p, 0, 1; "
                "tcgen05.mma.ws.cta_group::1.kind::tf32 "
                "[%0], [%1], %2, %3, p; }"
                :: "r"(daddr), "r"(aaddr), "l"(bdesc), "r"(idesc));
            asm volatile(
                "{ .reg .pred p; setp.ne.u32 p, 0, 1; "
                "tcgen05.mma.ws.sp.cta_group::1.kind::tf32 "
                "[%0], [%1], %2, [%3], %4, p; }"
                :: "r"(daddr), "r"(aaddr), "l"(bdesc), "r"(sp_meta),
                   "r"(idesc));
            if (threadIdx.x == 0) out[0] = 1;
        }
        """,
    ),
    "compile_tcgen05_fence_commit": (
        "sm_100a",
        r"""
        #include <cuda_runtime.h>
        extern "C" __global__ void probe(int* out) {
            __shared__ unsigned long long barrier;
            unsigned addr = static_cast<unsigned>(__cvta_generic_to_shared(&barrier));
            if (threadIdx.x == 0) {
                asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;"
                             :: "r"(addr), "r"(1));
            }
            __syncthreads();
            asm volatile("tcgen05.fence::before_thread_sync;");
            asm volatile("tcgen05.fence::after_thread_sync;");
            asm volatile(
                "tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];"
                :: "r"(addr));
            if (threadIdx.x == 0) out[0] = 1;
        }
        """,
    ),
    "compile_st_bulk": (
        "sm_100",
        r"""
        #include <cuda_runtime.h>
        extern "C" __global__ void probe(int* out) {
            extern __shared__ int smem[];
            unsigned smem_addr =
                static_cast<unsigned>(__cvta_generic_to_shared(smem));
            asm volatile("st.bulk.weak.shared::cta [%0], 4096, 0;"
                         :: "r"(smem_addr));
            if (threadIdx.x == 0) out[0] = 1;
        }
        """,
    ),
    "compile_stmatrix_b8": (
        "sm_100a",
        r"""
        #include <cuda_runtime.h>
        extern "C" __global__ void probe(int* out) {
            __shared__ int smem[64];
            unsigned addr = static_cast<unsigned>(__cvta_generic_to_shared(smem));
            unsigned r = 0;
            asm volatile(
                "stmatrix.sync.aligned.m16n8.x1.trans.shared.b8 [%0], {%1};"
                :: "r"(addr), "r"(r));
            if (threadIdx.x == 0) out[0] = 1;
        }
        """,
    ),
    "compile_clusterlaunchcontrol": (
        "sm_100",
        r"""
        #include <cuda_runtime.h>
        extern "C" __global__ void probe(int* out) {
            __shared__ unsigned long long barrier;
            __shared__ unsigned handle[4];
            unsigned barrier_addr =
                static_cast<unsigned>(__cvta_generic_to_shared(&barrier));
            unsigned handle_addr =
                static_cast<unsigned>(__cvta_generic_to_shared(handle));
            if (threadIdx.x == 0) {
                asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;"
                             :: "r"(barrier_addr), "r"(1));
                asm volatile(
                    "clusterlaunchcontrol.try_cancel.async"
                    ".mbarrier::complete_tx::bytes.b128 [%0], [%1];"
                    :: "r"(handle_addr), "r"(barrier_addr));
                out[0] = 1;
            }
            unsigned pval, x, y, z, w;
            asm volatile(
                "{ .reg .pred p; .reg .b128 h; "
                "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p, h; "
                "clusterlaunchcontrol.query_cancel.get_first_ctaid.v4.b32.b128 "
                "{%1, %2, %3, %4}, h; "
                "selp.u32 %0, 1, 0, p; }"
                : "=r"(pval), "=r"(x), "=r"(y), "=r"(z), "=r"(w));
            if (threadIdx.x == 0) out[0] += pval + x + y + z + w;
        }
        """,
    ),
    "compile_tensormap_replace_ext": (
        "sm_100a",
        r"""
        #include <cuda_runtime.h>
        extern "C" __global__ void probe(int* out) {
            __shared__ unsigned tensor_map[32];
            unsigned addr =
                static_cast<unsigned>(__cvta_generic_to_shared(tensor_map));
            asm volatile(
                "tensormap.replace.tile.swizzle_atomicity"
                ".shared::cta.b1024.b32 [%0], 0;"
                :: "r"(addr));
            if (threadIdx.x == 0) out[0] = 1;
        }
        """,
    ),
    "compile_cvt_small_float": (
        "sm_100a",
        r"""
        #include <cuda_runtime.h>
        extern "C" __global__ void probe(unsigned short* out) {
            float a = 1.0f, b = -1.0f;
            unsigned short e23, e32, ue8;
            asm volatile("cvt.rn.satfinite.e2m3x2.f32 %0, %1, %2;"
                         : "=h"(e23) : "f"(a), "f"(b));
            asm volatile("cvt.rn.satfinite.e3m2x2.f32 %0, %1, %2;"
                         : "=h"(e32) : "f"(a), "f"(b));
            asm volatile("cvt.rz.satfinite.ue8m0x2.f32 %0, %1, %2;"
                         : "=h"(ue8) : "f"(a), "f"(b));
            if (threadIdx.x == 0) out[0] = e23 + e32 + ue8;
        }
        """,
    ),
    "compile_mma_f8f6f4": (
        "sm_120a",
        r"""
        #include <cuda_runtime.h>
        extern "C" __global__ void probe(int* out) {
            unsigned a0 = 0, a1 = 0, a2 = 0, a3 = 0;
            unsigned b0 = 0, b1 = 0, b2 = 0, b3 = 0, e = 0;
            float c0 = 0, c1 = 0, c2 = 0, c3 = 0, d0, d1, d2, d3;
            asm volatile(
                "mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4"
                ".f32.e3m2.e2m3.f32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};"
                : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
                : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
                  "f"(c0), "f"(c1), "f"(c2), "f"(c3));
            asm volatile(
                "mma.sp::ordered_metadata.sync.aligned.m16n8k64.row.col"
                ".kind::f8f6f4.f32.e3m2.e2m3.f32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9,%10,%11}, "
                "{%12,%13,%14,%15}, %16, 0;"
                : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
                : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
                  "r"(b0), "r"(b1), "r"(b2), "r"(b3),
                  "f"(c0), "f"(c1), "f"(c2), "f"(c3), "r"(e));
            if (threadIdx.x == 0) out[0] = static_cast<int>(d0 + d1 + d2 + d3);
        }
        """,
    ),
}


def compile_probe(validator):
    if validator not in COMPILE_PROBES:
        raise RuntimeError(f"unknown compile probe {validator}")
    if validator in COMPILE_CACHE:
        return COMPILE_CACHE[validator]
    arch, source = COMPILE_PROBES[validator]
    nvcc = shutil.which("nvcc")
    if not nvcc:
        raise RuntimeError("nvcc not found")

    with tempfile.TemporaryDirectory(prefix="gpu_ops_") as tmpdir:
        source_path = os.path.join(tmpdir, "probe.cu")
        object_path = os.path.join(tmpdir, "probe.o")
        with open(source_path, "w", encoding="utf-8") as handle:
            handle.write(textwrap.dedent(source))
        cmd = [nvcc, "-std=c++17"]
        if arch[-1].isalpha():
            compute = f"compute_{arch[3:]}"
            cmd.extend(["-gencode", f"arch={compute},code={arch}"])
        else:
            cmd.extend(["-arch", arch])
        cmd.extend(["-c", source_path, "-o", object_path])
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            detail = (proc.stderr or proc.stdout).strip().splitlines()
            message = detail[-1] if detail else f"nvcc exited {proc.returncode}"
            raise RuntimeError(message)
    COMPILE_CACHE[validator] = f"compiled with {arch}"
    return COMPILE_CACHE[validator]


def compile_all_arch_probes():
    feature_by_validator = {
        feature.validator: feature
        for feature in FEATURES
        if feature.validator is not None
    }
    results = []
    for validator, (arch, _source) in COMPILE_PROBES.items():
        feature = feature_by_validator.get(validator)
        label = feature.label if feature else validator
        feature_id = feature.feature_id if feature else validator
        try:
            detail = compile_probe(validator)
            status = "pass"
        except Exception as exc:
            detail = str(exc)
            status = "fail"
        results.append(
            CompileResult(
                feature_id=feature_id,
                label=label,
                validator=validator,
                arch=arch,
                status=status,
                detail=detail,
            )
        )
    return results


def run_feature(device, props, capability, feature, args):
    if not cc_at_least(capability, feature.min_cc):
        return status_result(
            device,
            props,
            feature,
            "skip",
            f"requires compute capability {feature.min_cc[0]}.{feature.min_cc[1]}+",
        )

    if feature.validator is None:
        return status_result(device, props, feature, "mapped", feature.detail)

    if feature.validator.startswith("compile_"):
        if not args.compile_probes:
            return status_result(
                device,
                props,
                feature,
                "mapped",
                f"{feature.detail} Use --compile-probes to validate nvcc support.",
            )
        try:
            detail = compile_probe(feature.validator)
            return status_result(device, props, feature, "pass", detail)
        except Exception as exc:
            return status_result(device, props, feature, "fail", str(exc))

    validator = VALIDATORS[feature.validator]
    try:
        detail = validator(device, args)
        return status_result(device, props, feature, "pass", detail)
    except Exception as exc:
        return status_result(device, props, feature, "fail", str(exc))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validate CUDA operation families by GPU architecture")
    parser.add_argument("--devices", nargs="*", type=int,
                        help="visible CUDA device indices to validate")
    parser.add_argument("--size", type=int, default=128,
                        help="square GEMM size for runtime GEMM probes")
    parser.add_argument("--elements", type=int, default=1 << 20,
                        help="element count for vector and memory probes")
    parser.add_argument("--compile-probes", action="store_true",
                        help="compile architecture-specific PTX/intrinsic probes with nvcc")
    parser.add_argument("--compile-all-archs", action="store_true",
                        help="compile every sm_80+ architecture PTX/intrinsic probe and exit")
    parser.add_argument("--list", action="store_true",
                        help="print the architecture feature map and exit")
    parser.add_argument("--json", action="store_true",
                        help="emit JSON results")
    parser.add_argument("--strict", action="store_true",
                        help="exit non-zero on failed expected validations")
    return parser.parse_args()


def validate_args(args):
    if args.size <= 0:
        raise SystemExit("--size must be positive")
    if args.elements <= 0:
        raise SystemExit("--elements must be positive")


def print_feature_map():
    print("feature_id,min_cc,label,validator")
    for feature in FEATURES:
        validator = feature.validator or "mapped-only"
        print(
            f"{feature.feature_id},{feature.min_cc[0]}.{feature.min_cc[1]},"
            f"{feature.label},{validator}"
        )


def print_results(results):
    current = None
    for result in results:
        key = (result.device, result.gpu, result.cc)
        if key != current:
            current = key
            print(f"cuda:{result.device} {result.gpu} cc={result.cc}")
        print(
            f"  {result.status.upper():6} {result.feature_id}: "
            f"{result.detail}"
        )


def print_compile_results(results):
    print("architecture PTX/ISA compile probes")
    for result in results:
        print(
            f"  {result.status.upper():6} {result.arch:7} "
            f"{result.feature_id}: {result.detail}"
        )


def main():
    args = parse_args()
    validate_args(args)

    if args.list:
        print_feature_map()
        return

    compile_results = []
    if args.compile_all_archs:
        compile_results = compile_all_arch_probes()
        if args.json:
            print(json.dumps(
                {"compile_probes": [asdict(result) for result in compile_results]},
                indent=2,
            ))
        else:
            print_compile_results(compile_results)
        if args.strict and any(result.status == "fail" for result in compile_results):
            raise SystemExit(1)
        return

    require_torch()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available to PyTorch")

    count = torch.cuda.device_count()
    devices = args.devices if args.devices else list(range(count))
    for device in devices:
        if device < 0 or device >= count:
            raise SystemExit(f"invalid visible CUDA device {device}")

    results = []
    for device in devices:
        torch.cuda.set_device(device)
        props = torch.cuda.get_device_properties(device)
        capability = (props.major, props.minor)
        for feature in FEATURES:
            results.append(run_feature(device, props, capability, feature, args))

    if args.json:
        print(json.dumps([asdict(result) for result in results], indent=2))
    else:
        print_results(results)

    if args.strict and any(result.status == "fail" for result in results):
        raise SystemExit(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
