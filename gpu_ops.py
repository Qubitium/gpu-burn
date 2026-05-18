#!/usr/bin/env python3
"""
Validate CUDA architecture feature families exposed by the visible GPUs.

This is a feature-family validator, not a full SASS instruction exerciser. It
maps compute capability to the CUDA operations that should be available on that
architecture, runs practical runtime tests for library-exposed operations, and
optionally compiles small ISA probes for architecture-specific PTX/intrinsic
features that are not directly exposed by PyTorch.
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
    message = str(exc)
    if "ncclCommResume" in message:
        raise SystemExit(
            "PyTorch failed to import because libtorch_cuda.so is loading an "
            "NCCL runtime without ncclCommResume. This is a PyTorch/NCCL "
            "install or LD_LIBRARY_PATH mismatch, not a gpu_ops.py failure."
        ) from exc
    raise


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
        "ampere_mbarrier",
        "Ampere shared-memory mbarrier",
        (8, 0),
        "compile_mbarrier",
        "Compiles sm_80 mbarrier init/invalidate instructions.",
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
        "hopper_dpx",
        "Hopper DPX intrinsics",
        (9, 0),
        "compile_dpx",
        "Compiles CUDA DPX intrinsic probes such as __vimax3_s32.",
    ),
    Feature(
        "hopper_wgmma",
        "Hopper warp-group MMA",
        (9, 0),
        "compile_wgmma",
        "Compiles sm_90a WGMMA proxy instructions.",
    ),
    Feature(
        "hopper_tma_cp_async_bulk",
        "Hopper TMA / cp.async.bulk",
        (9, 0),
        "compile_cp_async_bulk",
        "Compiles an sm_90+ cp.async.bulk.prefetch probe.",
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
        "Blackwell tcgen05 Tensor Core ISA",
        (10, 0),
        "compile_tcgen05",
        "Compiles an sm_100a tcgen05 assembly probe when nvcc supports it.",
    ),
    Feature(
        "blackwell_tensor_memory",
        "Blackwell tensor memory",
        (10, 0),
        None,
        "Mapped to sm_100a-family features. Runtime validation needs a tcgen05 kernel.",
    ),
]


def cc_at_least(actual: Capability, required: Capability) -> bool:
    return actual[0] > required[0] or (
        actual[0] == required[0] and actual[1] >= required[1])


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


COMPILE_PROBES = {
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
    "compile_cp_async_bulk": (
        "sm_90",
        r"""
        #include <cuda_runtime.h>
        extern "C" __global__ void probe(const int* src) {
            const void* ptr = src;
            asm volatile("cp.async.bulk.prefetch.L2.global [%0], 128;"
                         :: "l"(ptr));
        }
        """,
    ),
    "compile_wgmma": (
        "sm_90a",
        r"""
        #include <cuda_runtime.h>
        extern "C" __global__ void probe() {
            asm volatile("wgmma.fence.sync.aligned;");
            asm volatile("wgmma.commit_group.sync.aligned;");
            asm volatile("wgmma.wait_group.sync.aligned 0;");
        }
        """,
    ),
    "compile_tcgen05": (
        "sm_100a",
        r"""
        #include <cuda_runtime.h>
        extern "C" __global__ void probe() {
            asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
        }
        """,
    ),
}


def compile_probe(validator):
    if validator not in COMPILE_PROBES:
        raise RuntimeError(f"unknown compile probe {validator}")
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
    return f"compiled with {arch}"


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


def main():
    args = parse_args()
    validate_args(args)

    if args.list:
        print_feature_map()
        return

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
