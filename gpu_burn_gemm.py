#!/usr/bin/env python3
"""
GPU-only GEMM burn for PyTorch CUDA tensors.

The measured loop keeps matrix data, validation, and timing on the GPU. Python
still launches work, but elapsed time is measured with CUDA events and can use a
CUDA graph so CPU dispatch overhead is not included in the reported GEMM time.
"""

import argparse
import ctypes
import ctypes.util
import sys
from dataclasses import dataclass
from typing import Optional, Tuple

import torch


CUDA_R_32F = 0
CUDA_R_8F_E4M3 = 28
CUDA_R_8F_E5M2 = 29
CUBLAS_COMPUTE_32F = 68
CUBLAS_OP_N = 0
CUBLAS_OP_T = 1
CUBLASLT_POINTER_MODE_DEVICE = 1
CUBLASLT_MATMUL_DESC_POINTER_MODE = 2
CUBLASLT_MATMUL_DESC_TRANSA = 3
CUBLASLT_MATMUL_DESC_TRANSB = 4
CUBLAS_STATUS_SUCCESS = 0


class UnsupportedMode(RuntimeError):
    pass


def load_library(name):
    path = ctypes.util.find_library(name)
    if not path:
        raise RuntimeError(f"could not find {name}")
    return ctypes.CDLL(path)


class CublasLt:
    def __init__(self):
        self.lib = load_library("cublasLt")
        self.lib.cublasLtGetStatusString.restype = ctypes.c_char_p
        self.lib.cublasLtCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        self.lib.cublasLtDestroy.argtypes = [ctypes.c_void_p]
        self.lib.cublasLtMatmulDescCreate.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_int,
            ctypes.c_int,
        ]
        self.lib.cublasLtMatmulDescDestroy.argtypes = [ctypes.c_void_p]
        self.lib.cublasLtMatmulDescSetAttribute.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_size_t,
        ]
        self.lib.cublasLtMatrixLayoutCreate.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_int,
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_int64,
        ]
        self.lib.cublasLtMatrixLayoutDestroy.argtypes = [ctypes.c_void_p]
        self.lib.cublasLtMatmul.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_void_p,
        ]
        self.handle = ctypes.c_void_p()
        self.check(self.lib.cublasLtCreate(ctypes.byref(self.handle)),
                   "cublasLtCreate")

    def check(self, status, desc):
        if status == CUBLAS_STATUS_SUCCESS:
            return
        msg = self.lib.cublasLtGetStatusString(status)
        text = msg.decode("utf-8") if msg else f"status {status}"
        if "not supported" in text.lower():
            raise UnsupportedMode(f"{desc}: {text}")
        raise RuntimeError(f"{desc}: {text}")

    def close(self):
        if self.handle:
            self.lib.cublasLtDestroy(self.handle)
            self.handle = ctypes.c_void_p()


class Fp8LtMatmul:
    def __init__(self, n, fp8_type, device):
        self.n = n
        self.lt = CublasLt()
        self.desc = ctypes.c_void_p()
        self.a_layout = ctypes.c_void_p()
        self.b_layout = ctypes.c_void_p()
        self.c_layout = ctypes.c_void_p()
        self.alpha = torch.ones((), device=device, dtype=torch.float32)
        self.beta = torch.zeros((), device=device, dtype=torch.float32)
        self.workspace = torch.empty(64 * 1024 * 1024, device=device,
                                     dtype=torch.uint8)

        self.lt.check(
            self.lt.lib.cublasLtMatmulDescCreate(
                ctypes.byref(self.desc), CUBLAS_COMPUTE_32F, CUDA_R_32F),
            "cublasLtMatmulDescCreate",
        )
        self._set_attr(CUBLASLT_MATMUL_DESC_POINTER_MODE,
                       CUBLASLT_POINTER_MODE_DEVICE)
        self._set_attr(CUBLASLT_MATMUL_DESC_TRANSA, CUBLAS_OP_T)
        self._set_attr(CUBLASLT_MATMUL_DESC_TRANSB, CUBLAS_OP_N)

        self.lt.check(
            self.lt.lib.cublasLtMatrixLayoutCreate(
                ctypes.byref(self.a_layout), fp8_type, n, n, n),
            "cublasLtMatrixLayoutCreate(A)",
        )
        self.lt.check(
            self.lt.lib.cublasLtMatrixLayoutCreate(
                ctypes.byref(self.b_layout), fp8_type, n, n, n),
            "cublasLtMatrixLayoutCreate(B)",
        )
        self.lt.check(
            self.lt.lib.cublasLtMatrixLayoutCreate(
                ctypes.byref(self.c_layout), CUDA_R_32F, n, n, n),
            "cublasLtMatrixLayoutCreate(C)",
        )

    def _set_attr(self, attr, value):
        raw = ctypes.c_int(value)
        self.lt.check(
            self.lt.lib.cublasLtMatmulDescSetAttribute(
                self.desc, attr, ctypes.byref(raw), ctypes.sizeof(raw)),
            "cublasLtMatmulDescSetAttribute",
        )

    def __call__(self, a, b, out):
        stream = torch.cuda.current_stream(out.device).cuda_stream
        self.lt.check(
            self.lt.lib.cublasLtMatmul(
                self.lt.handle,
                self.desc,
                ctypes.c_void_p(self.alpha.data_ptr()),
                ctypes.c_void_p(a.data_ptr()),
                self.a_layout,
                ctypes.c_void_p(b.data_ptr()),
                self.b_layout,
                ctypes.c_void_p(self.beta.data_ptr()),
                ctypes.c_void_p(out.data_ptr()),
                self.c_layout,
                ctypes.c_void_p(out.data_ptr()),
                self.c_layout,
                None,
                ctypes.c_void_p(self.workspace.data_ptr()),
                self.workspace.numel(),
                ctypes.c_void_p(stream),
            ),
            "cublasLtMatmul(fp8)",
        )

    def close(self):
        for handle, destroy in (
            (self.c_layout, self.lt.lib.cublasLtMatrixLayoutDestroy),
            (self.b_layout, self.lt.lib.cublasLtMatrixLayoutDestroy),
            (self.a_layout, self.lt.lib.cublasLtMatrixLayoutDestroy),
            (self.desc, self.lt.lib.cublasLtMatmulDescDestroy),
        ):
            if handle:
                destroy(handle)
        self.lt.close()


@dataclass(frozen=True)
class Mode:
    name: str
    torch_dtype: torch.dtype
    output_dtype: torch.dtype
    min_capability: Tuple[int, int]
    fp8_cuda_type: Optional[int] = None
    tolerance: float = 0.0


MODES = {
    "fp32": Mode("fp32", torch.float32, torch.float32, (0, 0), None, 1e-3),
    "fp16": Mode("fp16", torch.float16, torch.float16, (0, 0), None, 1e-2),
    "bf16": Mode("bf16", torch.bfloat16, torch.bfloat16, (8, 0), None, 1e-1),
    "fp8": Mode("fp8", torch.float8_e4m3fn, torch.float32, (8, 9),
                CUDA_R_8F_E4M3, 1e-1),
    "fp8-e4m3": Mode("fp8-e4m3", torch.float8_e4m3fn, torch.float32, (8, 9),
                     CUDA_R_8F_E4M3, 1e-1),
    "fp8-e5m2": Mode("fp8-e5m2", torch.float8_e5m2, torch.float32, (8, 9),
                     CUDA_R_8F_E5M2, 1e-1),
}


def parse_modes(values):
    if values == ["all"]:
        return ["fp32", "fp16", "bf16", "fp8"]
    modes = []
    for value in values:
        if value not in MODES:
            raise SystemExit(f"unknown mode {value}")
        modes.append(value)
    return modes


def capability_at_least(actual, required):
    return actual[0] > required[0] or (
        actual[0] == required[0] and actual[1] >= required[1])


def make_input(n, mode, device):
    if mode.fp8_cuda_type is None:
        return torch.empty((n, n), device=device,
                           dtype=mode.torch_dtype).uniform_(-1.0, 1.0)
    tmp = torch.empty((n, n), device=device,
                      dtype=torch.float32).uniform_(-1.0, 1.0)
    out = tmp.to(mode.torch_dtype)
    del tmp
    return out


def make_op(mode, n, device):
    if mode.fp8_cuda_type is not None:
        return Fp8LtMatmul(n, mode.fp8_cuda_type, device)

    def matmul(a, b, out):
        torch.mm(a, b, out=out)

    return matmul


def run_mode(mode_name, args, device, capability):
    mode = MODES[mode_name]
    if not capability_at_least(capability, mode.min_capability):
        message = (f"{mode.name}: skipped, requires compute capability "
                   f"{mode.min_capability[0]}.{mode.min_capability[1]}+")
        if args.strict:
            raise RuntimeError(message)
        print(message)
        return

    torch.cuda.empty_cache()
    a = make_input(args.size, mode, device)
    b = make_input(args.size, mode, device)
    out = torch.empty((args.size, args.size), device=device,
                      dtype=mode.output_dtype)
    ref = torch.empty_like(out)
    op = make_op(mode, args.size, device)

    try:
        for _ in range(args.warmup):
            op(a, b, out)
        op(a, b, ref)
        torch.cuda.synchronize(device)

        graph = None
        if args.graph:
            try:
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    op(a, b, out)
            except Exception as exc:
                if args.strict:
                    raise
                graph = None
                print(f"{mode.name}: CUDA graph disabled: {exc}")
                torch.cuda.synchronize(device)

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        if graph is not None:
            for _ in range(args.iters):
                graph.replay()
        else:
            for _ in range(args.iters):
                op(a, b, out)
        end.record()
        end.synchronize()
        elapsed_ms = start.elapsed_time(end)

        errors = 0
        max_diff = 0.0
        if not args.no_validate:
            diff = (out.float() - ref.float()).abs()
            bad = torch.count_nonzero(diff > mode.tolerance)
            nonfinite = torch.count_nonzero(~torch.isfinite(out.float()))
            max_diff_gpu = diff.max()
            torch.cuda.synchronize(device)
            errors = int((bad + nonfinite).item())
            max_diff = float(max_diff_gpu.item())

        ops = 2.0 * args.size * args.size * args.size * args.iters
        tflops = ops / (elapsed_ms / 1000.0) / 1.0e12
        graph_text = "graph" if graph is not None else "eager"
        print(
            f"{mode.name}: {args.iters} GEMMs, n={args.size}, "
            f"{elapsed_ms:.3f} ms GPU-event time, {tflops:.2f} TFLOP/s, "
            f"errors={errors}, max_diff={max_diff:.6g}, {graph_text}"
        )
    except UnsupportedMode as exc:
        if args.strict:
            raise
        print(f"{mode.name}: skipped, {exc}")
    finally:
        if isinstance(op, Fp8LtMatmul):
            op.close()


def main():
    parser = argparse.ArgumentParser(
        description="GPU-resident GEMM burn using CUDA events for timing")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--size", type=int, default=8192)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--modes", nargs="+", default=["all"],
                        help="all, fp32, fp16, bf16, fp8, fp8-e4m3, fp8-e5m2")
    parser.add_argument("--no-graph", dest="graph", action="store_false",
                        help="do not capture GEMM in a CUDA graph")
    parser.add_argument("--no-validate", action="store_true",
                        help="skip GPU-side repeated-output validation")
    parser.add_argument("--strict", action="store_true",
                        help="treat skipped modes or graph capture failures as errors")
    parser.set_defaults(graph=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available to PyTorch")

    torch.cuda.set_device(args.device)
    device = torch.device("cuda", args.device)
    capability = torch.cuda.get_device_capability(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

    print(f"device={args.device} {torch.cuda.get_device_name(device)} "
          f"cc={capability[0]}.{capability[1]}")
    print("matrix data, validation, and timing stay on CUDA; only summary "
          "scalars are copied back after synchronization")

    for mode_name in parse_modes(args.modes):
        run_mode(mode_name, args, device, capability)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
