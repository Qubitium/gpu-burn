#!/usr/bin/env python3
"""
GPU internal memory bandwidth burn for PyTorch CUDA tensors.

The measured loop copies between device tensors and uses CUDA events for timing.
No host tensor participates in the measured path; only final summary scalars are
printed after synchronization.
"""

import argparse
import re
import sys

try:
    import torch
except ImportError as exc:
    message = str(exc)
    if "ncclCommResume" in message:
        raise SystemExit(
            "PyTorch failed to import because libtorch_cuda.so is loading an "
            "NCCL runtime without ncclCommResume. This is a PyTorch/NCCL "
            "install or LD_LIBRARY_PATH mismatch, not a gpu_bandwidth.py "
            "failure."
        ) from exc
    raise


def bytes_from_mem(value):
    match = re.fullmatch(r"\s*([0-9]+(?:[.][0-9]+)?)\s*([kKmMgG]?)\s*", value)
    if not match:
        raise argparse.ArgumentTypeError(
            "memory must look like 4096, 512M, 4G, or 0.5G")

    amount = float(match.group(1))
    suffix = match.group(2).lower()
    multiplier = {
        "": 1024 * 1024,
        "k": 1024,
        "m": 1024 * 1024,
        "g": 1024 * 1024 * 1024,
    }[suffix]
    size = int(amount * multiplier)
    if size <= 0:
        raise argparse.ArgumentTypeError("memory must be positive")
    return size


def mem_label(num_bytes):
    gib = num_bytes / 1024.0 / 1024.0 / 1024.0
    if gib >= 1.0:
        return f"{gib:.3f}G"
    mib = num_bytes / 1024.0 / 1024.0
    return f"{mib:.3f}M"


def make_pattern(numel, device):
    return torch.arange(numel, device=device, dtype=torch.uint8)


def parse_args():
    parser = argparse.ArgumentParser(
        description="GPU-resident device-to-device bandwidth burn")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--mem", type=bytes_from_mem, default=bytes_from_mem("4G"),
                        help="buffer size per device tensor, e.g. 4096M or 4G")
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--streams", type=int, default=1,
                        help="parallel CUDA streams and buffer pairs")
    parser.add_argument("--no-graph", dest="graph", action="store_false",
                        help="do not capture copies in a CUDA graph")
    parser.add_argument("--no-validate", action="store_true",
                        help="skip GPU-side copy validation")
    parser.add_argument("--strict", action="store_true",
                        help="treat graph capture failures as errors")
    parser.set_defaults(graph=True)
    return parser.parse_args()


def validate_args(args):
    if args.mem <= 0:
        raise SystemExit("--mem must be positive")
    if args.iters <= 0:
        raise SystemExit("--iters must be positive")
    if args.warmup < 0:
        raise SystemExit("--warmup must be non-negative")
    if args.streams <= 0:
        raise SystemExit("--streams must be positive")


def allocate_buffers(numel, streams, device):
    src = []
    dst = []
    for stream_index in range(streams):
        source = make_pattern(numel, device)
        source.add_(stream_index & 0xFF)
        src.append(source)
        dst.append(torch.empty_like(source))
    return src, dst


def issue_copies(src, dst, streams=None):
    if streams is None:
        for source, target in zip(src, dst):
            target.copy_(source, non_blocking=True)
        return

    for source, target, stream in zip(src, dst, streams):
        with torch.cuda.stream(stream):
            target.copy_(source, non_blocking=True)


def run():
    args = parse_args()
    validate_args(args)

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available to PyTorch")

    torch.cuda.set_device(args.device)
    device = torch.device("cuda", args.device)
    capability = torch.cuda.get_device_capability(device)
    numel = args.mem

    print(f"device={args.device} {torch.cuda.get_device_name(device)} "
          f"cc={capability[0]}.{capability[1]}")
    print("device-to-device copies, validation, and timing stay on CUDA; "
          "only summary scalars are copied back after synchronization")

    src, dst = allocate_buffers(numel, args.streams, device)
    streams = [torch.cuda.Stream(device=device) for _ in range(args.streams)]

    for _ in range(args.warmup):
        issue_copies(src, dst, streams)
    torch.cuda.synchronize(device)

    graph = None
    if args.graph and args.streams == 1:
        try:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                issue_copies(src, dst)
        except Exception as exc:
            if args.strict:
                raise
            graph = None
            print(f"CUDA graph disabled: {exc}")
            torch.cuda.synchronize(device)
    elif args.graph:
        print("CUDA graph disabled: multiple streams requested")

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    if graph is not None:
        for _ in range(args.iters):
            graph.replay()
    else:
        for _ in range(args.iters):
            issue_copies(src, dst, streams)
    end.record()
    end.synchronize()
    elapsed_ms = start.elapsed_time(end)

    errors = 0
    if not args.no_validate:
        checks = [
            torch.count_nonzero(source != target)
            for source, target in zip(src, dst)
        ]
        total = torch.stack(checks).sum()
        torch.cuda.synchronize(device)
        errors = int(total.item())

    total_bytes = numel * args.streams * args.iters
    seconds = elapsed_ms / 1000.0
    payload_gib_s = (total_bytes / 1024.0 / 1024.0 / 1024.0) / seconds
    payload_gb_s = (total_bytes / 1.0e9) / seconds
    hbm_bytes = total_bytes * 2
    hbm_gib_s = (hbm_bytes / 1024.0 / 1024.0 / 1024.0) / seconds
    hbm_gb_s = (hbm_bytes / 1.0e9) / seconds
    graph_text = "graph" if graph is not None else "eager"
    print(
        f"bandwidth: {args.iters} copies, {args.streams} stream(s), "
        f"{mem_label(numel)} per stream, {elapsed_ms:.3f} ms GPU-event time, "
        f"payload={payload_gib_s:.2f} GiB/s ({payload_gb_s:.2f} GB/s), "
        f"hbm_read_write={hbm_gib_s:.2f} GiB/s ({hbm_gb_s:.2f} GB/s), "
        f"errors={errors}, {graph_text}"
    )


if __name__ == "__main__":
    try:
        run()
    except KeyboardInterrupt:
        sys.exit(130)
