#!/usr/bin/env python3
"""
GPU peer-to-peer fabric connectivity and bandwidth test.

The measured path uses CUDA device tensors and CUDA events. Topology labels are
read from nvidia-smi when available; bandwidth is measured with peer copies.
"""

import argparse
import re
import subprocess
import sys

try:
    import torch
except ImportError as exc:
    message = str(exc)
    if "ncclCommResume" in message:
        raise SystemExit(
            "PyTorch failed to import because libtorch_cuda.so is loading an "
            "NCCL runtime without ncclCommResume. This is a PyTorch/NCCL "
            "install or LD_LIBRARY_PATH mismatch, not a gpu_p2p.py failure."
        ) from exc
    raise


def bytes_from_mem(value):
    match = re.fullmatch(r"\s*([0-9]+(?:[.][0-9]+)?)\s*([kKmMgG]?)\s*", value)
    if not match:
        raise argparse.ArgumentTypeError(
            "memory must look like 1024, 512M, 1G, or 0.5G")

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


def parse_args():
    parser = argparse.ArgumentParser(
        description="GPU peer-to-peer connectivity and bandwidth test")
    parser.add_argument("--devices", nargs="*", type=int,
                        help="visible CUDA device indices to test")
    parser.add_argument("--mem", type=bytes_from_mem, default=bytes_from_mem("1G"),
                        help="copy buffer size per peer transfer")
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--no-validate", action="store_true",
                        help="skip GPU-side copy validation")
    return parser.parse_args()


def validate_args(args):
    if args.mem <= 0:
        raise SystemExit("--mem must be positive")
    if args.iters <= 0:
        raise SystemExit("--iters must be positive")
    if args.warmup < 0:
        raise SystemExit("--warmup must be non-negative")


def topo_matrix():
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "topo", "-m"], text=True, stderr=subprocess.DEVNULL)
    except Exception:
        return {}

    lines = [line.rstrip() for line in output.splitlines() if line.strip()]
    if not lines:
        return {}

    ansi = re.compile(r"\x1b\[[0-9;]*m")
    header = [ansi.sub("", token) for token in lines[0].split()]
    gpu_columns = [name for name in header
                   if re.fullmatch(r"GPU[0-9]+", name)]
    matrix = {}
    for line in lines[1:]:
        parts = [ansi.sub("", token) for token in line.split()]
        if not parts or not re.fullmatch(r"GPU[0-9]+", parts[0]):
            continue
        row = int(parts[0][3:])
        values = parts[1:1 + len(gpu_columns)]
        for name, value in zip(gpu_columns, values):
            col = int(name[3:])
            matrix[(row, col)] = value
    return matrix


def connection_label(matrix, src, dst):
    label = matrix.get((src, dst), "UNKNOWN")
    if label.startswith("NV"):
        return label
    if label in {"PIX", "PXB", "PHB", "NODE", "SYS"}:
        return label
    return label


def can_access_peer(src, dst):
    with torch.cuda.device(src):
        return bool(torch.cuda.can_device_access_peer(src, dst))


def make_pattern(numel, device, salt):
    data = torch.arange(numel, device=device, dtype=torch.uint8)
    data.add_(salt & 0xFF)
    return data


def measure_pair(src_id, dst_id, numel, args):
    src_device = torch.device("cuda", src_id)
    dst_device = torch.device("cuda", dst_id)
    source = make_pattern(numel, src_device, src_id * 17 + dst_id)
    with torch.cuda.device(dst_device):
        target = torch.empty(numel, device=dst_device, dtype=torch.uint8)

    copy_stream = torch.cuda.Stream(device=dst_device)
    for _ in range(args.warmup):
        with torch.cuda.stream(copy_stream):
            target.copy_(source, non_blocking=True)
    copy_stream.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    with torch.cuda.stream(copy_stream):
        start.record(copy_stream)
        for _ in range(args.iters):
            target.copy_(source, non_blocking=True)
        end.record(copy_stream)
    end.synchronize()
    elapsed_ms = start.elapsed_time(end)

    errors = 0
    if not args.no_validate:
        with torch.cuda.device(dst_device):
            bad = torch.count_nonzero(target != source.to(dst_device))
        torch.cuda.synchronize(dst_device)
        errors = int(bad.item())

    total_bytes = numel * args.iters
    seconds = elapsed_ms / 1000.0
    gib_s = (total_bytes / 1024.0 / 1024.0 / 1024.0) / seconds
    gb_s = (total_bytes / 1.0e9) / seconds
    return elapsed_ms, gib_s, gb_s, errors


def run():
    args = parse_args()
    validate_args(args)

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available to PyTorch")

    count = torch.cuda.device_count()
    devices = args.devices if args.devices else list(range(count))
    for device in devices:
        if device < 0 or device >= count:
            raise SystemExit(f"invalid visible CUDA device {device}")
    if len(devices) < 2:
        raise SystemExit("at least two visible CUDA devices are required")

    numel = args.mem
    matrix = topo_matrix()
    print("visible devices:")
    for device in devices:
        props = torch.cuda.get_device_properties(device)
        print(f"  cuda:{device} {props.name} "
              f"cc={props.major}.{props.minor} bus={props.pci_bus_id}")
    print("src -> dst, topo, peer_access, bandwidth")

    for src in devices:
        for dst in devices:
            if src == dst:
                continue
            topo = connection_label(matrix, src, dst)
            peer = can_access_peer(src, dst)
            if not peer:
                print(f"{src} -> {dst}: {topo}, peer_access=no, skipped")
                continue
            try:
                elapsed_ms, gib_s, gb_s, errors = measure_pair(
                    src, dst, numel, args)
                print(
                    f"{src} -> {dst}: {topo}, peer_access=yes, "
                    f"{gib_s:.2f} GiB/s ({gb_s:.2f} GB/s), "
                    f"{elapsed_ms:.3f} ms, errors={errors}"
                )
            except RuntimeError as exc:
                print(f"{src} -> {dst}: {topo}, peer_access=yes, failed: {exc}")


if __name__ == "__main__":
    try:
        run()
    except KeyboardInterrupt:
        sys.exit(130)
