# gpu-burn

Multi-GPU CUDA stress test
<http://wili.cc/blog/gpu-burn.html>

- [gpu-burn](#gpu-burn)
  - [Easy docker build and run](#easy-docker-build-and-run)
  - [Binary packages](#binary-packages)
  - [Building](#building)
  - [Usage](#usage)

## Easy docker build and run

```plain
git clone https://github.com/wilicc/gpu-burn
cd gpu-burn
make image
docker run --rm --gpus all gpu-burn
```

## Binary packages

<https://repology.org/project/gpu-burn/versions>

## Building

To build GPU Burn:

`make`

To remove artifacts built by GPU Burn:

`make clean`

GPU Burn builds for the highest visible GPU compute capability reported by `nvidia-smi`.
On H100/H200 systems this selects Compute Capability 9.0 (`COMPUTE=90`).
If no GPU capability can be detected, GPU Burn falls back to Compute Capability 7.5.
On mixed-generation systems, set `COMPUTE` to the lowest capability you intend
to run because the compare PTX must be loadable by every selected GPU.
To override this with a different value:

`make COMPUTE=<compute capability value>`

CFLAGS can be added when invoking make to add to the default
list of compiler flags:

`make CFLAGS=-Wall`

LDFLAGS can be added when invoking make to add to the default
list of linker flags:

`make LDFLAGS=-lmylib`

NVCCFLAGS can be added when invoking make to add to the default
list of nvcc flags:

`make NVCCFLAGS=-ccbin <path to host compiler>`

CUDAPATH can be added to point to a non standard install or
specific version of the cuda toolkit (default is
/usr/local/cuda):

`make CUDAPATH=/usr/local/cuda-<version>`

CCPATH can be specified to point to a specific gcc (default is
/usr/bin):

`make CCPATH=/usr/local/bin`

CUDA_VERSION and IMAGE_DISTRO can be used to override the base
images used when building the Docker `image` target, while IMAGE_NAME
can be set to change the resulting image tag:

`make IMAGE_NAME=myregistry.private.com/gpu-burn CUDA_VERSION=13.0.0 IMAGE_DISTRO=ubuntu22.04 image`

## Usage

```plain
    GPU Burn
    Usage: gpu_burn [OPTIONS] [TIME]
    
    -m X   Use X MB of memory
    -m N%  Use N% of the available GPU memory
    -d     Use doubles
    -tc    Use TF32 Tensor Core compute for float GEMM
    -bf16  Use BF16 Tensor Core compute with FP32 output
    -fp8   Use FP8 E4M3 Tensor Core compute with FP32 output
    -fp8-e5m2
           Use FP8 E5M2 Tensor Core compute with FP32 output
    -l     List all GPUs in the system
    -i N   Execute only on GPU N
    -h     Show this help message
    
    Example:
    gpu_burn -d 3600
```

BF16 mode requires Compute Capability 8.0 or newer. FP8 modes require Compute
Capability 8.9 or newer; H100/H200 support these FP8 paths.

## Python GPU tools

`gpu_burn.py` provides a PyTorch CUDA GEMM burn for FP32, FP64, FP16, BF16,
and FP8. Matrix allocation, GEMM output, validation, and elapsed-time
measurement stay on CUDA; it uses CUDA events for timing and CUDA graphs by
default to avoid including Python dispatch overhead in the reported GEMM time.

```plain
./gpu_burn.py --modes ALL --size 8192 --time 10
./gpu_burn.py --modes FP8,FP8-E5M2 --size 8192 --time 10 --device 0
```

`--size` is the square GEMM dimension, so `--size 8192` multiplies two
8192x8192 matrices. `--time` is the target seconds of GPU-event GEMM time per
selected mode. Use `--iters` instead of `--time` when an exact GEMM count is
needed. `--modes` is one comma-delimited argument; for example,
`--modes FP32,FP64,BF16`.

`ALL` runs FP32, FP64, FP16, BF16, and the default supported FP8 E4M3 path.
`fp8-e5m2` is available as an explicit optional mode and is skipped cleanly when
the local cuBLASLt build does not support that exact datatype/layout/output
combination. FP8 in the Python tool uses cuBLASLt directly and requires Compute
Capability 8.9 or newer.

If importing PyTorch fails with `undefined symbol: ncclCommResume`, PyTorch is
loading an older incompatible NCCL runtime before the benchmark starts. Check
which NCCL is being loaded with:

```plain
ldd ~/.local/lib/python*/site-packages/torch/lib/libtorch_cuda.so | grep nccl
```

Unset conflicting `LD_LIBRARY_PATH` entries or reinstall matching `torch` and
`nvidia-nccl-cu13` packages.

`gpu_bandwidth.py` measures internal device memory bandwidth with GPU-resident
device-to-device copies and CUDA event timing. It reports copy payload bandwidth
plus estimated HBM read+write bandwidth, which is the comparable number for HBM
spec sheets:

```plain
./gpu_bandwidth.py --device 0 --mem 4G --iters 100
./gpu_bandwidth.py --device 0 --mem 2G --streams 4 --iters 100
```

`gpu_p2p.py` reports visible GPU pair connectivity from `nvidia-smi topo -m`,
checks CUDA peer-access support, and measures GPU-to-GPU peer-copy bandwidth.
The default output is one-way payload bandwidth. On A100 `NV12` links, compare
that value with roughly 300 GB/s; NVIDIA's 600 GB/s NVLink number is
bidirectional link capacity. Use `--bidirectional` to diagnose simultaneous
opposite-direction peer copies:

```plain
./gpu_p2p.py --mem 1G --iters 20
./gpu_p2p.py --mem 1G --iters 20 --bidirectional
./gpu_p2p.py --devices 0 1 2 3 --mem 512M --iters 50
```

`gpu_ops.py` validates CUDA operation families according to each visible GPU's
compute capability. Runtime checks exercise memory copy, CUDA graphs, CUDA core
arithmetic, atomics, and tensor-core GEMM modes that apply to the detected
architecture. Optional compile probes check architecture-specific PTX and CUDA
intrinsics. For local `sm_80` Ampere GPUs, the compile probes cover `ldmatrix`,
FP64 DMMA, INT4 MMA, binary BMMA, `cp.async`, `mbarrier`, `redux.sync`, and L2
cache-hint policy instructions. Hopper probes cover WGMMA, DPX and
`cp.async.bulk`; Blackwell probes cover `tcgen05`:

```plain
./gpu_ops.py --devices 0 --size 128 --compile-probes
./gpu_ops.py --devices 0 1 2 3 --size 256
./gpu_ops.py --list
```

H100/H200 are Hopper `sm_90` GPUs, so they map to WGMMA, TMA /
`cp.async.bulk`, DPX, FP8 tensor cores, and thread-block clusters. `tcgen05` is
mapped as a Blackwell `sm_100a` feature, not a Hopper feature.
