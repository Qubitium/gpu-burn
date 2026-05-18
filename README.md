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

## Python GPU GEMM burn

`gpu_burn_gemm.py` provides a PyTorch CUDA GEMM burn for FP32, FP16, BF16, and
FP8. Matrix allocation, GEMM output, validation, and elapsed-time measurement
stay on CUDA; it uses CUDA events for timing and CUDA graphs by default to avoid
including Python dispatch overhead in the reported GEMM time.

```plain
./gpu_burn_gemm.py --modes all --size 8192 --iters 100
./gpu_burn_gemm.py --modes fp8 fp8-e5m2 --size 8192 --iters 100 --device 0
```

`all` runs the default supported FP8 E4M3 path. `fp8-e5m2` is available as an
explicit optional mode and is skipped cleanly when the local cuBLASLt build does
not support that exact datatype/layout/output combination. FP8 in the Python
tool uses cuBLASLt directly and requires Compute Capability 8.9 or newer.
