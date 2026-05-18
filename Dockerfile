ARG CUDA_VERSION=13.0.0
ARG IMAGE_DISTRO=ubi8
ARG COMPUTE=75

FROM nvidia/cuda:${CUDA_VERSION}-devel-${IMAGE_DISTRO} AS builder
ARG COMPUTE

WORKDIR /build

COPY . /build/

RUN make COMPUTE=${COMPUTE}

FROM nvidia/cuda:${CUDA_VERSION}-runtime-${IMAGE_DISTRO}

COPY --from=builder /build/gpu_burn /app/
COPY --from=builder /build/compare.ptx /app/

WORKDIR /app

CMD ["./gpu_burn", "60"]
