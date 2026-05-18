/* 
 * Copyright (c) 2016, Ville Timonen
 * All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 * 
 * The views and conclusions contained in the software and documentation are those
 * of the authors and should not be interpreted as representing official policies,
 * either expressed or implied, of the FreeBSD Project.
 */

// Actually, there are no rounding errors due to results being accumulated in an arbitrary order..
// Therefore EPSILON = 0.0f is OK
#define EPSILON 0.001f
#define EPSILOND 0.0000001

__device__ unsigned int mixIndex(size_t index, unsigned int salt) {
	unsigned int x = (unsigned int)index ^ salt;
	x ^= x >> 16;
	x *= 0x7feb352du;
	x ^= x >> 15;
	x *= 0x846ca68bu;
	x ^= x >> 16;
	return x;
}

__device__ float inputValue(size_t index, unsigned int salt) {
	return 0.5f + (float)(mixIndex(index, salt) & 7u) * 0.25f;
}

__device__ unsigned short floatToBfloat16(float value) {
	unsigned int bits = __float_as_uint(value);
	const unsigned int lsb = (bits >> 16) & 1u;
	bits += 0x7fffu + lsb;
	return (unsigned short)(bits >> 16);
}

__device__ unsigned char fp8E4m3Value(size_t index, unsigned int salt) {
	switch (mixIndex(index, salt) & 7u) {
	case 0: return 0x30; // 0.5
	case 1: return 0x34; // 0.75
	case 2: return 0x38; // 1.0
	case 3: return 0x3a; // 1.25
	case 4: return 0x3c; // 1.5
	case 5: return 0x3e; // 1.75
	case 6: return 0x40; // 2.0
	default: return 0x42; // 2.5
	}
}

__device__ unsigned char fp8E5m2Value(size_t index, unsigned int salt) {
	switch (mixIndex(index, salt) & 7u) {
	case 0: return 0x38; // 0.5
	case 1: return 0x3a; // 0.75
	case 2: return 0x3c; // 1.0
	case 3: return 0x3d; // 1.25
	case 4: return 0x3e; // 1.5
	case 5: return 0x3f; // 1.75
	case 6: return 0x40; // 2.0
	default: return 0x41; // 2.5
	}
}

extern "C" __global__ void initFloat(float *A, float *B, size_t elems) {
	size_t index = blockIdx.x*blockDim.x + threadIdx.x;
	if (index >= elems)
		return;

	A[index] = inputValue(index, 0x9e3779b9u);
	B[index] = inputValue(index, 0x85ebca6bu);
}

extern "C" __global__ void initDouble(double *A, double *B, size_t elems) {
	size_t index = blockIdx.x*blockDim.x + threadIdx.x;
	if (index >= elems)
		return;

	A[index] = (double)inputValue(index, 0x9e3779b9u);
	B[index] = (double)inputValue(index, 0x85ebca6bu);
}

extern "C" __global__ void initBfloat16(unsigned short *A, unsigned short *B,
                                         size_t elems) {
	size_t index = blockIdx.x*blockDim.x + threadIdx.x;
	if (index >= elems)
		return;

	A[index] = floatToBfloat16(inputValue(index, 0x9e3779b9u));
	B[index] = floatToBfloat16(inputValue(index, 0x85ebca6bu));
}

extern "C" __global__ void initFp8E4m3(unsigned char *A, unsigned char *B,
                                       size_t elems) {
	size_t index = blockIdx.x*blockDim.x + threadIdx.x;
	if (index >= elems)
		return;

	A[index] = fp8E4m3Value(index, 0x9e3779b9u);
	B[index] = fp8E4m3Value(index, 0x85ebca6bu);
}

extern "C" __global__ void initFp8E5m2(unsigned char *A, unsigned char *B,
                                       size_t elems) {
	size_t index = blockIdx.x*blockDim.x + threadIdx.x;
	if (index >= elems)
		return;

	A[index] = fp8E5m2Value(index, 0x9e3779b9u);
	B[index] = fp8E5m2Value(index, 0x85ebca6bu);
}

extern "C" __global__ void compare(float *C, int *faultyElems, size_t iters) {
	size_t iterStep = blockDim.x*blockDim.y*gridDim.x*gridDim.y;
	size_t myIndex = (blockIdx.y*blockDim.y + threadIdx.y)* // Y
		gridDim.x*blockDim.x + // W
		blockIdx.x*blockDim.x + threadIdx.x; // X

	int myFaulty = 0;
	for (size_t i = 1; i < iters; ++i)
		if (fabsf(C[myIndex] - C[myIndex + i*iterStep]) > EPSILON)
			myFaulty++;

	atomicAdd(faultyElems, myFaulty);
}

extern "C" __global__ void compareD(double *C, int *faultyElems, size_t iters) {
	size_t iterStep = blockDim.x*blockDim.y*gridDim.x*gridDim.y;
	size_t myIndex = (blockIdx.y*blockDim.y + threadIdx.y)* // Y
		gridDim.x*blockDim.x + // W
		blockIdx.x*blockDim.x + threadIdx.x; // X

	int myFaulty = 0;
	for (size_t i = 1; i < iters; ++i)
		if (fabs(C[myIndex] - C[myIndex + i*iterStep]) > EPSILOND)
			myFaulty++;

	atomicAdd(faultyElems, myFaulty);
}
