/*
 * Copyright (c) 2022, Ville Timonen
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *	this list of conditions and the following disclaimer in the
 *documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 *AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 *IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
 *FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 *DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 *SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 *OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 *OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * The views and conclusions contained in the software and documentation are
 *those of the authors and should not be interpreted as representing official
 *policies, either expressed or implied, of the FreeBSD Project.
 */

// Matrices are SIZE*SIZE..  POT should be efficiently implemented in CUBLAS
#define SIZE 8192ul
#define USEMEM 0.9 // Try to allocate 90% of memory
#define COMPARE_KERNEL "compare.ptx"

// Used to report op/s, measured through Visual Profiler, CUBLAS from CUDA 7.5
// (Seems that they indeed take the naive dim^3 approach)
//#define OPS_PER_MUL 17188257792ul // Measured for SIZE = 2048
#define OPS_PER_MUL 1100048498688ul // Extrapolated for SIZE = 8192

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <errno.h>
#include <exception>
#include <fstream>
#include <map>
#include <signal.h>
#include <stdexcept>
#include <string.h>
#include <string>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <thread>
#include <time.h>
#include <unistd.h>
#include <vector>
#include <regex>

#define SIGTERM_TIMEOUT_THRESHOLD_SECS 30 // number of seconds for sigterm to kill child processes before forcing a sigkill

#include "cublas_v2.h"
#define CUDA_ENABLE_DEPRECATED
#include <cuda.h>
#include <cuda_fp8.h>

enum TestMode {
    TEST_FP32,
    TEST_FP64,
    TEST_TF32,
    TEST_BF16,
    TEST_FP8_E4M3,
    TEST_FP8_E5M2
};

bool computeCapabilityAtLeast(int major, int minor, int reqMajor,
                              int reqMinor) {
    return major > reqMajor || (major == reqMajor && minor >= reqMinor);
}

bool isFp8Mode(TestMode mode) {
    return mode == TEST_FP8_E4M3 || mode == TEST_FP8_E5M2;
}

bool usesDoubleOutput(TestMode mode) { return mode == TEST_FP64; }

size_t resultElementSize(TestMode mode) {
    return usesDoubleOutput(mode) ? sizeof(double) : sizeof(float);
}

const char *gemmModeName(TestMode mode) {
    switch (mode) {
    case TEST_FP64:
        return "using DOUBLES";
    case TEST_TF32:
        return "using FLOATS (TF32 Tensor Cores)";
    case TEST_BF16:
        return "using BFLOAT16 Tensor Cores (FP32 accumulate/output)";
    case TEST_FP8_E4M3:
        return "using FP8 E4M3 Tensor Cores (FP32 accumulate/output)";
    case TEST_FP8_E5M2:
        return "using FP8 E5M2 Tensor Cores (FP32 accumulate/output)";
    case TEST_FP32:
    default:
        return "using FLOATS (pedantic FP32)";
    }
}

#if CUBLAS_VERSION >= 11000
cudaDataType_t inputDataType(TestMode mode) {
    switch (mode) {
    case TEST_BF16:
        return CUDA_R_16BF;
    case TEST_FP8_E4M3:
        return CUDA_R_8F_E4M3;
    case TEST_FP8_E5M2:
        return CUDA_R_8F_E5M2;
    case TEST_FP32:
    case TEST_TF32:
    default:
        return CUDA_R_32F;
    }
}

cublasComputeType_t computeType(TestMode mode) {
    switch (mode) {
    case TEST_FP32:
        return CUBLAS_COMPUTE_32F_PEDANTIC;
    case TEST_TF32:
        return CUBLAS_COMPUTE_32F_FAST_TF32;
    case TEST_BF16:
    case TEST_FP8_E4M3:
    case TEST_FP8_E5M2:
    default:
        return CUBLAS_COMPUTE_32F;
    }
}
#endif

uint16_t floatToBfloat16(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    const uint32_t lsb = (bits >> 16) & 1u;
    bits += 0x7fffu + lsb;
    return (uint16_t)(bits >> 16);
}

template <class T> T convertInput(float value, TestMode mode) {
    (void)mode;
    return (T)value;
}

template <> uint16_t convertInput<uint16_t>(float value, TestMode mode) {
    (void)mode;
    return floatToBfloat16(value);
}

template <> uint8_t convertInput<uint8_t>(float value, TestMode mode) {
    return __nv_cvt_float_to_fp8(
        value, __NV_SATFINITE,
        mode == TEST_FP8_E5M2 ? __NV_E5M2 : __NV_E4M3);
}

void selectTestMode(TestMode *mode, bool *modeSelected, TestMode selected,
                    const char *arg) {
    if (*modeSelected) {
        fprintf(stderr, "Only one precision mode can be selected; got %s\n",
                arg);
        exit(EINVAL);
    }

    *mode = selected;
    *modeSelected = true;
}

void _checkError(int rCode, std::string file, int line, std::string desc = "") {
    if (rCode != CUDA_SUCCESS) {
        const char *err;
        cuGetErrorString((CUresult)rCode, &err);

        throw std::runtime_error(
            (desc == "" ? std::string("Error (")
                        : (std::string("Error in ") + desc + " (")) +
            file + ":" + std::to_string(line) + "): " + err);
        // Yes, this *is* a memory leak, but this block is only executed on
        // error, so it's not a big deal
    }
}

void _checkError(cublasStatus_t rCode, std::string file, int line, std::string desc = "") {
    if (rCode != CUBLAS_STATUS_SUCCESS) {
#if CUBLAS_VER_MAJOR >= 12
		const char *err = cublasGetStatusString(rCode);
#else
		const char *err = "";
#endif
        throw std::runtime_error(
            (desc == "" ? std::string("Error (")
                        : (std::string("Error in ") + desc + " (")) +
            file + ":" + std::to_string(line) + "): " + err);
        // Yes, this *is* a memory leak, but this block is only executed on
        // error, so it's not a big deal
    }
}

#define checkError(rCode, ...)                                                 \
    _checkError(rCode, __FILE__, __LINE__, ##__VA_ARGS__)

double getTime() {
    struct timeval t;
    gettimeofday(&t, NULL);
    return (double)t.tv_sec + (double)t.tv_usec / 1e6;
}

bool g_running = false;

template <class T> class GPU_Test {
  public:
    GPU_Test(int dev, TestMode mode, const char *kernelFile)
        : d_mode(mode), d_devNumber(dev), d_kernelFile(kernelFile) {
        checkError(cuDeviceGet(&d_dev, d_devNumber));
        checkModeSupported();
#if defined(CUDA_VERSION) && CUDA_VERSION >= 13000
        checkError(cuCtxCreate(&d_ctx, nullptr, 0, d_dev));
#else
        checkError(cuCtxCreate(&d_ctx, 0, d_dev));
#endif

        bind();

        // checkError(cublasInit());
        checkError(cublasCreate(&d_cublas), "init");

#if CUBLAS_VERSION < 11000
        if (d_mode == TEST_TF32)
            checkError(cublasSetMathMode(d_cublas, CUBLAS_TENSOR_OP_MATH));
#endif

        checkError(cuMemAllocHost((void **)&d_faultyElemsHost, sizeof(int)));
        d_error = 0;

        g_running = true;

        struct sigaction action;
        memset(&action, 0, sizeof(struct sigaction));
        action.sa_handler = termHandler;
        sigaction(SIGTERM, &action, NULL);
    }
    ~GPU_Test() {
        bind();
        checkError(cuMemFree(d_Cdata), "Free A");
        checkError(cuMemFree(d_Adata), "Free B");
        checkError(cuMemFree(d_Bdata), "Free C");
        cuMemFreeHost(d_faultyElemsHost);
        printf("Freed memory for dev %d\n", d_devNumber);

        cublasDestroy(d_cublas);
        printf("Uninitted cublas\n");
    }

    static void termHandler(int signum) { g_running = false; }

    unsigned long long int getErrors() {
        if (*d_faultyElemsHost) {
            d_error += (long long int)*d_faultyElemsHost;
        }
        unsigned long long int tempErrs = d_error;
        d_error = 0;
        return tempErrs;
    }

    size_t getIters() { return d_iters; }

    void bind() { checkError(cuCtxSetCurrent(d_ctx), "Bind CTX"); }

    size_t totalMemory() {
        bind();
        size_t freeMem, totalMem;
        checkError(cuMemGetInfo(&freeMem, &totalMem));
        return totalMem;
    }

    size_t availMemory() {
        bind();
        size_t freeMem, totalMem;
        checkError(cuMemGetInfo(&freeMem, &totalMem));
        return freeMem;
    }

    void initBuffers(T *A, T *B, ssize_t useBytes = 0) {
        bind();

        if (useBytes == 0)
            useBytes = (ssize_t)((double)availMemory() * USEMEM);
        if (useBytes < 0)
            useBytes = (ssize_t)((double)availMemory() * (-useBytes / 100.0));

        printf("Initialized device %d with %lu MB of memory (%lu MB available, "
               "using %lu MB of it), %s\n",
               d_devNumber, totalMemory() / 1024ul / 1024ul,
               availMemory() / 1024ul / 1024ul, useBytes / 1024ul / 1024ul,
               gemmModeName(d_mode));
        const size_t d_inputSize = sizeof(T) * SIZE * SIZE;
        d_resultSize = resultElementSize(d_mode) * SIZE * SIZE;
        const size_t minBytes = 2 * d_inputSize + d_resultSize;
        if ((size_t)useBytes < minBytes)
            throw std::string("Low mem for result. aborting.\n");
        d_iters = ((size_t)useBytes - 2 * d_inputSize) / d_resultSize;
        printf("Results are %zu bytes each, thus performing %zu iterations\n",
               d_resultSize, d_iters);
        checkError(cuMemAlloc(&d_Cdata, d_iters * d_resultSize), "C alloc");
        checkError(cuMemAlloc(&d_Adata, d_inputSize), "A alloc");
        checkError(cuMemAlloc(&d_Bdata, d_inputSize), "B alloc");

        checkError(cuMemAlloc(&d_faultyElemData, sizeof(int)), "faulty data");

        // Populating matrices A and B
        checkError(cuMemcpyHtoD(d_Adata, A, d_inputSize), "A -> device");
        checkError(cuMemcpyHtoD(d_Bdata, B, d_inputSize), "B -> device");

        initCompareKernel();
    }

    void compute() {
        bind();
        static const float alpha = 1.0f;
        static const float beta = 0.0f;
        static const double alphaD = 1.0;
        static const double betaD = 0.0;

        for (size_t i = 0; i < d_iters; ++i) {
            if (d_mode == TEST_FP64) {
                checkError(
                    cublasDgemm(d_cublas, CUBLAS_OP_N, CUBLAS_OP_N, SIZE, SIZE,
                                SIZE, &alphaD, (const double *)d_Adata, SIZE,
                                (const double *)d_Bdata, SIZE, &betaD,
                                (double *)d_Cdata + i * SIZE * SIZE, SIZE),
                    "DGEMM");
            } else {
#if CUBLAS_VERSION >= 11000
                checkError(
                    cublasGemmEx(d_cublas, CUBLAS_OP_N, CUBLAS_OP_N, SIZE,
                                 SIZE, SIZE, &alpha, (const void *)d_Adata,
                                 inputDataType(d_mode), SIZE,
                                 (const void *)d_Bdata, inputDataType(d_mode),
                                 SIZE, &beta,
                                 (void *)((float *)d_Cdata + i * SIZE * SIZE),
                                 CUDA_R_32F, SIZE, computeType(d_mode),
                                 CUBLAS_GEMM_DEFAULT),
                    gemmModeName(d_mode));
#else
                if (d_mode != TEST_FP32 && d_mode != TEST_TF32) {
                    throw std::runtime_error(
                        "BF16 and FP8 GEMM require cuBLAS 11 or newer");
                }
                checkError(
                    cublasSgemm(d_cublas, CUBLAS_OP_N, CUBLAS_OP_N, SIZE, SIZE,
                                SIZE, &alpha, (const float *)d_Adata, SIZE,
                                (const float *)d_Bdata, SIZE, &beta,
                                (float *)d_Cdata + i * SIZE * SIZE, SIZE),
                    "SGEMM");
#endif
            }
        }
    }

    void initCompareKernel() {
        {
            std::ifstream f(d_kernelFile);
            checkError(f.good() ? CUDA_SUCCESS : CUDA_ERROR_NOT_FOUND,
                       std::string("couldn't find compare kernel: ") + d_kernelFile);
        }
        checkError(cuModuleLoad(&d_module, d_kernelFile), "load module");
        checkError(cuModuleGetFunction(&d_function, d_module,
                                       usesDoubleOutput(d_mode) ? "compareD"
                                                                : "compare"),
                   "get func");

        checkError(cuFuncSetCacheConfig(d_function, CU_FUNC_CACHE_PREFER_L1),
                   "L1 config");
        const size_t faultyElemOffset = __alignof(void *);
        const size_t itersOffset = faultyElemOffset + __alignof(int *);
        checkError(cuParamSetSize(d_function, itersOffset + sizeof(size_t)),
                   "set param size");
        checkError(cuParamSetv(d_function, 0, &d_Cdata, sizeof(CUdeviceptr)),
                   "set param");
        checkError(cuParamSetv(d_function, faultyElemOffset, &d_faultyElemData,
                               sizeof(CUdeviceptr)),
                   "set param");
        checkError(cuParamSetv(d_function, itersOffset, &d_iters, sizeof(size_t)),
                   "set param");

        checkError(cuFuncSetBlockShape(d_function, g_blockSize, g_blockSize, 1),
                   "set block size");
    }

    void compare() {
        checkError(cuMemsetD32Async(d_faultyElemData, 0, 1, 0), "memset");
        checkError(cuLaunchGridAsync(d_function, SIZE / g_blockSize,
                                     SIZE / g_blockSize, 0),
                   "Launch grid");
        checkError(cuMemcpyDtoHAsync(d_faultyElemsHost, d_faultyElemData,
                                     sizeof(int), 0),
                   "Read faultyelemdata");
    }

    bool shouldRun() { return g_running; }

  private:
    void checkModeSupported() {
        int major, minor;
        checkError(cuDeviceGetAttribute(
                       &major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                       d_dev),
                   "device major capability");
        checkError(cuDeviceGetAttribute(
                       &minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
                       d_dev),
                   "device minor capability");

        if (d_mode == TEST_BF16 &&
            !computeCapabilityAtLeast(major, minor, 8, 0)) {
            throw std::runtime_error(
                "BF16 GEMM requires compute capability 8.0 or newer");
        }
        if (isFp8Mode(d_mode) &&
            !computeCapabilityAtLeast(major, minor, 8, 9)) {
            throw std::runtime_error(
                "FP8 GEMM requires compute capability 8.9 or newer");
        }
    }

    TestMode d_mode;
    int d_devNumber;
    const char *d_kernelFile;
    size_t d_iters;
    size_t d_resultSize;

    long long int d_error;

    static const int g_blockSize = 16;

    CUdevice d_dev;
    CUcontext d_ctx;
    CUmodule d_module;
    CUfunction d_function;

    CUdeviceptr d_Cdata;
    CUdeviceptr d_Adata;
    CUdeviceptr d_Bdata;
    CUdeviceptr d_faultyElemData;
    int *d_faultyElemsHost;

    cublasHandle_t d_cublas;
};

// Returns the number of devices
int initCuda() {
    try {
        CUresult initResult = cuInit(0);
        const char *initErrStr = "<unavailable>";
        if (cuGetErrorString(initResult, &initErrStr) != CUDA_SUCCESS ||
            initErrStr == nullptr) {
                initErrStr = "<unavailable>";
            }
        fprintf(stderr, "cuInit returned %d (%s)\n", initResult,
            initErrStr);
        checkError(initResult);
    } catch (std::runtime_error e) {
        fprintf(stderr, "Couldn't init CUDA: %s\n", e.what());
        return 0;
    }
    int deviceCount = 0;
    checkError(cuDeviceGetCount(&deviceCount));

    if (!deviceCount)
        throw std::string("No CUDA devices");

#ifdef USEDEV
    if (USEDEV >= deviceCount)
        throw std::string("Not enough devices for USEDEV");
#endif

    return deviceCount;
}

template <class T>
void startBurn(int index, int writeFd, T *A, T *B, TestMode mode,
               ssize_t useBytes, const char *kernelFile) {
    GPU_Test<T> *our;
    try {
        our = new GPU_Test<T>(index, mode, kernelFile);
        our->initBuffers(A, B, useBytes);
    } catch (const std::exception &e) {
        fprintf(stderr, "Couldn't init a GPU test: %s\n", e.what());
        exit(EMEDIUMTYPE);
    }

    // The actual work
    try {
        int eventIndex = 0;
        const int maxEvents = 2;
        CUevent events[maxEvents];
        for (int i = 0; i < maxEvents; ++i)
            cuEventCreate(events + i, 0);

        int nonWorkIters = maxEvents;

        while (our->shouldRun()) {
            our->compute();
            our->compare();
            checkError(cuEventRecord(events[eventIndex], 0), "Record event");

            eventIndex = ++eventIndex % maxEvents;

            while (cuEventQuery(events[eventIndex]) != CUDA_SUCCESS)
                usleep(1000);

            if (--nonWorkIters > 0)
                continue;

            int ops = our->getIters();
            write(writeFd, &ops, sizeof(int));
            ops = our->getErrors();
            write(writeFd, &ops, sizeof(int));
        }

        for (int i = 0; i < maxEvents; ++i)
            cuEventSynchronize(events[i]);
        delete our;
    } catch (const std::exception &e) {
        fprintf(stderr, "Failure during compute: %s\n", e.what());
        int ops = -1;
        // Signalling that we failed
        write(writeFd, &ops, sizeof(int));
        write(writeFd, &ops, sizeof(int));
        exit(ECONNREFUSED);
    }
}

int pollTemp(pid_t *p) {
    int tempPipe[2];
    pipe(tempPipe);

    pid_t myPid = fork();

    if (!myPid) {
        close(tempPipe[0]);
        dup2(tempPipe[1], STDOUT_FILENO);
#if IS_JETSON
        execlp("tegrastats", "tegrastats", "--interval", "5000", NULL);
        fprintf(stderr, "Could not invoke tegrastats, no temps available\n");
#else
        execlp("nvidia-smi", "nvidia-smi", "-l", "5", "-q", "-d", "TEMPERATURE",
               NULL);
        fprintf(stderr, "Could not invoke nvidia-smi, no temps available\n");
#endif

        exit(ENODEV);
    }

    *p = myPid;
    close(tempPipe[1]);

    return tempPipe[0];
}

void updateTemps(int handle, std::vector<int> *temps) {
    const int readSize = 10240;
    static int gpuIter = 0;
    char data[readSize + 1];

    int curPos = 0;
    do {
        read(handle, data + curPos, sizeof(char));
    } while (data[curPos++] != '\n');

    data[curPos - 1] = 0;

#if IS_JETSON
    std::string data_str(data);
    std::regex pattern("GPU@([0-9]+)C");
    std::smatch matches;
    if (std::regex_search(data_str, matches, pattern)) {
        if (matches.size() > 1) {
            int tempValue = std::stoi(matches[1]);
            temps->at(gpuIter) = tempValue;
            gpuIter = (gpuIter + 1) % (temps->size());
        }
    }
#else
    // FIXME: The syntax of this print might change in the future..
    int tempValue;
    if (sscanf(data,
               "		GPU Current Temp			: %d C",
               &tempValue) == 1) {
        temps->at(gpuIter) = tempValue;
        gpuIter = (gpuIter + 1) % (temps->size());
    } else if (!strcmp(data, "		Gpu				"
                             "	 : N/A"))
        gpuIter =
            (gpuIter + 1) %
            (temps->size()); // We rotate the iterator for N/A values as well
#endif
}

void listenClients(std::vector<int> clientFd, std::vector<pid_t> clientPid,
                   int runTime, std::chrono::seconds sigterm_timeout_threshold_secs) {
    fd_set waitHandles;

    pid_t tempPid;
    int tempHandle = pollTemp(&tempPid);
    int maxHandle = tempHandle;

    FD_ZERO(&waitHandles);
    FD_SET(tempHandle, &waitHandles);

    for (size_t i = 0; i < clientFd.size(); ++i) {
        if (clientFd.at(i) > maxHandle)
            maxHandle = clientFd.at(i);
        FD_SET(clientFd.at(i), &waitHandles);
    }

    std::vector<int> clientTemp;
    std::vector<int> clientErrors;
    std::vector<int> clientCalcs;
    std::vector<struct timespec> clientUpdateTime;
    std::vector<float> clientGflops;
    std::vector<bool> clientFaulty;

    time_t startTime = time(0);

    for (size_t i = 0; i < clientFd.size(); ++i) {
        clientTemp.push_back(0);
        clientErrors.push_back(0);
        clientCalcs.push_back(0);
        struct timespec thisTime;
        clock_gettime(CLOCK_REALTIME, &thisTime);
        clientUpdateTime.push_back(thisTime);
        clientGflops.push_back(0.0f);
        clientFaulty.push_back(false);
    }

    int changeCount;
    float nextReport = 10.0f;
    bool childReport = false;
    while (
        (changeCount = select(maxHandle + 1, &waitHandles, NULL, NULL, NULL))) {
        size_t thisTime = time(0);
        struct timespec thisTimeSpec;
        clock_gettime(CLOCK_REALTIME, &thisTimeSpec);

        // Going through all descriptors
        for (size_t i = 0; i < clientFd.size(); ++i)
            if (FD_ISSET(clientFd.at(i), &waitHandles)) {
                // First, reading processed
                int processed, errors;
                int res = read(clientFd.at(i), &processed, sizeof(int));
                if (res < sizeof(int)) {
                    fprintf(stderr, "read[%zu] error %d", i, res);
                    processed = -1;
                }
                // Then errors
                read(clientFd.at(i), &errors, sizeof(int));

                clientErrors.at(i) += errors;
                if (processed == -1)
                    clientCalcs.at(i) = -1;
                else {
                    double flops = (double)processed * (double)OPS_PER_MUL;
                    struct timespec clientPrevTime = clientUpdateTime.at(i);
                    double clientTimeDelta =
                        (double)thisTimeSpec.tv_sec +
                        (double)thisTimeSpec.tv_nsec / 1000000000.0 -
                        ((double)clientPrevTime.tv_sec +
                         (double)clientPrevTime.tv_nsec / 1000000000.0);
                    clientUpdateTime.at(i) = thisTimeSpec;

                    clientGflops.at(i) =
                        (double)((unsigned long long int)processed *
                                 OPS_PER_MUL) /
                        clientTimeDelta / 1000.0 / 1000.0 / 1000.0;
                    clientCalcs.at(i) += processed;
                }

                childReport = true;
            }

        if (FD_ISSET(tempHandle, &waitHandles))
            updateTemps(tempHandle, &clientTemp);

        // Resetting the listeners
        FD_ZERO(&waitHandles);
        FD_SET(tempHandle, &waitHandles);
        for (size_t i = 0; i < clientFd.size(); ++i)
            FD_SET(clientFd.at(i), &waitHandles);

        // Printing progress (if a child has initted already)
        if (childReport) {
            float elapsed =
                fminf((float)(thisTime - startTime) / (float)runTime * 100.0f,
                      100.0f);
            printf("\r%.1f%%  ", elapsed);
            printf("proc'd: ");
            for (size_t i = 0; i < clientCalcs.size(); ++i) {
                printf("%d (%.0f Gflop/s) ", clientCalcs.at(i),
                       clientGflops.at(i));
                if (i != clientCalcs.size() - 1)
                    printf("- ");
            }
            printf("  errors: ");
            for (size_t i = 0; i < clientErrors.size(); ++i) {
                std::string note = "%d ";
                if (clientCalcs.at(i) == -1)
                    note += " (DIED!)";
                else if (clientErrors.at(i))
                    note += " (WARNING!)";

                printf(note.c_str(), clientErrors.at(i));
                if (i != clientCalcs.size() - 1)
                    printf("- ");
            }
            printf("  temps: ");
            for (size_t i = 0; i < clientTemp.size(); ++i) {
                printf(clientTemp.at(i) != 0 ? "%d C " : "-- ",
                       clientTemp.at(i));
                if (i != clientCalcs.size() - 1)
                    printf("- ");
            }

            fflush(stdout);

            for (size_t i = 0; i < clientErrors.size(); ++i)
                if (clientErrors.at(i))
                    clientFaulty.at(i) = true;

            if (nextReport < elapsed) {
                nextReport = elapsed + 10.0f;
                printf("\n\tSummary at:   ");
                fflush(stdout);
                system("date"); // Printing a date
                fflush(stdout);
                printf("\n");
                for (size_t i = 0; i < clientErrors.size(); ++i)
                    clientErrors.at(i) = 0;
            }
        }

        // Checking whether all clients are dead
        bool oneAlive = false;
        for (size_t i = 0; i < clientCalcs.size(); ++i)
            if (clientCalcs.at(i) != -1)
                oneAlive = true;
        if (!oneAlive) {
            fprintf(stderr, "\n\nNo clients are alive!  Aborting\n");
            exit(ENOMEDIUM);
        }

        if (startTime + runTime < thisTime)
            break;
    }

    printf("\nKilling processes with SIGTERM (soft kill)\n");
    fflush(stdout);
    for (size_t i = 0; i < clientPid.size(); ++i)
        kill(clientPid.at(i), SIGTERM);

    kill(tempPid, SIGTERM);

    // processes should be terminated by SIGTERM within threshold time (so wait and then check pids)
    std::this_thread::sleep_for(sigterm_timeout_threshold_secs);

    // check each process and see if they are alive
    std::vector<int> killed_processes; // track the number of killed processes
    // loop through pids for each client / GPU
    for (size_t i = 0; i < clientPid.size(); ++i) {
        int status;
        pid_t return_pid = waitpid(clientPid.at(i), &status, WNOHANG);
        if (return_pid == clientPid.at(i)) {
            /* child is finished. exit status in status */
            killed_processes.push_back(return_pid);
        }
    }
    // handle the tempPid
    int status;
    pid_t return_pid = waitpid(tempPid, &status, WNOHANG);
    if (return_pid == tempPid) {
        /* child is finished. exit status in status */
        killed_processes.push_back(return_pid);
    }

    // number of killed process should be number GPUs + 1 (need to add tempPid process) to exit while loop early
    if (killed_processes.size() != clientPid.size() + 1) {
        printf("\nKilling processes with SIGKILL (force kill)\n");

        for (size_t i = 0; i < clientPid.size(); ++i) {
            // check if pid was already killed with SIGTERM before using SIGKILL
            if (std::find(killed_processes.begin(), killed_processes.end(), clientPid.at(i)) == killed_processes.end())
                kill(clientPid.at(i), SIGKILL);
        }

        // check if pid was already killed with SIGTERM before using SIGKILL
        if (std::find(killed_processes.begin(), killed_processes.end(), tempPid) == killed_processes.end())
            kill(tempPid, SIGKILL);
    }

    close(tempHandle);

    while (wait(NULL) != -1)
        ;
    printf("done\n");

    printf("\nTested %d GPUs:\n", (int)clientPid.size());
    for (size_t i = 0; i < clientPid.size(); ++i)
        printf("\tGPU %d: %s\n", (int)i, clientFaulty.at(i) ? "FAULTY" : "OK");
}

template <class T>
void launch(int runLength, TestMode mode, ssize_t useBytes, int device_id,
            const char * kernelFile,
            std::chrono::seconds sigterm_timeout_threshold_secs) {
#if IS_JETSON
    std::ifstream f_model("/proc/device-tree/model");
    std::stringstream ss_model;
    ss_model << f_model.rdbuf();
    printf("%s\n", ss_model.str().c_str());
#else
    system("nvidia-smi -L");
#endif

    // Initting A and B with random data
    T *A = (T *)malloc(sizeof(T) * SIZE * SIZE);
    T *B = (T *)malloc(sizeof(T) * SIZE * SIZE);
    srand(10);
    for (size_t i = 0; i < SIZE * SIZE; ++i) {
        A[i] = convertInput<T>((float)((double)(rand() % 1000000) / 100000.0),
                               mode);
        B[i] = convertInput<T>((float)((double)(rand() % 1000000) / 100000.0),
                               mode);
    }

    // Forking a process..  This one checks the number of devices to use,
    // returns the value, and continues to use the first one.
    int mainPipe[2];
    pipe(mainPipe);
    int readMain = mainPipe[0];
    std::vector<int> clientPipes;
    std::vector<pid_t> clientPids;
    clientPipes.push_back(readMain);

    if (device_id > -1) {
        pid_t myPid = fork();
        if (!myPid) {
            // Child
            close(mainPipe[0]);
            int writeFd = mainPipe[1];
            initCuda();
            int devCount = 1;
            write(writeFd, &devCount, sizeof(int));
            startBurn<T>(device_id, writeFd, A, B, mode, useBytes, kernelFile);
            close(writeFd);
            return;
        } else {
            clientPids.push_back(myPid);
            close(mainPipe[1]);
            int devCount;
            read(readMain, &devCount, sizeof(int));
            listenClients(clientPipes, clientPids, runLength, sigterm_timeout_threshold_secs);
        }
        for (size_t i = 0; i < clientPipes.size(); ++i)
            close(clientPipes.at(i));
    } else {
        pid_t myPid = fork();
        if (!myPid) {
            // Child
            close(mainPipe[0]);
            int writeFd = mainPipe[1];
            int devCount = initCuda();
            write(writeFd, &devCount, sizeof(int));

            startBurn<T>(0, writeFd, A, B, mode, useBytes, kernelFile);

            close(writeFd);
            return;
        } else {
            clientPids.push_back(myPid);

            close(mainPipe[1]);
            int devCount;
            read(readMain, &devCount, sizeof(int));

            if (!devCount) {
                fprintf(stderr, "No CUDA devices\n");
                exit(ENODEV);
            } else {
                for (int i = 1; i < devCount; ++i) {
                    int slavePipe[2];
                    pipe(slavePipe);
                    clientPipes.push_back(slavePipe[0]);

                    pid_t slavePid = fork();

                    if (!slavePid) {
                        // Child
                        close(slavePipe[0]);
                        initCuda();
                        startBurn<T>(i, slavePipe[1], A, B, mode, useBytes,
                                     kernelFile);

                        close(slavePipe[1]);
                        return;
                    } else {
                        clientPids.push_back(slavePid);
                        close(slavePipe[1]);
                    }
                }

                listenClients(clientPipes, clientPids, runLength, sigterm_timeout_threshold_secs);
            }
        }
        for (size_t i = 0; i < clientPipes.size(); ++i)
            close(clientPipes.at(i));
    }

    free(A);
    free(B);
}

void showHelp() {
    printf("GPU Burn\n");
    printf("Usage: gpu-burn [OPTIONS] [TIME]\n\n");
    printf("-m X\tUse X MB of memory.\n");
    printf("-m N%%\tUse N%% of the available GPU memory.  Default is %d%%\n",
           (int)(USEMEM * 100));
    printf("-d\tUse doubles\n");
    printf("-tc\tUse TF32 Tensor Core compute for float GEMM\n");
    printf("-bf16\tUse BF16 Tensor Core compute with FP32 output\n");
    printf("-fp8\tUse FP8 E4M3 Tensor Core compute with FP32 output\n");
    printf("-fp8-e5m2\tUse FP8 E5M2 Tensor Core compute with FP32 output\n");
    printf("-l\tLists all GPUs in the system\n");
    printf("-i N\tExecute only on GPU N\n");
    printf("-c FILE\tUse FILE as compare kernel.  Default is %s\n",
           COMPARE_KERNEL);
    printf("-stts T\tSet timeout threshold to T seconds for using SIGTERM to abort child processes before using SIGKILL.  Default is %d\n",
           SIGTERM_TIMEOUT_THRESHOLD_SECS);
    printf("-h\tShow this help message\n\n");
    printf("Examples:\n");
    printf("  gpu-burn -d 3600 # burns all GPUs with doubles for an hour\n");
    printf(
        "  gpu-burn -m 50%% # burns using 50%% of the available GPU memory\n");
    printf("  gpu-burn -l # list GPUs\n");
    printf("  gpu-burn -i 2 # burns only GPU of index 2\n");
}

// NNN MB
// NN% <0
// 0 --- error
ssize_t decodeUSEMEM(const char *s) {
    char *s2;
    int64_t r = strtoll(s, &s2, 10);
    if (s == s2)
        return 0;
    if (*s2 == '%')
        return (s2[1] == 0) ? -r : 0;
    return (*s2 == 0) ? r * 1024 * 1024 : 0;
}

int main(int argc, char **argv) {
    int runLength = 10;
    TestMode testMode = TEST_FP32;
    bool modeSelected = false;
    int thisParam = 0;
    ssize_t useBytes = 0; // 0 == use USEMEM% of free mem
    int device_id = -1;
    char *kernelFile = (char *)COMPARE_KERNEL;
    std::chrono::seconds sigterm_timeout_threshold_secs = std::chrono::seconds(SIGTERM_TIMEOUT_THRESHOLD_SECS);

    std::vector<std::string> args(argv, argv + argc);
    for (size_t i = 1; i < args.size(); ++i) {
        if (argc >= 2 && std::string(argv[i]).find("-h") != std::string::npos) {
            showHelp();
            return 0;
        }
        if (argc >= 2 && std::string(argv[i]).find("-l") != std::string::npos) {
            int count = initCuda();
            if (count == 0) {
                throw std::runtime_error("No CUDA capable GPUs found.\n");
            }
            for (int i_dev = 0; i_dev < count; i_dev++) {
                CUdevice device_l;
                char device_name[255];
                checkError(cuDeviceGet(&device_l, i_dev));
                checkError(cuDeviceGetName(device_name, 255, device_l));
                size_t device_mem_l;
                checkError(cuDeviceTotalMem(&device_mem_l, device_l));
                printf("ID %i: %s, %ldMB\n", i_dev, device_name,
                       device_mem_l / 1000 / 1000);
            }
            thisParam++;
            return 0;
        }
        if (argc >= 2 && args[i] == "-d") {
            selectTestMode(&testMode, &modeSelected, TEST_FP64, argv[i]);
            thisParam++;
            continue;
        }
        if (argc >= 2 && args[i] == "-tc") {
            selectTestMode(&testMode, &modeSelected, TEST_TF32, argv[i]);
            thisParam++;
            continue;
        }
        if (argc >= 2 && args[i] == "-bf16") {
            selectTestMode(&testMode, &modeSelected, TEST_BF16, argv[i]);
            thisParam++;
            continue;
        }
        if (argc >= 2 && (args[i] == "-fp8" || args[i] == "-fp8-e4m3")) {
            selectTestMode(&testMode, &modeSelected, TEST_FP8_E4M3, argv[i]);
            thisParam++;
            continue;
        }
        if (argc >= 2 && args[i] == "-fp8-e5m2") {
            selectTestMode(&testMode, &modeSelected, TEST_FP8_E5M2, argv[i]);
            thisParam++;
            continue;
        }
        if (argc >= 2 && strncmp(argv[i], "-m", 2) == 0) {
            thisParam++;

            // -mNNN[%]
            // -m NNN[%]
            if (argv[i][2]) {
                useBytes = decodeUSEMEM(argv[i] + 2);
            } else if (i + 1 < args.size()) {
                i++;
                thisParam++;
                useBytes = decodeUSEMEM(argv[i]);
            } else {
                fprintf(stderr, "Syntax error near -m\n");
                exit(EINVAL);
            }
            if (useBytes == 0) {
                fprintf(stderr, "Syntax error near -m\n");
                exit(EINVAL);
            }
        }
        if (argc >= 2 && strncmp(argv[i], "-i", 2) == 0) {
            thisParam++;

            if (argv[i][2]) {
                device_id = strtol(argv[i] + 2, NULL, 0);
            } else if (i + 1 < args.size()) {
                i++;
                thisParam++;
                device_id = strtol(argv[i], NULL, 0);
            } else {
                fprintf(stderr, "Syntax error near -i\n");
                exit(EINVAL);
            }
        }
        if (argc >= 2 && strncmp(argv[i], "-c", 2) == 0) {
            thisParam++;

            if (argv[i + 1]) {
                kernelFile = argv[i + 1];
                thisParam++;
            }
        }
        if (argc >= 2 && strncmp(argv[i], "-stts", 2) == 0) {
            thisParam++;

            if (argv[i + 1]) {
                sigterm_timeout_threshold_secs = std::chrono::seconds(atoi(argv[i + 1]));
                thisParam++;
            }
        }
    }

    if (argc - thisParam < 2)
        printf("Run length not specified in the command line. ");
    else
        runLength = atoi(argv[1 + thisParam]);
    printf("Using compare file: %s\n", kernelFile);
    printf("Burning for %d seconds.\n", runLength);

    switch (testMode) {
    case TEST_FP64:
        launch<double>(runLength, testMode, useBytes, device_id, kernelFile,
                       sigterm_timeout_threshold_secs);
        break;
    case TEST_BF16:
        launch<uint16_t>(runLength, testMode, useBytes, device_id, kernelFile,
                         sigterm_timeout_threshold_secs);
        break;
    case TEST_FP8_E4M3:
    case TEST_FP8_E5M2:
        launch<uint8_t>(runLength, testMode, useBytes, device_id, kernelFile,
                        sigterm_timeout_threshold_secs);
        break;
    case TEST_FP32:
    case TEST_TF32:
    default:
        launch<float>(runLength, testMode, useBytes, device_id, kernelFile,
                      sigterm_timeout_threshold_secs);
        break;
    }

    return 0;
}
