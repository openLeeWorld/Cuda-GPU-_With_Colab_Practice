%%writefile MultiStreamAsync.cu
// 비동기로 동작 가능한 멀티 스트림을 cuda event로 수행 시간을 측정한다.
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <iostream>
#include <chrono>

#define NUM_BLOCK (128 * 1024)
#define ARRAY_SIZE (1024 * NUM_BLOCK)
#define NUM_STREAMS 4
#define WORK_LOAD 256

__global__ void myKernel(int *_in, int* _out)
{
    int tID = blockDim.x * blockDim.x + threadIdx.x;

    int temp = 0;
    int in = _in[tID];
    for (int i = 0; i < WORK_LOAD; i++) {
        temp = (temp + in % 5) % 10;
    }
    _out[tID] = temp;
} // 단순 내부 연산 시키기

int main() {
    
    int *in = NULL, *out = NULL, *dIn = NULL, *dOut = NULL;

    cudaMallocHost(&in, sizeof(int) * ARRAY_SIZE); // pinned memory
    memset(in, 0, sizeof(int) * ARRAY_SIZE); // host 메모리 초기화

    cudaMallocHost(&out, sizeof(int) * ARRAY_SIZE); // pinned memory
    memset(out, 0, sizeof(int) * ARRAY_SIZE); // host 메모리 초기화

    cudaMalloc(&dIn, sizeof(int) * ARRAY_SIZE);
    cudaMalloc(&dOut, sizeof(int) * ARRAY_SIZE); // gpu global memoey 할당
    
    for (int i = 0; i < ARRAY_SIZE; i++) in[i] = rand() % 10; // 배열 초기화

    // single stream version (동기)
    cudaMemcpy(dIn, in, sizeof(int) * ARRAY_SIZE, cudaMemcpyHostToDevice);
    myKernel <<< NUM_BLOCK, 1024 >>> (dIn, dOut);
    cudaMemcpy(out, dOut, sizeof(int) * ARRAY_SIZE, cudaMemcpyDeviceToHost);

    // multi-stream version
    cudaStream_t stream[NUM_STREAMS]; // Non-Null 스트림 변수 선언 (배열)
    
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamCreate(&stream[i]); // 스트림 생성
    }
    
    int chunkSize = ARRAY_SIZE / NUM_STREAMS; // 스트림 당 데이터

    for (int i = 0; i < NUM_STREAMS; i++) {
        int offset = chunkSize * i;
        cudaMemcpyAsync(dIn + offset, in + offset, sizeof(int) * chunkSize, 
            cudaMemcpyHostToDevice, stream[i]);
    } // 각 스트림 당 호스트에서 디바이스로 메모리 옮김

    for (int i = 0; i < NUM_STREAMS; i++) {
        int offset = chunkSize * i;
        myKernel <<< NUM_BLOCK / NUM_STREAMS, 1024, 0, stream[i] >>> 
            (dIn + offset, dOut + offset);
    } // 각 스트림 당 해당 메모리 영역에 대해 커널 수행

    for (int i = 0; i < NUM_STREAMS; i++) {
        int offset = chunkSize * i;
        cudaMemcpyAsync(out + offset, dOut + offset, sizeof(int) * chunkSize, 
            cudaMemcpyDeviceToHost, stream[i]);
    } // 각 스트림 당 디바이스에서 호스트로 결과 메모리 옮김

    cudaDeviceSynchronize();

    for (int i = 0; i < NUM_STREAMS; i++) cudaStreamDestroy(stream[i]); // 스트림 제거

    cudaFree(dIn);
    cudaFree(dOut); // gpu 메모리 할당 해제
    cudaFreeHost(in);
    cudaFreeHost(out); // pinned memory 해제

    return 0;
}
