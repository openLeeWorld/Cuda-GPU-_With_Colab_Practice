%%writefile StreamEvent.cu
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

    // multi-stream version
    cudaStream_t stream[NUM_STREAMS]; // Non-Null 스트림 변수 선언 (배열)
    cudaEvent_t start[NUM_STREAMS], end[NUM_STREAMS]; // cuda 이벤트 변수 선언 (이벤트 배열)
    
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamCreate(&stream[i]); // 스트림 생성
        cudaEventCreate(&start[i]); cudaEventCreate(&end[i]); // 이벤트 생성
    }
    
    int chunkSize = ARRAY_SIZE / NUM_STREAMS; // 스트림 당 데이터

    int offset[NUM_STREAMS] = {0};
    for (int i = 0; i < NUM_STREAMS; i++) offset[i] = chunkSize * i;

    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaEventRecord(start[i], stream[i]); // cuda event 기록: cuda event를 stream에 넣음

        cudaMemcpyAsync(dIn + offset[i], in + offset[i], sizeof(int) * chunkSize, 
            cudaMemcpyHostToDevice, stream[i]);
    } // 각 스트림 당 호스트에서 디바이스로 메모리 옮김

    for (int i = 0; i < NUM_STREAMS; i++) {
        myKernel <<< NUM_BLOCK / NUM_STREAMS, 1024, 0, stream[i] >>> 
            (dIn + offset[i], dOut + offset[i]);
    } // 각 스트림 당 해당 메모리 영역에 대해 커널 수행

    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaMemcpyAsync(out + offset[i], dOut + offset[i], sizeof(int) * chunkSize, 
            cudaMemcpyDeviceToHost, stream[i]);

        cudaEventRecord(end[i], stream[i]); // 각 스트림 당 end event를 넣음
    } // 각 스트림 당 디바이스에서 호스트로 결과 메모리 옮김

    cudaDeviceSynchronize();

    for (int i = 0; i < NUM_STREAMS; i++) {
        if (cudaEventQuery(start[i]) == cudaSuccess 
            && cudaEventQuery(end[i]) == cudaSuccess) {
                float time = 0;
                cudaEventElapsedTime(&time, start[i], end[i]);
                printf("Stream[%d] : %f ms\n", i, time);
            } // 이벤트가 성공적으로 일어났다면
        else {
            printf("Event has not occured!");
        }
    }

    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamDestroy(stream[i]); // 스트림 제거
        cudaEventDestroy(start[i]); cudaEventDestroy(end[i]); // 이벤트 제거
    }

    cudaFree(dIn);
    cudaFree(dOut); // gpu 메모리 할당 해제
    cudaFreeHost(in);
    cudaFreeHost(out); // pinned memory 해제

    return 0;
}
