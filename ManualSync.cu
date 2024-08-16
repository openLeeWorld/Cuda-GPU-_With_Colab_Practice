%%writefile mySyncKernel.cu

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <iostream>
#include <chrono>

#define GRID_SIZE 128 * 4
#define BLOCK_SIZE 1024

__global__ void mySyncKernel(int* input, int* output)
{
    // 블록 내부에서 짝수 스레드와 홀수 스레드가 서로 다른 작업을 수행
    // 짝수 스레드들은 다른 모든 짝수 스레드가 작업을 완료할 때까지 대기
    __shared__ int lock;

    if (threadIdx.x % 2 == 0) {
        // work for even threads
        do {
            atomicInc(&lock);
        } while (lock < blockDim.x / 2); 
        // spin lock 방식: lock 변수가 원하는 값이 되었는지 계속 검사
        // lock 변수의 값이 짝수 스레드와 같아지면 다음으로 진행하여 스레드 공통 작업 수행
    } else {
        // work for odd threads
    }

    // common work for all threads
    __syncthreads();

    // next step
}

int main() {
    //mySyncKernel 호출
    
    return 0;
}
