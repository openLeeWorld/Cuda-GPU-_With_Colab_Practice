%%writefile Matrix_Multiplcation_Shared_Large.cu

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define BLOCK_SIZE 16

__global__ void matMul_kernel_shared_large
(int* matA, int* matB, int* matC, int m, int n, int k)
// A, B, C는 행렬 포인터, A의 차원: m * k, B의 차원: k * n, C의 차원: m * n
{
    int row = (blockDim.x * blockIdx.x) + threadIdx.x;
    int col = (blockDim.y * blockIdx.y) + threadIdx.y;

    int val = 0;
    __shared__ int subA[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ int subB[BLOCK_SIZE][BLOCK_SIZE];

    int localRow = threadIdx.x;
    int localCol = threadIdx.y;

    for (int bID = 0; bID < ceil((float)k / BLOCK_SIZE); bID++) {
        int stride = bId * BLOCK_SIZE;

        if (row >= m || stride + localCol >= k) subA[localRow][localCol] = 0;
        else subA[localRow][localCol] = matA[row * k + (stride + localCol)];

        if (col >= n || stride + localRow >= k) subB[localRow][localCol] = 0;
        else subB[localRow][localCol] = matB[(stride + localRow) * n + col];

        __syncthreads(); // 모든 데이터의 복사가 완료될 때까지 대기

        for (int i = 0; i < BLOCK_SIZE; i++) {
            val += subA[localRow][i] * subB[i][localCol];
        } // 서브 블록 행렬 계산 (C(localRow, localCol))
        __syncthreads(); // 모든 스레드 계산 완료 대기
    }

    if (row >= m || col >= n) return;

    matC[row * n + col] = val;
}

int main(int argc, char* argv[]) {

    int m,n,k; // 1024로 시도하기
    m = atoi(argv[1]); n = atoi(argv[2]); k = atoi(argv[3]);

    int sizeA = m * k;
    int sizeB = k * n;
    int sizeC = m * n;

    int* dA, *dB, *dC; 

    // 1. Allocate device memory for dA, dB dC
    cudaMalloc(&dA, sizeA * sizeof(int)); cudaMemset(dA, 0, sizeA * sizeof(int));
    cudaMalloc(&dB, sizeB * sizeof(int)); cudaMemset(dB, 0, sizeB * sizeof(int));
    cudaMalloc(&dC, sizeC * sizeof(int)); cudaMemset(dC, 0, sizeC * sizeof(int));

    // 2. Send(Copy) the input matrices to GPU (A -> dA, B -> dB)
    cudaMemcpy(dA, A, sizeA * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B, sizeB * sizeof(int), cudaMemcpyHostToDevice);

    // 3. Set the thread layout
    dim3 gridDim(ceil((float)m / BLOCK_SIZE), ceil((float)n / BLOCK_SIZE));
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);

    // 4. kernel call
    matMul_kernel_shared_large <<< gridDim, blockDim >>> (dA, dB, dC, m, n, k);
    cudaDeviceSynchronize();

    // 5. Get(Copy) the result from GPU to host memory (dC  -> C)
    cudaMemcpy(C, dC, sizeC * sizeof(int), cudaMemcpyDeviceToHost);

    // 6. Release device memory space
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);

    return 0;
}