%%writefile Matrix_Multiplcation.cu

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define BLOCK_SIZE 16

__global__ void matMul_kernel_lowerThan1024ver
(int* A, int* B, int* C, int m, int n, int k)
// A, B, C는 행렬 포인터, A의 차원: m * k, B의 차원: k * n, C의 차원: m * n
{
    int row = threadIdx.x;
    int col = threadIdx.y;
    int index = row * n + col;

    if (row >= m || col >= n) return;

    C[index] = 0;
    for (int offest = 0; offset < k; offset++) {
        C[index] += A[row * k + offset] * B[col + offset * n];
    }
}

__global__ void matMul_kernel_higherThan1024ver
(int* A, int* B, int* C, int m, int n, int k)
// A, B, C는 행렬 포인터, A의 차원: m * k, B의 차원: k * n, C의 차원: m * n
{
    int row = (blockDim.x * blockIdx.x) + threadIdx.x;
    int col = (blockDim.y * blockIdx.y) + threadIdx.y;
    int index = row * n + col;

    if (row >= m || col >= n) return;

    C[index] = 0;
    for (int offest = 0; offset < k; offset++) {
        C[index] += A[row * k + offset] * B[col + offset * n];
    }
}

int main(int argc, char* argv[]) {

    int m,n,k;
    m = atoi(argv[1]); n = atoi(argv[2]); k = atoi(argv[3]);

    int sizeA = m * k;
    int sizeB = k * n;
    int sizeC = m * n;

    int* dA, *dB, *dC; 

    // 1. Allocate device memory for dA, dB dC
    cudaMalloc(&dA, sizeA * sizeof(int)); cudaMemset(dA, 0, sizeA * sizeof(int));
    cudaMalloc(&dB, sizeB * sizeof(int)); cudaMemset(dB, 0, sizeB * sizeof(int));
    cudaMalloc(&dC, sizeC * sizeof(int)); cudaMemset(dC, 0, sizeC * sizeof(int));

    // 2. Send(Copy) tje input matrices to GPU (A -> dA, B -> dB)
    cudaMemcpy(dA, A, sizeA * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B, sizeB * sizeof(int), cudaMemcpyHostToDevice);

    // 3. Set the thread layout
    dim3 gridDim(ceil((float)m / BLOCK_SIZE), ceil((float)n / BLOCK_SIZE));
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);

    // 4. kernel call
    matMul_kernel_higherThan1024ver <<< gridDim, blockDim >>> (dA, dB, dC, m, n, k);
    cudaDeviceSynchronize();

    // 5. Get(Copy) the result from GPU to host memory (dC  -> Cgpu)
    cudaMemcpy(Cgpu, dC, sizeC * sizeof(int), cudaMemcpyDeviceToHost);

    // 6. Release device memory space
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);


    return 0;
}