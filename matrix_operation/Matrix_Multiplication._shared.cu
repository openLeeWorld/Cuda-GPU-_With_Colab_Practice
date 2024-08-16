%%writefile Matrix_Multiplcation_Shared.cu

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define BLOCK_SIZE 16
#define ROW_SIZE 32
#define COL_SIZE 32
#define K_SIZE 128

__global__ void matMul_kernel_shared // 1024개의 스레드로 한 개 블록 이내의 경우
(float* _A, float* _B, float* _C)
// _A, _B, _C는 행렬 포인터, A 차원은 32 * 128, B 차원은 128 * 32, C 차원은 32 * 32
{
    int row = threadIdx.x;
    int col = threadIdx.y;
    int index = row * blockDim.y + col;

    __shared__ float sA[ROW_SIZE][K_SIZE]; // 23*256*4 bytes = 16KB
    __shared__ float sB[K_SIZE][COL_SIZE]; // 16KB
    // 합계 32kb 는 48KB~96KB (GPU 공유 메모리)보다 작으므로 가능(공유 메모리 정적 할당)

    if (row == 0) { // 첫 row 스레드가 read matrix B의 column을 다 공유 메모리에 넣음 
        for (int k = 0; k < K_SIZE; k++) sB[k][col] = _B[col + k * COL_SIZE];
    }

    if (col == 0) { // 첫 col 스레드가 read matrix A의 row을 다 공유 메모리에 넣음 
        for (int k = 0; k < K_SIZE; k++) sA[row][k] = _B[row * K_SIZE + k];
    }

    __syncthreads(); // wait until all threads load the matrix

    float result = 0;
    for (int k = 0; k < K_SIZE; k++) result += sA[row][k] * sB[k][col];
    _C[index] = result;
}


int main(int argc, char* argv[]) {

    int sizeA = ROW_SIZE * K_SIZE;
    int sizeB = K_SIZE * COL_SIZE;
    int sizeC = ROW_SIZE * COL_SIZE;

    int* dA, *dB, *dC; 

    // 1. Allocate device memory for dA, dB dC
    cudaMalloc(&dA, sizeA * sizeof(int)); cudaMemset(dA, 0, sizeA * sizeof(int));
    cudaMalloc(&dB, sizeB * sizeof(int)); cudaMemset(dB, 0, sizeB * sizeof(int));
    cudaMalloc(&dC, sizeC * sizeof(int)); cudaMemset(dC, 0, sizeC * sizeof(int));

    // 2. Send(Copy) tje input matrices to GPU (A -> dA, B -> dB)
    cudaMemcpy(dA, A, sizeA * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B, sizeB * sizeof(int), cudaMemcpyHostToDevice);

    // 3. Set the thread layout
    dim3 gridDim(ceil((float)ROW_SIZE / BLOCK_SIZE), ceil((float)COL_SIZE / BLOCK_SIZE));
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);

    // 4. kernel call
    matMul_kernel_shared <<< gridDim, blockDim >>> (dA, dB, dC);
    cudaDeviceSynchronize();

    // 5. Get(Copy) the result from GPU to host memory (dC  -> C)
    cudaMemcpy(C, dC, sizeC * sizeof(int), cudaMemcpyDeviceToHost);

    // 6. Release device memory space
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);

    return 0;
}