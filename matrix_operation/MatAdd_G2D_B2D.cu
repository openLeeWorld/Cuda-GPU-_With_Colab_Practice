// 2차원 그리드, 2차원 블록 스레드 레이아웃에서 크기가 1024이상인 대규모 행렬 합
%%writefile MatAdd_G2D_B2D.cu
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
//#include "CUDA_definitions.cuh"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

__global__ void 
(float* MatA, float* MatB, float* MatC, int ROW_SIZE, int COL_SIZE) 
{
    unsigned int col = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int row = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int index = row * COL_SIZE + col;

    if (col < COL_SIZE && row < ROW_SIZE) MatC[index] = MatA[index] + MatB[index]; 
}

int main() {
    dim3 blockDim(32, 32); // 블럭당 최대 스레드 개수 1024
    dim3 gridDim(ceil((float)COL_SIZE / blockDim.x), ceil((float)ROW_SIZE / blockDim.y));
    MatAdd_G2D_B2D <<<gridDim, blockDim >>> (A, B, C, ROW_SIZE, COL_SIZE);
    cudaDeviceSynchronize();

    return 0;
}
