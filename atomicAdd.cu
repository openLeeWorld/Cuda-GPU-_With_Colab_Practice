%%writefile atomicAdd.cu

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

__global__ void threadCounting_noSync(int *a)
{
    (*a)++;
}

__global__ void threadCounting_atomicGlobal(int *a)
{
    atomicAdd(a, 1);
}

__global__ void threadCounting_atomicShared(int *a)
{
    __shared__ int sa; // 블록 당 할당된 공유 메모리
    if (threadIdx.x == 0) sa = 0;
    // 대표 스레드에서 공유 메모리 초기화
    __syncthreads(); // barrier for all initialization

    atomicAdd(&sa, 1); //block-level counting

    __syncthreads(); // barrier for all operations

    if (threadIdx.x == 0) atomicAdd(a, sa);
    // grid-level counting
    // 각 블록에서 하나의 스레드만 원자 함수를 호출하므로 
    // 동기화 참여 스레드 수는 블록의 수와 같음
}

int main() {
    int *noSyncKernels;
    int host_noSync = 0;

    cudaMalloc(&noSyncKernels, sizeof(int));
    cudaMemset(noSyncKernels, 0, sizeof(int));
    
    auto start_nosync = std::chrono::high_resolution_clock::now();

    threadCounting_noSync <<< GRID_SIZE, BLOCK_SIZE >>> (noSyncKernels);
    cudaDeviceSynchronize();  // Ensure the kernel completes before the program exits

    auto end_nosync = std::chrono::high_resolution_clock::now();

    cudaError_t err = cudaGetLastError();  // Check for kernel launch errors
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    cudaMemcpy(&host_noSync, noSyncKernels, sizeof(int), cudaMemcpyDeviceToHost);

    std::chrono::duration<double> elapsed_nosync = end_nosync - start_nosync;

    printf("[No sync] # of threads = %d\n", host_noSync);
    std::cout << "No Sync Elapsed time: " << elapsed_nosync.count() * 1000 << " ms" << std::endl;

    cudaFree(noSyncKernels);

    ///////////////////////////////////////////////////////////////////

    int *SyncKernels;
    int host_Sync = 0;

    cudaMalloc(&SyncKernels, sizeof(int));
    cudaMemset(SyncKernels, 0, sizeof(int));

    auto start_sync = std::chrono::high_resolution_clock::now();

    threadCounting_atomicGlobal <<< GRID_SIZE, BLOCK_SIZE >>> (SyncKernels);
    cudaDeviceSynchronize();  // Ensure the kernel completes before the program exits

    auto end_sync = std::chrono::high_resolution_clock::now();

    cudaError_t err2 = cudaGetLastError();  // Check for kernel launch errors
    if (err2 != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err2));
        return 1;
    }

    cudaMemcpy(&host_Sync, SyncKernels, sizeof(int), cudaMemcpyDeviceToHost);

    std::chrono::duration<double> elapsed_sync = end_sync - start_sync;

    printf("[Atomic Global] # of threads = %d\n", host_Sync);
    std::cout << "Atomic Elapsed time: " << elapsed_sync.count() * 1000 << " ms" << std::endl;

    cudaFree(SyncKernels);

    //////////////////////////////////////////////////////////////////

    int *SyncSharedKernels;
    int host_Sync_Shared = 0;

    cudaMalloc(&SyncSharedKernels, sizeof(int));
    cudaMemset(SyncSharedKernels, 0, sizeof(int));

    auto start_sync_shared = std::chrono::high_resolution_clock::now();

    threadCounting_atomicGlobal <<< GRID_SIZE, BLOCK_SIZE >>> (SyncSharedKernels);
    cudaDeviceSynchronize();  // Ensure the kernel completes before the program exits

    auto end_sync_shared = std::chrono::high_resolution_clock::now();

    cudaError_t err3 = cudaGetLastError();  // Check for kernel launch errors
    if (err3 != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err3));
        return 1;
    }

    cudaMemcpy(&host_Sync_Shared, SyncSharedKernels, sizeof(int), cudaMemcpyDeviceToHost);

    std::chrono::duration<double> elapsed_sync_shared = end_sync_shared - start_sync_shared;

    printf("[Atomic Shared] # of threads = %d\n", host_Sync_Shared);
    std::cout << "Shared Atomic Elapsed time: " << elapsed_sync_shared.count() * 1000 << " ms" << std::endl;

    cudaFree(SyncSharedKernels);
    
    return 0;
}
