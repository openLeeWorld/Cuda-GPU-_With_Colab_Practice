#include "kernelCall.h"

void main() {
    kernelCall(); // 헤더파일의 원형 함수 호출
    printf("Host code running on CPU\n");
    cudaDeviceSynchronize();
}