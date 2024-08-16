#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

void kernelCall(void);
// .cu 파일에 대한 커널 함수 원형 선언