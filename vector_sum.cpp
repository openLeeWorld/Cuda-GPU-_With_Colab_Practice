#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <chrono>

#define NUM_DATA 1024

using namespace std;

int main(void)
{
    auto start = std::chrono::high_resolution_clock::now();
    
    // 실행 시간을 측정하고 싶은 코드를 여기에 작성합니다.
    int *a, *b, *c;

    int memSize = sizeof(int) * NUM_DATA; // 4KB
    a = new int[NUM_DATA]; memset(a, 0, memSize);
    b = new int[NUM_DATA]; memset(b, 0, memSize);
    c = new int[NUM_DATA]; memset(c, 0, memSize);

    for (int i = 0; i < NUM_DATA; i++) {
        a[i] = rand() % 10;
        //cout << "a[" << i << "]: " << a[i] << "\n";
        b[i] = rand() % 10;
        //cout << "b[" << i << "]: " << b[i] << "\n";
    }

    for (int i = 0; i < NUM_DATA; i++) {
        c[i] = a[i] + b[i];
        //cout << "c[" << i << "]: " << c[i] << "\n";
    }

    delete[] a; delete[] b; delete[] c;

    auto end = std::chrono::high_resolution_clock::now();
    
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Elapsed time: " << elapsed.count() << " seconds" << std::endl;

    return 0;
}