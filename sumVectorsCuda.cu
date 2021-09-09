#define LEN_ARR 100000000
#define BLOCK_SIZE 500
#define N_BLOCKS 500

#include <stdio.h>
#include <stdlib.h>

__global__ void sumVector(float *a, float *b, float *result, int n) {
    int firstIndex = blockIdx.x * blockDim.x + threadIdx.x;
    for(int i = firstIndex; i < n; i += BLOCK_SIZE * N_BLOCKS) {
        result[i] = a[i] + b[i];
    }
}

int main() {
    float *a = (float*)malloc(sizeof(float) * LEN_ARR);
    float *b = (float*)malloc(sizeof(float) * LEN_ARR);
    float *result = (float*)malloc(sizeof(float) * LEN_ARR);
    float *gpuA, *gpuB, *gpuResult;

    for(int i = 0; i < LEN_ARR; i++){
        a[i] = 30.0f;
        b[i] = 12.0f;
    }

    cudaMalloc((void **) &gpuA, sizeof(float) * LEN_ARR);
    cudaMalloc((void **) &gpuB, sizeof(float) * LEN_ARR);
    cudaMalloc((void **) &gpuResult, sizeof(float) * LEN_ARR);

    cudaMemcpy(gpuA, a, sizeof(float) * LEN_ARR, cudaMemcpyHostToDevice);
    cudaMemcpy(gpuB, b, sizeof(float) * LEN_ARR, cudaMemcpyHostToDevice);

    sumVector<<<N_BLOCKS, BLOCK_SIZE>>>(gpuA, gpuB, gpuResult, LEN_ARR);

    cudaMemcpy(result, gpuResult, sizeof(float) * LEN_ARR, cudaMemcpyDeviceToHost);

    cudaFree(gpuA);
    cudaFree(gpuB);
    cudaFree(gpuResult);

    free(a); 
    free(b); 
    free(result);

    return 0;
}

