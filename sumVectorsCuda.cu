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
    float *d_a, *d_b, *d_result;

    for(int i = 0; i < LEN_ARR; i++){
        a[i] = 30.0f;
        b[i] = 12.0f;
    }

    cudaMalloc((void **) &d_a, sizeof(float)*LEN_ARR);
    cudaMalloc((void **) &d_b, sizeof(float)*LEN_ARR);
    cudaMalloc((void **) &d_result, sizeof(float)*LEN_ARR);

    cudaMemcpy(d_a, a, sizeof(float)*LEN_ARR, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, sizeof(float)*LEN_ARR, cudaMemcpyHostToDevice);

    sumVector<<<N_BLOCKS, BLOCK_SIZE>>>(d_a, d_b, d_result, LEN_ARR);

    cudaMemcpy(result, d_result, sizeof(float) * LEN_ARR, cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_result);

    free(a); 
    free(b); 
    free(result);

    return 0;
}

