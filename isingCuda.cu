#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <curand_kernel.h>
#include <curand.h>
#include <time.h>
#include <cuda_profiler_api.h>

#define MAX_TEMP 4
#define NUM_STEP 100
#define MIN_TEMP 0.01
#define N 50
#define J -1
#define H 0
#define Kb 1
#define TRIGGER 1000
#define SAMPLE_DELAY 10
#define N_SAMPLES 800
#define N_THREAD 40

char S[N][N];

void randomizeS() {
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            S[i][j] = rand() > RAND_MAX / 2 ? 1 : -1;
}

__device__ float generate(curandState* globalState, int ind) {
    curandState localState = globalState[ind];
    float RANDOM = curand_uniform(&localState);
    globalState[ind] = localState;
    return RANDOM;
}

__global__ void setup_kernel(curandState * state, unsigned long seed) {
    int id = threadIdx.x;
    curand_init(seed, id, 0, &state[id]);
}

__global__ void updateBoard(char* gpuS, float* T, curandState *states) {
    int id = threadIdx.x;
    int x = ((int) (generate(states, id) * 1000000)) % (N - 1);
    int y = ((int) (generate(states, id) * 1000000)) % (N - 1);
    float deltaE = -2 * J * *(gpuS + N * x + y) * ( *(gpuS + ((x + 1) % N) * N + (y + 1) % N) +
                                    *(gpuS + ((x + 1) % N) * N + (y - 1) % N) +
                                    *(gpuS + ((x - 1) % N) * N + (y + 1) % N) +
                                    *(gpuS + ((x - 1) % N) * N + (y - 1) % N)) -
                                    2 * H * *(gpuS + N*x + y);
    __syncthreads();
    if (deltaE < 0 || exp((float) - deltaE / (Kb * *T)) > generate(states, id))
        *(gpuS + N * x + y) *= -1;
}

float *runCycles(float T, curandState* devStates) {
    float *magnArr = (float *)malloc(N_SAMPLES * sizeof(float));
    float *energyArr = (float *)malloc(N_SAMPLES * sizeof(float));
    float magn, energy;
    int insertedSamples = 0;
    
    char* gpuS;
    float* Ts;
    cudaMalloc((void **) &gpuS, N * N * sizeof(char));
    cudaMalloc((void **) &Ts, sizeof(float));
    cudaMemcpy(Ts, S, sizeof(float), cudaMemcpyHostToDevice);

    for (unsigned int i = 0; i < TRIGGER + N_SAMPLES; i++) {
        magn = 0;
        energy = 0;
        cudaMemcpy(gpuS, S, N * N, cudaMemcpyHostToDevice);
        for (unsigned int j = 0; j < SAMPLE_DELAY; j++) {
            updateBoard<<<1, N_THREAD>>>(gpuS, Ts, devStates);
            cudaDeviceSynchronize();
        }
        cudaMemcpy(S, gpuS, N * N, cudaMemcpyDeviceToHost);
        if (i > TRIGGER) {
            for (int j = 0; j < N; j++) {
                for (int k = 0; k < N; k++)
                    magn += S[j][k];
            }
            magnArr[insertedSamples] = (float)magn / (N * N);
            for (int j = 0; j < N; j++) {
                for (int k = 0; k < N; k++)
                    energy += J * S[j][k] * (S[(j + 1) % N][k] + S[j][(k + 1) % N]) - 2 * H * S[j][k];
            }
            energyArr[insertedSamples] = energy;
            insertedSamples++;
        }
    }

    float susc = 0, sq_av = 0, av_sq = 0, cal = 0;
    magn = 0;

    for (int i = 0; i < N_SAMPLES; i++) {
        sq_av += magnArr[i];
        av_sq += sq_av * sq_av;
    }
    sq_av /= N_SAMPLES;
    av_sq /= N_SAMPLES;

    magn = abs(sq_av);
    sq_av = sq_av * sq_av;
    susc = (av_sq - sq_av) / T;
    sq_av = 0;
    av_sq = 0;

    for (int i = 0; i < N_SAMPLES; i++) {
        sq_av += energyArr[i];
        av_sq += sq_av * sq_av;
    }
    sq_av /= N_SAMPLES;
    av_sq /= N_SAMPLES;
    sq_av = sq_av * sq_av;
    cal = (av_sq - sq_av) / T;

    free(magnArr);
    free(energyArr);

    float *retArr = (float *)malloc(4 * sizeof(float));
    retArr[0] = T;
    retArr[1] = susc;
    retArr[2] = cal;
    retArr[3] = magn;
    return retArr;
}

int main() {
    srand(time(NULL));
    randomizeS();
    curandState* devStates;
    cudaMalloc(&devStates, N * sizeof(curandState));
    int seed = rand();
    setup_kernel<<<1, N_THREAD>>>(devStates,seed);
    cudaDeviceSynchronize();
    float *resArr;
    float susc[NUM_STEP], temp[NUM_STEP], heat[NUM_STEP], arrMagn[NUM_STEP];
    for (int i = 0; i < NUM_STEP; i++) {
        float t = MIN_TEMP + (MAX_TEMP - MIN_TEMP) / NUM_STEP * i;
        resArr = runCycles(t, devStates);
        temp[i] = resArr[0];
        susc[i] = resArr[1];
        heat[i] = resArr[2];
        arrMagn[i] = resArr[3];
        free(resArr);
    }    
    return 0;
}
