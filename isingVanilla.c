#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define MAX_TEMP 4
#define NUM_STEP 100
#define MIN_TEMP 0.01
#define N 50
#define J -1
#define H 0
#define Kb 1
#define TRIGGER 300000
#define SAMPLE_DELAY 500
#define N_SAMPLES 800

char S[N][N];

void randomizeS() {
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            S[i][j] = rand() > RAND_MAX / 2 ? 1 : -1;
}

float *runCycles(float T) {
    float *magnArr = malloc(N_SAMPLES * sizeof(float));
    float *energyArr = malloc(N_SAMPLES * sizeof(float));
    float magn, energy, deltaE;
    int x, y, insertedSamples = 0;
    for (int i = 0; i < TRIGGER + N_SAMPLES * SAMPLE_DELAY; i++) {
        magn = 0;
        energy = 0;
        x = rand() % (N - 1);
        y = rand() % (N - 1);
        deltaE = -2 * J * S[x][y] * (S[(x + 1) % N][(y + 1) % N] + S[(x + 1) % N][(y - 1) % N] + S[(x - 1) % N][(y + 1) % N] + S[(x - 1) % N][(y - 1) % N]) - 2 * H * S[x][y];
        if (deltaE < 0 || exp((float)-deltaE / (Kb * T)) > (float)rand() / RAND_MAX)
            S[x][y] *= -1;
        if (i % SAMPLE_DELAY == 0 && i > TRIGGER) {
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

    float *retArr = malloc(4 * sizeof(float));
    retArr[0] = T;
    retArr[1] = susc;
    retArr[2] = cal;
    retArr[3] = magn;
    return retArr;
}

int main() {
    srand(time(NULL));
    randomizeS();
    float *resArr;
    float susc[NUM_STEP], temp[NUM_STEP], heat[NUM_STEP], arrMagn[NUM_STEP];
    for (int i = 0; i < NUM_STEP; i++) {
        float t = MIN_TEMP + (MAX_TEMP - MIN_TEMP) / NUM_STEP * i;
        resArr = runCycles(t);
        temp[i] = resArr[0];
        susc[i] = resArr[1];
        heat[i] = resArr[2];
        arrMagn[i] = resArr[3];
        free(resArr);
    }

    return 0;
}
