#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <time.h>

#define N 50
#define J -1
#define H 0
#define Kb 1
#define TRIGGER 300000
#define SAMPLE_DELAY 500
#define N_SAMPLES 300
#define START_TEMP 400
#define END_TEMP 1
#define TEMP_STEP -4

char S[N][N];

void randomizeS() {
    for (int i = 0; i < N; i++) 
        for (int j = 0; j < N; j++) 
            S[i][j] = rand() > RAND_MAX / 2 ? 1 : -1;
}

float* runCycles(float T) {
    float magnArr[N_SAMPLES], energyArr[N_SAMPLES];
    int magn, energy, x, y, deltaE, insertedSamples = 0;
    for (int i = 0; i < TRIGGER + N_SAMPLES + SAMPLE_DELAY; i++) {
        magn = 0;
        energy = 0;
        x = rand() % (N - 1);
        y = rand() % (N - 1);
        deltaE = -2 * J * S[x][y] * (S[(x + 1) % N][(y + 1) % N] +
                    S[(x + 1) % N][(y - 1) % N] +
                    S[(x - 1) % N][(y + 1) % N] +
                    S[(x - 1) % N][(y - 1) % N]) - 2 * H * S[x][y];
        if (deltaE < 0 || exp((float) -deltaE / (Kb * T)) > (float) random() / RAND_MAX)
            S[x][y] *= 1;
        if (i % SAMPLE_DELAY == 0 && i > TRIGGER) {
            for (int j = 0; j < N; j++) {
                for (int k = 0; k < N; k++) 
                    magn += S[j][k];
            }
            magnArr[insertedSamples] = (float) magn / (N * N);
            for (int j = 0; j < N; j++) {
                for (int k = 0; k < N; k++)
                    energy += J * S[j][k] * (S[(j + 1) % N][k] +
                        S[j][(k + 1) % N]) - 2 * H * S[j][k];
            }
            energyArr[insertedSamples] = energy;
            insertedSamples++;
        }
    }

}

int main() {
    srand(time(NULL));
    randomizeS();
    return 0;
}