all: isingVanilla isingCuda

isingVanilla: isingVanilla.c
	gcc isingVanilla.c -lm -o isingVanilla

isingCuda: isingCuda.cu
	nvcc isingCuda.cu -o isingCuda
