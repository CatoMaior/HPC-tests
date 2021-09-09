all: isingVanilla isingCuda sumVectorsVanilla sumVectorsCuda

isingVanilla: isingVanilla.c
	gcc isingVanilla.c -lm -o isingVanilla

isingCuda: isingCuda.cu
	nvcc isingCuda.cu -o isingCuda

sumVectorsVanilla: sumVectorsVanilla.c
	gcc sumVectorsVanilla.c -o sumVectorsVanilla

sumVectorsCuda: sumVectorsCuda.cu
	nvcc sumVectorsCuda.cu -o sumVectorsCuda