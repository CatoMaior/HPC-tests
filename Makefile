all: vanilla test cuda

vanilla: vanilla.c
	gcc vanilla.c -lm -o vanilla

cuda: cuda.cu
	nvcc cuda.cu -o cuda

test: test.cu
	nvcc test.cu -o test
