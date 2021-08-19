all: vanilla test cuda

vanilla:
	gcc vanilla.c -lm -o vanilla

cuda:
	nvcc cuda.cu -o cuda

test:
	nvcc test.cu -o test
