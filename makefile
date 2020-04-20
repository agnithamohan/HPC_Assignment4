default: matrix-vec 2D-jacobi


matrix-vec: matrix-vec.cu
	nvcc  matrix-vec.cu  -o matrix-vec -Xcompiler -fopenmp;

2D-jacobi: 2D-jacobi.cu
	nvcc -arch=sm_61  2D-jacobi.cu -o 2D-jac



