default: matrix-vec 2D-jacobi matrix-vec-2


matrix-vec: matrix-vec.cu
	nvcc  matrix-vec.cu  -o matrix-vec -Xcompiler -fopenmp;

2D-jacobi: 2D-jacobi.cu
	nvcc -arch=sm_61  2D-jacobi.cu -o 2D-jac

matrix-vec-2: matrix-vec-2.cu
	nvcc  matrix-vec-2.cu  -o matrix-vec-2 -Xcompiler -fopenmp -arch=sm_61;




