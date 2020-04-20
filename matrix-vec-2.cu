#include <algorithm>
#include <stdio.h>
#include <omp.h>
#include <string>


__global__
void mat_mult_kernel(double* result, const double* matrix, const double* vect, long N){
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  double temp = 0.0; 
  if(idx<N*N){
    temp += matrix[idx] * vect[idx%N];
  }
  atomicAdd(&result[idx%N], temp); 
}



void Check_CUDA_Error(const char *message){
  cudaError_t error = cudaGetLastError();
  if(error!=cudaSuccess) {
    fprintf(stderr,"ERROR: %s: %s\n", message, cudaGetErrorString(error) );
    exit(-1);
  }
}



int main() {
  
  long N =  1024;

  printf("N: %ld\n", N);
  
  double  *y; 
  
  cudaMallocHost((void**)&y, N * sizeof(double));
  

  #pragma omp parallel for schedule(static)
  for (long i = 0; i < N; i++) {
    y[i] = 1.0/(i+1);
  }
  double  *y_d;
  cudaMalloc(&y_d, N*sizeof(double));

  

  printf("Matrix multiplication:\n"); 
  double *matrix_CPU, *matrix_GPU; 
  cudaMallocHost((void**)&matrix_CPU, N * N * sizeof(double));
  cudaMalloc(&matrix_GPU, N*N*sizeof(double)); 
  for(long i = 0; i<N; i++)
    for(long j = 0; j < N ; j++)
        *(matrix_CPU+i*N+j) = (double)j + 1.0; 


  double *matrix_mult_ref, *matrix_mult; 
  cudaMallocHost((void**)&matrix_mult_ref, N * sizeof(double));
  cudaMalloc(&matrix_mult, N*sizeof(double)); 
  for(long i = 0; i<N; i++ )
    matrix_mult_ref[i] = 0.0; 
  

    
  cudaMemcpy(matrix_mult, matrix_mult_ref, N*sizeof(double), cudaMemcpyHostToDevice);


  double tt = omp_get_wtime();
  
  
  for(long i = 0; i<N; i++){
    double product = 0; 
    for(long j = 0; j < N ; j++)
        product += (y[j]* *(matrix_CPU+i*N+j)); 
    matrix_mult_ref[i] = product; 

  }
  double time = (omp_get_wtime()-tt);
  printf("\nCPU Bandwidth = %f GB/s\n", (3*N*N+N)*sizeof(double) / time/1e9);
  printf("Time taken on CPU = %f s\n", time);
   

  
  
  double *product_GPU;
  cudaMallocHost((void**)&product_GPU, N * sizeof(double)); 

  tt = omp_get_wtime();
  cudaMemcpy(y_d, y, N*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(matrix_GPU, matrix_CPU, N*N*sizeof(double), cudaMemcpyHostToDevice);
  mat_mult_kernel<<<N*N/1024+1,1024>>>(matrix_mult, matrix_GPU, y_d, N); 
  cudaMemcpy(product_GPU, matrix_mult, N*sizeof(double), cudaMemcpyDeviceToHost);
  time = (omp_get_wtime()-tt);
  printf("\nGPU Bandwidth = %f GB/s\n", (3*N*N+N)*sizeof(double) / time/1e9);
  printf("Time taken on GPU = %f s\n", time);
   

  double error = 0.0; 
  for(long i = 0; i<N; i++)
    error += matrix_mult_ref[i] - product_GPU[i]; 

  printf("\nError:%lf", error); 

  
  
  cudaFree(matrix_GPU); 
  cudaFree(matrix_mult); 
  cudaFree(y_d);

  cudaFreeHost(matrix_CPU);
  cudaFreeHost(product_GPU); 
  cudaFreeHost(matrix_mult_ref); 
  cudaFreeHost(y);
  
  cudaDeviceSynchronize();
  return 0;
}





