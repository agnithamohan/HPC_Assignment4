#include <algorithm>
#include <stdio.h>
#include <omp.h>
#include <string>

#define BLOCK_SIZE 1024

//Inner product of 2 vectors - CPU 
void vec_mult(double *prod_ptr, double* c, const double* a, const double* b, long N){
  double product=0.0; 
  for (long i = 0; i < N; i++) {
    c[i] = a[i] * b[i];
    product+=c[i]; 
  }
  *prod_ptr = product; 
}

//Inner product of 2 vectors - GPU kernels
__global__
void vec_mult_kernel(double* c, const double* a, const double* b, long N, long offset){
  int idx = offset + blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) c[idx] = a[idx] * b[idx];
}
__global__ void reduction_kernel(double* sum, const double* a, long N){
  __shared__ double smem[BLOCK_SIZE];
  int idx = (blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < N) smem[threadIdx.x] = a[idx];
  else smem[threadIdx.x] = 0;
  __syncthreads();

  for (unsigned int s=blockDim.x/2; s>0; s>>=1) {
	if (threadIdx.x < s) {
		smem[threadIdx.x] += smem[threadIdx.x + s];
	}
	__syncthreads();
   }

  if (threadIdx.x == 0) sum[blockIdx.x] = smem[threadIdx.x];
}


void Check_CUDA_Error(const char *message){
  cudaError_t error = cudaGetLastError();
  if(error!=cudaSuccess) {
    fprintf(stderr,"ERROR: %s: %s\n", message, cudaGetErrorString(error) );
    exit(-1);
  }
}



int main() {
  const int blockSize = 256, nStreams = 4;
  long N =   100* blockSize * nStreams;
  const int streamSize = N / nStreams;
  const int streamBytes = streamSize * sizeof(double);

  printf("N: %ld\n", N);

  double *x, *y, *z;
  cudaMallocHost((void**)&x, N * sizeof(double));
  cudaMallocHost((void**)&y, N * sizeof(double));
  cudaMallocHost((void**)&z, N * sizeof(double));
  double* z_ref = (double*) malloc(N * sizeof(double));

  #pragma omp parallel for schedule(static)
  for (long i = 0; i < N; i++) {
    x[i] = i+1;
    y[i] = 1.0/(i+1);
    z[i] = 0;
    z_ref[i] = 0;
  }


  printf("\nVector Inner Product:"); 
  double sum_ref,sum; 
  double tt = omp_get_wtime();
  vec_mult(&sum_ref,z_ref, x, y, N);
  double time = (omp_get_wtime()-tt);
  printf("\nCPU Bandwidth = %f GB/s\n", 3*N*sizeof(double) / time/1e9);
  printf("Time taken on CPU = %f s\n", time);


  double *x_d, *y_d, *z_d;
  cudaMalloc(&x_d, N*sizeof(double));
  cudaMalloc(&y_d, N*sizeof(double));
  cudaMalloc(&z_d, N*sizeof(double));

  cudaStream_t stream[nStreams];
  for (int i = 0; i < nStreams; ++i)
    cudaStreamCreate(&stream[i]);

  tt = omp_get_wtime();
  //Using streams to compute the product of corresponding terms
  for (int i = 0; i < nStreams; ++i) {
    int offset = i * streamSize;
    cudaMemcpyAsync(&x_d[offset], &x[offset],
                               streamBytes, cudaMemcpyHostToDevice,
                               stream[i]);
    cudaMemcpyAsync(&y_d[offset], &y[offset],
                               streamBytes, cudaMemcpyHostToDevice,
                               stream[i]);
    vec_mult_kernel<<<streamSize/blockSize, blockSize, 0, stream[i]>>>(z_d, x_d, y_d, N, offset);
    cudaMemcpyAsync(&z[offset], &z_d[offset],
                               streamBytes, cudaMemcpyDeviceToHost,
                               stream[i]);
  }
  cudaDeviceSynchronize();

  //Using reduction kernel to compute the final sum
  double *partialsum_d; 
  long N_work = 1;
  for (long i = (N+BLOCK_SIZE-1)/(BLOCK_SIZE); i > 1; i = (i+BLOCK_SIZE-1)/(BLOCK_SIZE)) N_work += i;
  cudaMalloc(&partialsum_d, N_work*sizeof(double)); 


  double * sum_d = partialsum_d;
  long Nb = (N+BLOCK_SIZE-1)/(BLOCK_SIZE);
  reduction_kernel<<<Nb,BLOCK_SIZE>>>(sum_d, z_d, N);
  while (Nb > 1) {
    long N = Nb;
    Nb = (Nb+BLOCK_SIZE-1)/(BLOCK_SIZE);
    reduction_kernel<<<Nb,BLOCK_SIZE>>>(sum_d + N, sum_d, N);
    sum_d += N;
  }


  cudaMemcpyAsync(&sum, sum_d, 1*sizeof(double), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

  time = (omp_get_wtime()-tt);

  printf("\n\nGPU Bandwidth = %f GB/s\n", 4*N*sizeof(double) / time/1e9);
  printf("Time taken on GPU = %f s\n", time);
  printf("Error(vector inner product) = %f\n", fabs(sum-sum_ref));


   /*------Matrix(R*N) Vector Product------*/
  printf("\n\nMatrix-vector multiplication:"); 
  long R = N; 
  double *matrix[R]; 
  for (long i=0; i<R; i++) 
    matrix[i] = (double *)malloc(N * sizeof(double)); 

  for(long i = 0; i < R; i++)
    for(long j=0; j < N; j++){
      matrix[i][j] = (j%N) + 1.0; 
    }

    
  double* matrix_ref = (double*) malloc(R * sizeof(double));
  double* matrix_prod = (double*) malloc(R * sizeof(double));
  tt = omp_get_wtime();
  for(long i  = 0 ; i < R ; i++)
    vec_mult(&matrix_ref[i],z_ref, x, y, N);
 
  time = (omp_get_wtime()-tt);
  printf("\n\nCPU Bandwidth = %f GB/s\n", 3*R*N*sizeof(double) / time/1e9);
  printf("Time taken on CPU = %f s\n", time);

  
  tt = omp_get_wtime();  
  for(long i = 0; i < R; i++){
    for (long j = 0; j < nStreams; ++j) {
      int offset = j * streamSize;
      cudaMemcpyAsync(&x_d[offset], &matrix[i][offset],
                                 streamBytes, cudaMemcpyHostToDevice,
                                 stream[j]);
      cudaMemcpyAsync(&y_d[offset], &y[offset],
                                 streamBytes, cudaMemcpyHostToDevice,
                                 stream[j]);
      vec_mult_kernel<<<streamSize/blockSize, blockSize, 0, stream[j]>>>(z_d, x_d, y_d, N, offset);
      cudaMemcpyAsync(&z[offset], &z_d[offset],
                                 streamBytes, cudaMemcpyDeviceToHost,
                                 stream[j]);
    }
    cudaDeviceSynchronize();
  
    double *partialsum_d; 
    long N_work = 1;
    for (long k = (N+BLOCK_SIZE-1)/(BLOCK_SIZE); k > 1; k = (k+BLOCK_SIZE-1)/(BLOCK_SIZE)) N_work += k;
    cudaMalloc(&partialsum_d, N_work*sizeof(double)); 

    double * sum_d = partialsum_d;
    long Nb = (N+BLOCK_SIZE-1)/(BLOCK_SIZE);
    reduction_kernel<<<Nb,BLOCK_SIZE>>>(sum_d, z_d, N);
    while (Nb > 1) {
      long N = Nb;
      Nb = (Nb+BLOCK_SIZE-1)/(BLOCK_SIZE);
      reduction_kernel<<<Nb,BLOCK_SIZE>>>(sum_d + N, sum_d, N);
      sum_d += N;
    }
  
  
    cudaMemcpyAsync(&matrix_prod[i], sum_d, 1*sizeof(double), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    
  }
  time = (omp_get_wtime()-tt);
  printf("\n\nGPU Bandwidth = %f GB/s\n", 4*N*R*sizeof(double) / time/1e9);
  printf("Time taken on GPU = %f s\n", time);
  double error = 0.0; 
  for(long i = 0 ; i  < R ; i++)
    error += fabs(matrix_prod[i]-matrix_ref[i]);  
    
  printf("Error(matrix vector product) = %f\n", error);
  cudaFree(x_d);
  cudaFree(y_d);
  cudaFree(z_d);

  cudaFreeHost(x);
  cudaFreeHost(y);
  cudaFreeHost(z);
  free(z_ref);
  cudaDeviceSynchronize();
  return 0;
}



