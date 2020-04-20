#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream> 
#include <sys/time.h>

using namespace std; 


__device__ double norm_calc_device;  
__global__ void JacobiKernel(double *u, double *u_new, int N, double h_sq) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if( (idx>N) && (idx%N!=0) && (idx%N!=N-1) && (idx/N < N-1))
        *(u_new + idx)  = 0.25 * ((h_sq)+ *(u + idx - N) + *(u + idx - 1) + *(u + idx + N) +  *(u + idx + 1)); 
    
}
double norm (double *u , int N, double h_sq)
{
    double norm_2d = 0.0; 
    for(int i = 1 ; i < N-1 ; i++)
        for(int j = 1; j < N-1 ; j++){
            double temp = 0.0; 
            temp+=4.0 *  *(u + i*N + j); 
            temp-= *(u + (i-1)*N + j); 
            temp-= *(u + i*N + (j-1)); 
            temp-= *(u + (i+1)*N + j); 
            temp-= *(u + i*N + (j+1));
            temp/=h_sq; 
            norm_2d+= pow((temp-1.0),2);

        }   
    norm_2d = sqrt(norm_2d); 

    return norm_2d; 
}


__global__ void normKernel(double *u , int N , double  h_sq){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    double temp = 0.0; 
    if( (idx>N) && (idx%N!=0) && (idx%N!=N-1) && (idx/N < N-1)){
        temp+=4.0 *  *(u + idx); 
        temp-= *(u + idx - N); 
        temp-= *(u + idx - 1); 
        temp-= *(u + idx + N); 
        temp-= *(u + idx + 1);
        temp/=h_sq;
        temp = pow((temp-1.0),2);
        
    }
 
    atomicAdd(&norm_calc_device, temp); 

}
void jacobi(double *u, double * u_new, int N, double h, double h_sq, double norm_init)
{
    printf("Jacobi Method:\n");

    int iter = 1; 
    double norm_calc = norm_init; 
    int max_iter = 1000; 
    printf("\nInitial Norm:%f\n", norm_init); 
    while(norm_calc * 1000000 > norm_init && iter <= max_iter)
    {
        
        for(int i = 1; i < N-1 ; i++)
            for(int j = 1 ; j < N-1 ; j++)
                *(u_new + i*N + j)  = 0.25 * ((h_sq)+ *(u + (i-1)*N + j) + *(u + i*N + (j-1)) + *(u + (i+1)*N + j) +  *(u + i*N + (j+1))); 
       
        swap(u,u_new);

        norm_calc = norm( u, N,h_sq);
        iter++; 
    }  
    printf("\nFinal Norm:%f\n", norm_calc);  
}


void jacobiGPU(double *u, double * u_new, int N, double h, double h_sq, double norm_init, double *u_device, double *u_new_device)
{
    printf("Jacobi Method:\n");
 
    int iter = 1; 
    double norm_calc = norm_init; 
    int max_iter = 1000; 
    printf("\nInitial Norm:%f\n", norm_init); 
    cudaMemcpy(u_device, u, N*N*sizeof(double), cudaMemcpyHostToDevice);
    while(norm_calc * 1000000 > norm_init && iter <= max_iter)
    {
        JacobiKernel<<<N*N/1024+1,1024>>>(u_device, u_new_device, N,h_sq);
	    cudaDeviceSynchronize(); 
      
        cudaMemcpy(u_device, u_new_device, N*N*sizeof(double), cudaMemcpyDeviceToDevice);
	    norm_calc = 0.0;  
	    cudaMemcpyToSymbol(norm_calc_device, &norm_calc, sizeof(double) ); 

        normKernel<<<N*N/1024+1,1024>>>(u_device, N,h_sq);
        cudaMemcpyFromSymbol(&norm_calc, norm_calc_device, sizeof(double), 0, cudaMemcpyDeviceToHost);

        norm_calc = sqrt(norm_calc); 
        iter++; 
    } 
    cudaMemcpy(u, u_new_device, N*N*sizeof(double), cudaMemcpyDeviceToHost);   
   
    printf("\nFinal Norm:%f\n", norm_calc);  
}

int main(int argc, char **argv)
{
    int N; 
    double h, h_sq; 
    N = 1000; 
    if(argc == 2)
    N = atoi(argv[1]); 
    h = (double)1/(double)(N+1); 
    h_sq = h*h; 

   
   double *u, *u_new;
    cudaMallocHost((void**)&u, N * N * sizeof(double));
    cudaMallocHost((void**)&u_new, N * N * sizeof(double));
    
    for(int i = 0; i<N*N; i++)
        *(u+i) = 0.0; 
        
    double norm_init = norm(u,N,h_sq); 

    struct timeval start, end;
    gettimeofday(&start, NULL);  
    jacobi(u, u_new, N, h ,  h_sq,norm_init);  
    gettimeofday(&end, NULL);
    double time_taken = ((end.tv_sec  - start.tv_sec) * 1000000u + end.tv_usec - start.tv_usec) / 1.e6;
    printf("\nThe program took %f seconds to execute\n", time_taken);
    cudaFreeHost(u_new); 

    double *u_GPU, *u_new_GPU;  
    cudaMallocHost((void**)&u_GPU, N * N * sizeof(double));
    cudaMallocHost((void**)&u_new_GPU, N * N * sizeof(double));
    
    for(int i = 0; i<N*N; i++)
        *(u_GPU+i) = 0.0; 

    double *u_device; 
    double *u_new_device; 

    cudaMalloc(&u_device, N*N*sizeof(double)); 
    cudaMalloc(&u_new_device, N*N*sizeof(double)); 
    gettimeofday(&start, NULL);  
    jacobiGPU(u_GPU, u_new_GPU, N, h ,h_sq,norm_init,u_device,u_new_device);  
    gettimeofday(&end, NULL);

    time_taken = ((end.tv_sec  - start.tv_sec) * 1000000u + end.tv_usec - start.tv_usec) / 1.e6;
   
    printf("\nThe program took %f seconds to execute\n", time_taken);
    cudaFree(u_device);
    cudaFree(u_new_device);

    /*-----Calculating errors----*/ 
    double error = 0.0; 

    for(int i=0; i<N;i++)
        for(int j=0; j<N; j++)
    {
        error+= *(u + i*N + j) - *(u_GPU + i*N + j); 
    }

    printf("Calculated error between GPU and CPU code: %f", error); 
    cudaFreeHost(u);  
    cudaFreeHost(u_GPU); 
    cudaFreeHost(u_new_GPU); 

    return 0; 
}



	
	
	
	






