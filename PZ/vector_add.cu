#include <stdio.h>
#include <ctime>
#include "cuda_runtime.h"

#define array_len 200

__global__ void add(int *a, int *b, int *c)
{
	c[blockIdx.x] = a[blockIdx.x] + b[blockIdx.x];
}

int main()
{
	int *a, *b, *c;
	int *d_a, *d_b, *d_c;
	
	int size = sizeof(int) * array_len;
	
	a = (int*)malloc(size);
	b = new int[array_len];
	c = new int[array_len];
	
	cudaMalloc((void **)&d_a, size);
	cudaMalloc((void **)&d_b, size);
	cudaMalloc((void **)&d_c, size);
	
	std::srand(std::time(0));
	for(int i = 0; i < array_len; i++)
	{
		a[i] = 1 + std::rand() % 100;
		b[i] = 1 + std::rand() % 100;
	}
	
	cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);
	
	add<<<array_len,1>>>(d_a, d_b, d_c);
	cudaDeviceSynchronize();

	cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);
	
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);

	printf("Calculated\n");
	for(int i = 0; i < array_len; i++)
		printf("%-3d + %-3d = %-3d\n", a[i], b[i], c[i]);
		
	free(a);
	free(b);
	free(c);

	return 0;
}