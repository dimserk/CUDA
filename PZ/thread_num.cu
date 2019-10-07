#include <stdio.h>

__global__ void kernel()
{
	int num = threadIdx.x + blockIdx.x * blockDim.x;
	
	printf("Thread index: %d\n",num);
}

int main()
{
	kernel<<<4, 2>>>();
	cudaDeviceSynchronize();
	
	return 0;
}
