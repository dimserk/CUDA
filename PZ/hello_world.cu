#include <stdio.h>

__global__ void hello()
{
	printf("Hello world from device!\n");
}

int main()
{
	printf("Hello world from host!\n");
	
	hello<<<10,1>>>();
	cudaDeviceSynchronize();
	
	return 0;
}
	