#include <iostream>

using namespace std;

__global__ void kernel(int *num)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    printf("ID:%d - bIdx:%d, bDim:%d, tIdx:%d\n", id, blockIdx.x, blockDim.x, threadIdx.x);
    (*num)++;
}

int main(int argc, char** argv)
{
    int num = 0;
    int *d_num;

    cudaMalloc((void**)&d_num, sizeof(int));
    cudaMemcpy(d_num, &num, sizeof(int), cudaMemcpyHostToDevice);

    kernel<<<2, 3>>>();
    cudaDeviceSynchronize();

    cudaMemcpy(num, d_num, sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_num);

    return 0;
}
