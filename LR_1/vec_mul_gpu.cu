#include <iostream>
#include <cstdlib>
#include <ctime>
#include "cuda_runtime.h"

#define VEC_SIZE 20000
#define START 1
#define STOP 100

using namespace std;

__global__ void vect_mul(int *arr_a, int *arr_b, int *arr_c)
{
    arr_c[threadIdx.x] = arr_a[threadIdx.x] * arr_b[threadIdx.x];
}

int main()
{
    int *arr_a, *arr_b, *arr_c, total_sum = 0, dev_count;
    int *d_arr_a, *d_arr_b, *d_arr_c;
    int size = sizeof(int) * VEC_SIZE;

    float working_time = 0;

    cudaEvent_t e_start, e_stop;
    cudaError_t cuda_status;

    cuda_status = cudaEventCreate(&e_start);
    if(cuda_status != cudaSuccess)
    {
        cout << "Can not create cuda event!" << endl;
    }

    cuda_status = cudaEventCreate(&e_stop);
    if(cuda_status != cudaSuccess)
    {
        cout << "Can not create cuda event!" << endl;
    }

    arr_a = new int[VEC_SIZE];
    arr_b = new int[VEC_SIZE];
    arr_c = new int[VEC_SIZE];

    cuda_status = cudaMalloc((void**)&d_arr_a, size);
    if(cuda_status != cudaSuccess)
    {
        cout << "Cuda malloc error!" << endl;
        goto cuda_error;
    }

    cuda_status = cudaMalloc((void**)&d_arr_b, size);
    if(cuda_status != cudaSuccess)
    {
        cout << "Cuda malloc error!" << endl;
        goto cuda_error;
    }

    cuda_status = cudaMalloc((void**)&d_arr_c, size);
    if(cuda_status != cudaSuccess)
    {
        cout << "Cuda malloc error!" << endl;
        goto cuda_error;
    }

    srand(time(NULL));
    for (int i = 0; i < VEC_SIZE; i++)
    {
        arr_a[i] = START + rand() % STOP;
        arr_b[i] = START + rand() % STOP;
    }

    cuda_status = cudaMemcpy(d_arr_a, arr_a, VEC_SIZE, cudaMemcpyHostToDevice);
    if(cuda_status != cudaSuccess)
    {
        cout << "Cuda memcpy error!" << endl;
        goto cuda_error;
    }

    cuda_status = cudaMemcpy(d_arr_b, arr_b, VEC_SIZE, cudaMemcpyHostToDevice);
    if(cuda_status != cudaSuccess)
    {
        cout << "Cuda memcpy error!" << endl;
        goto cuda_error;
    }
    
    cuda_status = cudaGetDeviceCount(&dev_count);
    if(cuda_status != cudaSuccess)
    {
        cout << "Cuda get device count error!" << endl;
        goto cuda_error;
    }

    cuda_status = cudaEventRecord(e_start);
    if(cuda_status != cudaSuccess)
    {
        cout << "Cuda event error while recording!" << endl;
        goto cuda_error;
    }

    vect_mul<<<VEC_SIZE, 1>>>(d_arr_a, d_arr_b, d_arr_c);
    cudaDeviceSynchronize();
    
    cuda_status = cudaGetLastError();
    if(cuda_status != cudaSuccess)
    {
        cout << "Kernel error!" << endl;
        goto cuda_error;
    }

    cuda_status = cudaEventRecord(e_stop);
    if(cuda_status != cudaSuccess)
    {
        cout << "Cuda event error while recording!" << endl;
        goto cuda_error;
    }

    cuda_status = cudaMemcpy(arr_c, d_arr_c, VEC_SIZE, cudaMemcpyDeviceToHost);
    if(cuda_status != cudaSuccess)
    {
        cout << "Cuda memcpy error!" << endl;
        goto cuda_error;
    }

    for(int i = 0; i < VEC_SIZE; i++)
        total_sum += arr_c[i];

    cuda_status = cudaEventSynchronize(e_stop);
    if(cuda_status != cudaSuccess)
    {
        cout << "Cuda event error while synchronizing!" << endl;
        goto cuda_error;
    }

    cuda_status = cudaEventElapsedTime(&working_time, e_start, e_stop);
    if(cuda_status != cudaSuccess)
    {
        cout << "Cuda event error while elapsing!" << endl;
        goto cuda_error;
    }

    cout << "CUDA devices:  " << dev_count << endl;
    cout << "Result of vectors multiplication is " << total_sum << endl;
    cout << "Working time: " << working_time  << " ms"<< endl;

    cuda_error:
    delete[] arr_a;
    delete[] arr_b;
    delete[] arr_c;

    cudaFree(d_arr_a);
    cudaFree(d_arr_b);
    cudaFree(d_arr_c);
    cudaDeviceReset();

    return 0;
}
