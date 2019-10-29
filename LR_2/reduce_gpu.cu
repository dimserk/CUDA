#include <iostream>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include "cuda_runtime.h"

using namespace std;

__global__ void kernel(int *array, int *i)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int second =  (1 << *i + 1)  + (1 << *i + 1) * j - 1;
    int first = second - (1 << *i);

    array[second] = array[first] + array[second];
}

int main(int argc, char** argv)
{
    int real_array_size = 512, start = 0, stop = 100;
    
    //Obtaining command line arguments
    switch (argc)
    {
    case 1:
        cout << " #Warning# Default array size: " << real_array_size << endl;
        cout << " #Warning# Default random start: " << start << endl;
        cout << " #Warning# Default random stop: " << stop << endl;
        break;
    case 2:
        real_array_size = atoi(argv[1]);
        cout << " #Warning# Default random start: " << start << endl;
        cout << " #Warning# Default random stop: " << stop << endl;
        break;
    case 4:
        real_array_size = atoi(argv[1]);
        start = atoi(argv[2]);
        stop = atoi(argv[3]);
        break;   
    default:
        cout << " #Error# Wrong input! Default settings applied." << endl;
        cout << " #Warning# Default array size: " << real_array_size << endl;
        cout << " #Warning# Default random start: " << start << endl;
        cout << " #Warning# Default random stop: " << stop << endl;
    }
    cout << endl;

    if(real_array_size < 2)
    {
        cout << " #Error# Array size is too small, at least 2!" << endl;
        return 0 ;
    }

    //Every array size can be used!
    int tmp_size, array_size, degree = 1;
    while(true)
    {
        tmp_size = pow(2, degree);
        if(real_array_size <= tmp_size)
        {
            array_size = tmp_size;
            break;
        } 

        degree++;
    }
    int *array = new int[array_size];

    //Randomazing array
    srand(time(NULL));
    for(int i = 0; i < array_size; i++)
    {
        if(i <= real_array_size - 1)
        {
            array[i] = start + rand() % stop;
        }
        else
        {
            array[i] = 0;
        }
    }

    //Control summation
    int cpu_sum = 0;
    for(int i = 0; i < real_array_size; i++)
    {
        cpu_sum += array[i];
    }

    //Device varaibles
    int *d_array, *d_i;
    int size = sizeof(int) * array_size;
    float working_time = 0;

    cudaEvent_t e_start, e_stop;

    cudaEventCreate(&e_start);
    cudaEventCreate(&e_stop);

    cudaMalloc((void**)&d_array, size);
    cudaMalloc((void**)&d_i, sizeof(int));

    cudaMemcpy(d_array, array, size, cudaMemcpyHostToDevice);

    cudaEventRecord(e_start);

    int iteration_num = array_size;
    for (int i = 0; i < log10(array_size)/log10(2); i++)
    {
        iteration_num /= 2;

        cudaMemcpy(d_i, &i, sizeof(int), cudaMemcpyHostToDevice);

        kernel<<<iteration_num, 1>>>(d_array, d_i);
        cudaDeviceSynchronize();

        cudaError_t cuda_status = cudaGetLastError();
        if(cuda_status != cudaSuccess)
        {
            cout << " #Error# Kernel error!" << endl;
            goto cuda_error;
        }
    }

    cudaEventRecord(e_stop);

    cudaMemcpy(array, d_array, size, cudaMemcpyDeviceToHost);

    cudaEventSynchronize(e_stop);
    cudaEventElapsedTime(&working_time, e_start, e_stop);

    //Printing result
    cout << " GPU summation time: " << working_time << " ms" << endl;
    cout << " Total sum of the array: " << array[array_size - 1] << " (GPU)" << endl;
    cout << " Total sum of the array: " << cpu_sum << " (CPU)" << endl;

cuda_error:
    delete[] array;

    cudaFree(d_array);
    cudaFree(d_i);

    cudaEventDestroy(e_start);
    cudaEventDestroy(e_stop);

    return 0;
}
