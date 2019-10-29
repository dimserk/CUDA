#include <iostream>
#include <cstdlib>
#include <ctime>
#include "cuda_runtime.h"

#define PRTINTING_LIMIT 20

using namespace std;

__global__ void g_buble_sort(int *array, int* array_len, int* iter_num)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    bool work = false;
    int position = 0;

    while(true)
    {
        if(i * 2 == *iter_num)
        {
            work = true;
        }

        if (work)
        {
            if(array[position] > array[position  + 1])
            {
                int tmp = array[position];
                array[position] = array[position + 1];
                array[position + 1] = tmp;
            }

            position++;
        }

        if(i == 0)
        {
            (*iter_num)++;
        }

        if(position == *array_len - 1)
        {
            work = false;
            if(i != 0)
            {
                break;
            }
        }

        if(*iter_num == *array_len * 2 - 3)
        {
            break;
        }

        __syncthreads();
    }
}

void buble_sort(int* array, int array_len)
{
    for(int i = 1; i < array_len - 1; i++)
    {
        for(int j = 0; j < array_len - i; j++)
        {
            if(array[j] > array[j + 1])
            {  
                int tmp = array[j];
                array[j] = array[j + 1];
                array[j + 1] = tmp;
            }
        }
    }
}

void array_print(int* array, int array_len, const char* message)
{
    cout << " " << message << ":\n [ ";
    for (int i = 0; i < array_len; i++)
    {
        cout << array[i] << " ";
    }
    cout << "]" << endl;
}

int main(int argc, char** argv)
{
    int array_len = 15, start = 0, stop = 100;
    
    //Obtaining command line arguments
    switch (argc)
    {
    case 1:
        cout << " #Warning# Default array size: " << array_len << endl;
        cout << " #Warning# Default random start: " << start << endl;
        cout << " #Warning# Default random stop: " << stop << endl;
        break;
    case 2:
        array_len = atoi(argv[1]);
        cout << " #Warning# Default random start: " << start << endl;
        cout << " #Warning# Default random stop: " << stop << endl;
        break;
    case 4:
        array_len = atoi(argv[1]);
        start = atoi(argv[2]);
        stop = atoi(argv[3]);
        break;   
    default:
        cout << " #Error# Wrong input! Default settings applied." << endl;
        cout << " #Warning# Default array size: " << array_len << endl;
        cout << " #Warning# Default random start: " << start << endl;
        cout << " #Warning# Default random stop: " << stop << endl;
    }
    cout << endl;

    if(array_len < 2)
    {
        cout << " #Error# Array length is too small, at least 2!" << endl;
        return 0;
    }

    //Prepairing variables
    int *init_array = new int[array_len], *gpu_array = new int[array_len];
    int *d_array, *d_array_len, *d_iter_num;
    int array_size = sizeof(int) * array_len;
    float working_time = 0;

    cudaEvent_t e_start, e_stop;

    cudaEventCreate(&e_start);
    cudaEventCreate(&e_stop);

    cudaMalloc((void**)&d_array, array_size);
    cudaMalloc((void**)&d_array_len, sizeof(int));
    cudaMalloc((void**)&d_iter_num, sizeof(int));

    //Randomizing array
    srand(time(NULL));
    for (int i = 0; i < array_len; i++)
    {
        init_array[i] = start + rand() % stop;
    }

    //Copy array
    memcpy(gpu_array, init_array, array_size);
    cudaMemcpy(d_array, gpu_array, array_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_array_len, &array_len, sizeof(int), cudaMemcpyHostToDevice);
    int iter_num = 0;
    cudaMemcpy(d_iter_num, &iter_num, sizeof(int), cudaMemcpyHostToDevice);

    if(array_len <= PRTINTING_LIMIT)
    {
        array_print(init_array, array_len, "Initial array");
    }

    //GPU sorting
    cudaEventRecord(e_start);

    g_buble_sort<<<array_len - 1, 1>>>(d_array, d_array_len, d_iter_num);
    cudaError_t cuda_status = cudaGetLastError();
    if(cuda_status != cudaSuccess)
    {
        cout << "Kernel error!" << endl;
        goto cuda_error;
    }

    cudaDeviceSynchronize();
    
    cudaEventRecord(e_stop);
    cudaEventSynchronize(e_stop);
    cudaEventElapsedTime(&working_time, e_start, e_stop);

    cudaMemcpy(gpu_array, d_array, array_size, cudaMemcpyDeviceToHost);
    
    //GPU printing
    if(array_len <= PRTINTING_LIMIT)
    {
        array_print(gpu_array, array_len, "Array after GPU sort");
    }
    cout << " GPU sorting time: " << working_time << " ms" << endl;

    //CPU sorting
    clock_t c_start = clock();
    buble_sort(init_array, array_len);
    clock_t c_end = clock();

    //CPU printing
    if(array_len <= PRTINTING_LIMIT)
    {
        array_print(init_array, array_len, "Array after CPU sort");
    }
    cout << " CPU sorting time: " << 1000.0 * (c_end-c_start) / CLOCKS_PER_SEC << " ms" << endl;

cuda_error:
    delete[] init_array, gpu_array;

    cudaEventDestroy(e_start);
    cudaEventDestroy(e_stop);

    cudaFree(d_array);
    cudaFree(d_array_len);
    cudaFree(d_iter_num);

    return 0;
}
