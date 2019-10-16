#include <iostream>
#include <cstdlib>
#include <ctime>
#include "cuda_runtime.h"

using namespace std;

__global__ void g_buble_sort(int *array);

void buble_sort(int* array, int array_len);
void swap(int* first, int* second);
void array_print(int* array, int array_len, const char* message);

int main(int argc, char** argv)
{
    int array_len, start, stop;
    
    //Obtaining command line arguments
    switch (argc)
    {
    case 1:
        array_len = 15;
        cout << " #Warning# Default array size: " << array_len << endl;
        start = 0;
        cout << " #Warning# Default random start: " << start << endl;
        stop = 100;
        cout << " #Warning# Default random stop: " << stop << endl;
        cout << endl;
        break;
    case 2:
        array_len = atoi(argv[1]);
        start = 0;
        cout << " #Warning# Default random start: " << start << endl;
        stop = 100;
        cout << " #Warning# Default random stop: " << stop << endl;
        cout << endl;
        break;
    case 4:
        array_len = atoi(argv[1]);
        start = atoi(argv[2]);
        stop = atoi(argv[3]);
        cout << endl;
        break;   
    default:
        cout << "Wrong input!" << endl;
    }

    //Prepairing variables
    int *init_array = new int[array_len], *gpu_array = new int[array_len];
    int *d_array;
    int array_size = sizeof(int) * array_len;

    cudaMalloc((void**)&d_array, array_size);

    //Randomizing array
    srand(time(NULL));
    for (int i = 0; i < array_len; i++)
    {
        init_array[i] = start + rand() % stop;
    }
    
    //Copy array
    memcpy(gpu_array, init_array, array_size);
    cudaMemcpy(d_array, gpu_array, array_size, cudaMemcpyHostToDevice);

    array_print(init_array, array_len, "Initial array");
    
    ////////////////
    //GPU sorting
    g_buble_sort<<<1, 1>>>(d_array);
    cudaDeviceSynchronize();

    cudaError_t cuda_status = cudaGetLastError();
    if(cuda_status != cudaSuccess)
    {
        cout << "Kernel error!" << endl;
        goto cuda_error;
    }

    cudaMemcpy(gpu_array, d_array, array_size, cudaMemcpyDeviceToHost);
    
    array_print(gpu_array, array_len, "Array after GPU sort");
    ////////////////

    //CPU sorting
    buble_sort(init_array, array_len);
    array_print(init_array, array_len, "Array after CPU sort");

cuda_error:
    delete[] init_array, gpu_array;

    cudaFree(d_array);

    cudaDeviceReset();

    return 0;
}

__global__ void g_buble_sort(int *array)
{
    printf("Hello from device!\n");
}

void buble_sort(int* array, int array_len)
{
    for(int i = 1; i < array_len - 1; i++)
    {
        for(int j = 0; j < array_len - i; j++)
        {
            if(array[j] > array[j + 1])
            {  
                swap(&array[j], &array[j+1]);
            }
        }
    }
}

void swap(int* first, int* second)
{
    int temp = *first;
    *first = *second;
    *second = temp;
}

void array_print(int* array, int array_len, const char* message)
{
    cout << " " << message << ":\n [ ";
    for (int i = 0; i < array_len; i++)
    {
        cout << array[i] << " ";
    }
    cout << "]" << endl << endl;
}
