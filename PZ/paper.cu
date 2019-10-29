#include <iostream>
#include <cstdlib>
#include <ctime>
#include "cuda_runtime.h"

using namespace std;

__global__ void action(int *array1, int* array2, int* array_res) 
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    switch(i % 3)
    {
    case 0:
        array_res[i] = array1[i] + 1;
        break;
    case 1:
        array_res[i] = array2[i] - 1;
        break;
    case 2:
        array_res[i] = array1[i] * array2[i];
        break;
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
    int *array1 = new int[array_len];
    int *array2 = new int[array_len];
    int *array_res =  new int[array_len];
    int * d_array1, *d_array2, *d_array_res;

    cudaMalloc((void**)&d_array1, sizeof(int)*array_len);
    cudaMalloc((void**)&d_array2, sizeof(int)*array_len);
    cudaMalloc((void**)&d_array_res, sizeof(int)*array_len);

    //Randomizing array
    srand(time(NULL));
    for (int i = 0; i < array_len; i++)
    {
        array1[i] = start + rand() % stop;
        array2[i] = start + rand() % stop;
    }

    array_print(array1, array_len, "Array1");
    array_print(array2, array_len, "Array2");

    //Some copies 
    cudaMemcpy(d_array1, array1, sizeof(int) * array_len, cudaMemcpyHostToDevice);
    cudaMemcpy(d_array2, array2, sizeof(int) * array_len, cudaMemcpyHostToDevice);

    action<<<array_len, 1>>>(d_array1, d_array2, d_array_res);
    cudaDeviceSynchronize();

    //Some copies 
    cudaMemcpy(array_res, d_array_res, sizeof(int) * array_len, cudaMemcpyDeviceToHost);
    
    array_print(array_res, array_len, "Res array");
    
    delete[] array1, array2, array_res;

    cudaFree(d_array1);
    cudaFree(d_array2);
    cudaFree(d_array_res);

    cudaDeviceReset();

    return 0;
}