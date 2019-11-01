#include <iostream>
#include <cstdlib>
#include <list>
#include <ctime>
#include "cuda_runtime.h"

using namespace std;

__global__ void gpu_radix_sort(int* array, int *tmp_array, int *b_array, int* s_array, int *array_len)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    for (int k = 0; k < 32; k++) {
        b_array[id] = (array[id] >> k) & 1;

        __syncthreads();
        if(id == 0) {
            s_array[0] = 0;
            for (int i = 1; i < *array_len + 1; i++) {
                s_array[i] = s_array[i - 1] + b_array[i - 1];
            }
        }
        __syncthreads();

        if (b_array[id] == 0) {
            tmp_array[id - s_array[id]] = array[id];
        }
        else {
            tmp_array[s_array[id] + (*array_len - s_array[*array_len])] = array[id];
        }

        __syncthreads();
        array[id] = tmp_array[id];
        __syncthreads();

    }
}

void cpu_radix_sort(int* array, int array_len, int discharge) {
    auto *tmp_lists = new list<int>[10];
    int factor = 10;

    for (int d = 0; d < discharge; d++) {
        for(int i = 0; i < array_len; i++) {
            int j = array[i] % factor / (factor / 10);
            tmp_lists[j].push_back(array[i]);
        }

        int init_ind = 0;
        for(int i = 0; i < 10; i++) {
            if(!tmp_lists[i].empty()) {
                int size = tmp_lists[i].size();
                for(int j = 0; j < size; j++) {
                    array[init_ind] = tmp_lists[i].front();
                    init_ind++;
                    tmp_lists[i].pop_front();
                } 
            }
        }

        factor *= 10;
    }
}

void array_print(int* array, int array_len, const char* message) {
    cout << " " << message << ":\n [ ";
    for (int i = 0; i < array_len; i++) {
        cout << array[i] << " ";
    }
    cout << "]" << endl;
}

int main(int argc, char** argv) {
    int const PRINTING_LIMIT = 26;
    int array_len = 15, start = 0, stop = 101;
    
    //Obtaining command line arguments
    switch (argc) {
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

    if(array_len < 2) {
        cout << " #Error# Array length is too small, at least 2!" << endl;
        return 0;
    }
    int *init_array = new int[array_len];
    int *gpu_array = new int[array_len];

    int max_number = 0;
    int max_discharge = 0, discharge_factor = 1;

    //Randomizing array
    srand(time(NULL));
    for (int i = 0; i < array_len; i++) {
        init_array[i] = start + rand() % stop;
    }

    if(array_len < PRINTING_LIMIT) {
        array_print(init_array, array_len, "Initial array");
    }

    //GPU radix sort
    int *d_array, *d_tmp_array, *d_b_array, *d_s_array, *d_array_len;

    cudaMalloc((void**)&d_array, sizeof(int) * array_len);
    cudaMalloc((void**)&d_tmp_array, sizeof(int) * array_len);
    cudaMalloc((void**)&d_b_array, sizeof(int) * array_len);
    cudaMalloc((void**)&d_s_array, sizeof(int) * (array_len + 1));
    cudaMalloc((void**)&d_array_len, sizeof(int));

    cudaMemcpy(d_array, init_array, sizeof(int) * array_len, cudaMemcpyHostToDevice);
    cudaMemcpy(d_array_len, &array_len, sizeof(int), cudaMemcpyHostToDevice);

    gpu_radix_sort<<<1, array_len>>>(d_array, d_tmp_array, d_b_array, d_s_array, d_array_len);
    cudaError_t cuda_status = cudaGetLastError();
    if(cuda_status != cudaSuccess) {
        cout << "Kernel error!" << endl;
        goto cuda_error;
    }

    cudaMemcpy(gpu_array, d_array, sizeof(int) * array_len, cudaMemcpyDeviceToHost);

    if(array_len < PRINTING_LIMIT) {
        array_print(gpu_array, array_len, "After GPU sort");
    }

    //CPU radix sort

    //Finding maximum number
    for(int i = 0; i < array_len; i++) {
        if(init_array[i] > max_number) {
            max_number = init_array[i];
        }
    }

    //Finding maximum discharge
    while(true) {
        if(max_number % discharge_factor != max_number) {
            max_discharge++;
            discharge_factor *= 10;
        }
        else {
            break;
        }
    }

    cpu_radix_sort(init_array, array_len, max_discharge);

    if(array_len < PRINTING_LIMIT) {
        array_print(init_array, array_len, "After CPU sort");
    }

cuda_error:
    delete[] init_array;
    delete[] gpu_array;

    cudaFree(d_array);
    cudaFree(d_tmp_array);
    cudaFree(d_b_array);
    cudaFree(d_s_array);
    cudaFree(d_array_len);

    return 0;
}
