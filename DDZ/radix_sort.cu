#include <iostream>
#include <fstream>
#include <cstdlib>
#include <ctime>
#include "cuda_runtime.h"

using namespace std;

__global__ void fast_radix_sort(int *array, int array_len) {
    extern __shared__ int tmp_array[];
    int *b_array = tmp_array + array_len;
    int *s_array = tmp_array + array_len * 2;
    int *t_array = tmp_array + array_len * 3;

    tmp_array[threadIdx.x] = array[threadIdx.x + array_len * blockIdx.x];

    __syncthreads();

    for(int i = 0; i < sizeof(int) * 8; i++) {
        b_array[threadIdx.x] = (tmp_array[threadIdx.x] >> i) & 1;

        __syncthreads();
        if (threadIdx.x == 0) {
            s_array[0] = 0;
            for (int i = 1; i < array_len + 1; i++) {
                s_array[i] = s_array[i - 1] + b_array[i - 1];
            }
        }
        __syncthreads();

        if (b_array[threadIdx.x] == 0) {
            t_array[threadIdx.x - s_array[threadIdx.x]] = tmp_array[threadIdx.x];
        }
        else {
            t_array[s_array[threadIdx.x] + (array_len - s_array[array_len])] = tmp_array[threadIdx.x];
        }

        __syncthreads();
        tmp_array[threadIdx.x] = t_array[threadIdx.x];
        __syncthreads();
    }

    __syncthreads();
    array[threadIdx.x + array_len * blockIdx.x] = tmp_array[threadIdx.x];    
}

void merge(int *array1, int *array2, int array1_len, int array2_len) {
	int i = 0, j = 0, total_array_len = array1_len + array2_len;
	int *new_array = new int[total_array_len];

	for (int k = 0; k < total_array_len; k++) {
		if (i == array1_len) {
			new_array[k] = array2[j++];
		}
		else if (j == array2_len) {
			new_array[k] = array1[i++];
		}
		else if (array1[i] < array2[j]) {
			new_array[k] = array1[i++];
		}
		else {
			new_array[k] = array2[j++];
		}
	}

	memcpy(array1, new_array, sizeof(int) * total_array_len);
	delete[] new_array;
}

void cpu_radix_sort(int* array, int array_len) {
	bool *b_array = new bool[array_len];
	int *s_array = new int[array_len + 1];
	int *tmp_array = new int[array_len];

	int j;

	for (int k = 0; k < sizeof(int) * 8; k++) {
		for (int i = 0; i < array_len; i++) {
			b_array[i] = (array[i] >> k) & 1;
		}

		s_array[0] = 0;
		for (int i = 1; i < array_len + 1; i++) {
			s_array[i] = s_array[i - 1] + b_array[i - 1];
		}

		for (int i = 0; i < array_len; i++) {
			if (b_array[i] == 0) {
				tmp_array[i - s_array[i]] = array[i];
			}
			else {
				j = s_array[i] + (array_len - s_array[array_len]);
				tmp_array[j] = array[i];
			}
		}

		for (int i = 0; i < array_len; i++) {
			array[i] = tmp_array[i];
		}
	};

	delete[] b_array;
	delete[] s_array;
	delete[] tmp_array;
}

void array_print(int* array, int array_len, const char* message) {
    cout << " " << message << ":\n [ ";
    for (int i = 0; i < array_len; i++) {
        cout << array[i] << " ";
    }
    cout << "]" << endl;
}

int main(int argc, char** argv) {
    int const PRINTING_LIMIT = 101;
    int array_len = 50, start = 0, stop = 101;
    
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
        cout << " #Error# Array length is too small. at least 2!" << endl;
        return 0;
    }

    int *init_array = new int[array_len];
    int *gpu_array = new int[array_len];

    clock_t c_start, c_end;

    ofstream file_out("res.csv", ios_base::app);

    //Randomizing array
    srand(time(NULL));
    for (int i = 0; i < array_len; i++) {
        init_array[i] = start + rand() % (stop - 10);
    }

    if(array_len < PRINTING_LIMIT) {
        array_print(init_array, array_len, "Initial array");
    }

    //GPU radix sort
    int *d_array;
    float working_time;
    double gpu_time, cpu_time, cpu_merge_time; 
    int block_num, thread_num, subarray_len;

    // Splitting data to blocks
    for (int f = 1024; f > 0; f--) {
        if (array_len % f == 0) {
            block_num = array_len / f;
            thread_num = subarray_len = f;
            break;
        }
    }

    cudaEvent_t e_start, e_stop;
    cudaError_t cuda_status;

    cudaEventCreate(&e_start);
    cudaEventCreate(&e_stop);

    cudaMalloc((void**)&d_array, sizeof(int) * array_len);

    cudaMemcpy(d_array, init_array, sizeof(int) * array_len, cudaMemcpyHostToDevice);

    cudaEventRecord(e_start);
    fast_radix_sort<<<block_num, thread_num, (subarray_len * sizeof(int)) * 4>>>(d_array, subarray_len);
    cudaEventRecord(e_stop);

    cuda_status = cudaGetLastError();
    if(cuda_status != cudaSuccess) {
        cout << " #Error# CUDA kernel error!" << endl;
        goto cuda_error;
    }

    cudaDeviceSynchronize();

    cudaEventSynchronize(e_stop);
    cudaEventElapsedTime(&working_time, e_start, e_stop);

    cudaMemcpy(gpu_array, d_array, sizeof(int) * array_len, cudaMemcpyDeviceToHost);

    //Merging sorted parts of array
    c_start = clock();
    for (int i = 0; i < block_num - 1; i++) {
        merge(gpu_array, gpu_array + subarray_len * (i + 1), subarray_len * (i + 1), subarray_len);
    }
    c_end = clock();

    //Printing if alloweded
    if(array_len < PRINTING_LIMIT) {
        array_print(gpu_array, array_len, "After GPU sort");
    }

    gpu_time = working_time / 1000;
    cout << " GPU sorting time: " << gpu_time << " s" << endl;
    
    cpu_merge_time = (double)(c_end - c_start) / CLOCKS_PER_SEC;
    cout << " Merging time: ";
    if (cpu_merge_time == 0) {
        cout << "less then 0.001 s" << endl;
    }
    else {
        cout << cpu_merge_time << " s" << endl;
    }

    //CPU radix sort
    c_start = clock();
    cpu_radix_sort(init_array, array_len);
    c_end = clock();

    if(array_len < PRINTING_LIMIT) {
        array_print(init_array, array_len, "After CPU sort");
    }

    cpu_time = (double)(c_end - c_start) / CLOCKS_PER_SEC;
    cout << " CPU sorting time: ";
    if (cpu_merge_time == 0) {
        cout << "less then 0.001 s" << endl;
    }
    else {
        cout << cpu_time << " s" << endl;
    }

    //logging
    file_out << array_len << ';' << gpu_time << ';' << cpu_merge_time << ';' << cpu_time << ';' << endl;

cuda_error:
    file_out.close();

    delete[] init_array;
    delete[] gpu_array;

    cudaEventDestroy(e_start);
    cudaEventDestroy(e_stop);

    cudaFree(d_array);

    cudaDeviceReset();

    return 0;
}
