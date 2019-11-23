#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <fstream>
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

int main(int argc, char** argv) {
    int ARR_LEN = atoi(argv[1]);
    // int deviceCount;
    // cudaDeviceProp deviceProp;
  
    // //Сколько устройств CUDA установлено на PC.
    // cudaGetDeviceCount(&deviceCount);
  
    // printf("Device count: %d\n\n", deviceCount);
  
    // for (int i = 0; i < deviceCount; i++)
    // {
    //   //Получаем информацию об устройстве
    //   cudaGetDeviceProperties(&deviceProp, i);
  
    //   //Выводим иформацию об устройстве
    //   printf("Device name: %s\n", deviceProp.name);
    //   printf("Total global memory: %d\n", deviceProp.totalGlobalMem);
    //   printf("Shared memory per block: %d\n", deviceProp.sharedMemPerBlock);
    //   printf("Registers per block: %d\n", deviceProp.regsPerBlock);
    //   printf("Warp size: %d\n", deviceProp.warpSize);
    //   printf("Memory pitch: %d\n", deviceProp.memPitch);
    //   printf("Max threads per block: %d\n", deviceProp.maxThreadsPerBlock);
      
    //   printf("Max threads dimensions: x = %d, y = %d, z = %d\n",
    //     deviceProp.maxThreadsDim[0],
    //     deviceProp.maxThreadsDim[1],
    //     deviceProp.maxThreadsDim[2]);
      
    //   printf("Max grid size: x = %d, y = %d, z = %d\n",
    //     deviceProp.maxGridSize[0],
    //     deviceProp.maxGridSize[1],
    //     deviceProp.maxGridSize[2]);
  
    //   printf("Clock rate: %d\n", deviceProp.clockRate);
    //   printf("Total constant memory: %d\n", deviceProp.totalConstMem);
    //   printf("Compute capability: %d.%d\n", deviceProp.major, deviceProp.minor);
    //   printf("Texture alignment: %d\n", deviceProp.textureAlignment);
    //   printf("Device overlap: %d\n", deviceProp.deviceOverlap);
    //   printf("Multiprocessor count: %d\n", deviceProp.multiProcessorCount);
  
    //   printf("Kernel execution timeout enabled: %s\n",
    //     deviceProp.kernelExecTimeoutEnabled ? "true" : "false");
    // }







    int *array = new int[ARR_LEN];
    int *d_array;
    
    int block_num, thread_num, array_len;

    for (int f = 1024; f > 0; f--) {
        if (ARR_LEN % f == 0) {
            block_num = ARR_LEN / f;
            thread_num = f;
            array_len = f;
            break;
        }
    }

    cout << "BlockNum: " << block_num << " ThredNum: " << thread_num << " ArrayLen: " << array_len << endl;

    float gpu_time, working_time;
    cudaEvent_t e_start, e_stop;

    srand(time(NULL));
    for (int i = 0; i < ARR_LEN; i++) {
        array[i] = 1 + rand() % 100;
    }

    // for (int i = 0; i < ARR_LEN; i++) {
    //     printf("%d ", array[i]);
    // }
    // printf("\n");

    cudaEventCreate(&e_start);
    cudaEventCreate(&e_stop);

    cudaError_t cuda_status;

    cuda_status = cudaMalloc((void**)&d_array, ARR_LEN * sizeof(int));

    cuda_status = cudaMemcpy(d_array, array, ARR_LEN * sizeof(int), cudaMemcpyHostToDevice);

    cudaEventRecord(e_start);
    fast_radix_sort<<<block_num, thread_num, (array_len * sizeof(int)) * 4>>>(d_array, array_len);
    cudaEventRecord(e_stop);

    cuda_status = cudaGetLastError();
    if(cuda_status != cudaSuccess) {
        cout << " #Error# CUDA fast_radix_sort error!" << endl;
        goto cuda_error;
    }

    cudaDeviceSynchronize();
    
    cudaEventSynchronize(e_stop);
    cudaEventElapsedTime(&working_time, e_start, e_stop);

    cudaMemcpy(array, d_array, ARR_LEN * sizeof(int), cudaMemcpyDeviceToHost);

    double cpu_time;
    clock_t c_start, c_end;

    c_start = clock();
    for (int i = 0; i < block_num - 1; i++) {
        merge(array, array + array_len * (i + 1), array_len * (i + 1), array_len);
    }
    c_end = clock();

    for (int i = 0; i < ARR_LEN; i++) {
        printf("%d ", array[i]);
    }
    printf("\n");

    cpu_time = (double)(c_end - c_start) / CLOCKS_PER_SEC;
    cout << " Merging time: " << cpu_time << " s" << endl;

    gpu_time = working_time / 1000;
    cout << " GPU sorting time: " << gpu_time << " s" << endl;

cuda_error:
    cudaEventDestroy(e_start);
    cudaEventDestroy(e_stop);

    cudaFree(d_array);

    // for (int i = 0; i < ARR_LEN; i++) {
    //     printf("%d ", array[i]);
    // }
    // printf("\n");
    ofstream out("out.txt");
    for (int j = 0; j < ARR_LEN; j++) {
        out << array[j] << endl;
    //     for (int i = 0; i < block_num; i+=2) {
    //         merge(array + array_len * i, array + array_len * (i+j), array_len, array_len);
    //     }
    }
    out.close();

    // double cpu_time;
    // clock_t c_start, c_end;

    // c_start = clock();
    // merge(array, array + array_len, array_len, array_len);
    // merge(array, array + array_len * 2, array_len * 2, array_len);
    // c_end = clock();

    // cpu_time = (double)(c_end - c_start) / CLOCKS_PER_SEC;
    // cout << " Merging time: " << cpu_time << " s" << endl;
    delete[] array;
    return 0;
}
