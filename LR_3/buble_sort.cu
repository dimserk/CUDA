#include <iostream>
#include <cstdlib>
#include <ctime>
#include "cuda_runtime.h"

using namespace std;

__global__ void buble_sort();

void buble_sort(int* array, int array_size);
void swap(int* first, int* second);
void array_print(int* array, int array_size, const char* message);

int main(int argc, char** argv)
{
    int array_size, start, stop;
    
    //Obtaining command line arguments
    switch (argc)
    {
    case 1:
        array_size = 15;
        cout << " #Warning# Default array size: " << array_size << endl;
        start = 0;
        cout << " #Warning# Default random start: " << start << endl;
        stop = 100;
        cout << " #Warning# Default random stop: " << stop << endl;
        cout << endl;
        break;
    case 2:
        array_size = atoi(argv[1]);
        start = 0;
        cout << " #Warning# Default random start: " << start << endl;
        stop = 100;
        cout << " #Warning# Default random stop: " << stop << endl;
        cout << endl;
        break;
    case 4:
        array_size = atoi(argv[1]);
        start = atoi(argv[2]);
        stop = atoi(argv[3]);
        cout << endl;
        break;   
    default:
        cout << "Wrong input!" << endl;
    }

    int* init_array = new int[array_size];

    srand(time(NULL));
    for (int i = 0; i < array_size; i++)
    {
        init_array[i] = start + rand() % stop;
    }

    array_print(init_array, array_size, "Initial array");

    buble_sort(init_array, array_size);

    array_print(init_array, array_size, "Array after CPU sort");

    return 0;
}

__global__ void buble_sort()
{
    printf("Hello from device!\n");
}

void buble_sort(int* array, int array_size)
{
    for(int i = 1; i < array_size - 1; i++)
    {
        for(int j = 0; j < array_size - i; j++)
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

void array_print(int* array, int array_size, const char* message)
{
    cout << " " << message << ":\n [ ";
    for (int i = 0; i < array_size; i++)
    {
        cout << array[i] << " ";
    }
    cout << "]" << endl << endl;
}
