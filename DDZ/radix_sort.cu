#include <iostream>
#include <cstdlib>
#include <ctime>
#include <vector>
#include <list>
#include "cuda_runtime.h"


using namespace std;

__global__ void kernel()
{

}

void cpu_radix_sort(int* array, int array_len, int discharge)
{
    auto *tmp_lists = new list<int>[10];
    int factor = 10;

    for (int d = 0; d < discharge; d++)
    {
        for(int i = 0; i < array_len; i++)
        {
            int j = array[i] % factor / (factor / 10);
            tmp_lists[j].push_back(array[i]);
        }

        int init_ind = 0;
        for(int i = 0; i < 10; i++)
        {
            if(!tmp_lists[i].empty())
            {
                int size = tmp_lists[i].size();
                for(int j = 0; j < size; j++)
                {
                    array[init_ind] = tmp_lists[i].front();
                    init_ind++;
                    tmp_lists[i].pop_front();
                } 
            }
        }

        factor *= 10;
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
    int array_len = 15, start = 0, stop = 101;
    
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
    int *init_array = new int[array_len];

    //Randomizing array
    srand(time(NULL));
    for (int i = 0; i < array_len; i++)
    {
        init_array[i] = start + rand() % stop;
    }

    array_print(init_array, array_len, "Initial array");

    //Finding maximum number
    int max_number = 0;
    for(int i = 0; i < array_len; i++)
    {
        if(init_array[i] > max_number)
        {
            max_number = init_array[i];
        }
    }

    //Finding maximum discharge
    int max_discharge = 0, discharge_factor = 1;
    while(true)
    {
        if(max_number % discharge_factor != max_number)
        {
            max_discharge++;
            discharge_factor *= 10;
        }
        else 
        {
            break;
        }
    }

    //CPU radix sort
    cpu_radix_sort(init_array, array_len, max_discharge);

    array_print(init_array, array_len, "After CPU sort");

    return 0;
}
