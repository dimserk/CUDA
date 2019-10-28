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

    //radix_sort wip
    cpu_radix_sort(init_array, array_len, max_discharge);

    array_print(init_array, array_len, "After cpu sort");

    return 0;
}
