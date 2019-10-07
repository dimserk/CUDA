#include <iostream>
#include <cstdlib>
#include <ctime>

#define VEC_SIZE 200
#define START 1
#define STOP 100

using namespace std;

int main()
{
    int *arr_a, *arr_b, total_sum;

    arr_a = new int[VEC_SIZE];
    arr_b = new int[VEC_SIZE];

    srand(time(NULL));
    for (int i = 0; i < VEC_SIZE; i++)
    {
        arr_a[i] = START + rand() % STOP;
        arr_b[i] = START + rand() % STOP;
    }

    total_sum = 0; 
    for (int i = 0; i < VEC_SIZE; i++)
    {
        total_sum += arr_a[i] * arr_b[i];
    }
    
    cout << "Result of vectors multiplication is " << total_sum << endl;

    delete[] arr_a;
    delete[] arr_b;

    return 0;
}