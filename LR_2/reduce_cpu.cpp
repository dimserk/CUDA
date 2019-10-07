#include <iostream>
#include <cstdlib>
#include <ctime>

using namespace std;

int main(int argc, char** argv)
{
    int array_size, start, stop;

    switch (argc)
    {
    case 1:
        array_size = 100;
        cout << "Default array size: " << array_size << endl;
        start = 0;
        cout << "Default random start: " << start << endl;
        stop = 100;
        cout << "Default random stop: " << stop << endl;
        break;
    case 2:
        array_size = atoi(argv[1]);
        start = 0;
        cout << "Default random start: " << start << endl;
        stop = 100;
        cout << "Default random stop: " << stop << endl;
        break;
    case 4:
        array_size = atoi(argv[1]);
        start = atoi(argv[2]);
        stop = atoi(argv[3]);
        break;   
    default:
        cout << "Wrong input!" << endl;
    }
    
    int *array = new int[array_size];

    srand(time(NULL));
    for (int i = 0; i < array_size; i++)
    {
        array[i] = start + rand() % stop;
    }
    
    cout << "Initial array: ";
    for(int i = 0; i < array_size; i++)
    {
        if(i % 5 == 0)
            cout << endl;
        cout << array[i] << " ";
    }
    cout << endl;

    int sum = 0;

    for(int i = 0; i < array_size; i++)
    {
        sum += array[i];
    }

    cout << "Sum of array: " << sum  << endl;

    delete[] array;

    return 0;
}
