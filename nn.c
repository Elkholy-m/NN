#include <stdio.h>
#include <time.h>

#define NN_IMPLEMENTATION
#include "nn.h"

int main() {
    srand(time(0));
    float unit_arr[] = {
        1 ,0,
        0 ,1,
    };

    Mat b = mat_alloc(2, 2);
    mat_rand(b, 0, 10);

    Mat unit_mat = {
        .rows = 2,
        .cols = 2,
        .data = unit_arr,
    };

    MAT_PRINT(b);
    MAT_PRINT(unit_mat);
    printf("----------------------\n");
    Mat dest = mat_alloc(2, 2);
    mat_dot(dest, unit_mat, b);
    MAT_PRINT(dest);


    return 0;
}
