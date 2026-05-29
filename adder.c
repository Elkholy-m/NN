#define NN_IMPLEMENTATION 
#include "nn.h"

#define BITS 2

void make_train(float arr[]) {
    int count = 0;
    int big_num = 1<<(2*BITS);
    int lower_mask = (1<<BITS) - 1;
    int upper_mask = lower_mask << BITS;

    for (int i = 0; i < big_num; i++) {
        for (int j = 2*BITS - 1; j >= 0; j--) {
            int bit = (i >> j) & 1;
            arr[count++] = bit;
        }
        int n1 = (i&upper_mask) >> BITS;
        int n2 = (i&lower_mask);

        int r = (n1 + n2) & lower_mask;
        for (int j = BITS - 1; j >= 0; j--) {
            int bit = (r >> j) & 1;
            arr[count++] = bit;
        }
        int c = ((n1 + n2) >> BITS) == 1;
        arr[count++] = c;
    }
}

void adder_test(NN nn, Mat ti) {
    printf("\n----------TEST----------\n");
    int lower_mask = (1<<BITS) - 1;
    int upper_mask = lower_mask << BITS;
    for (size_t i = 0; i < ti.rows; i++) {
        mat_copy(NN_INPUT(nn), mat_row(ti, i));
        nn_forward(nn);
        int n1 = (i&upper_mask) >> BITS;
        int n2 = (i&lower_mask);
        float res = 0.f;
        for (int j = 0; j < BITS; j++) {
             res += MAT_AT(NN_OUTPUT(nn), 0, BITS - 1 - j) * powf(2, j);
        }
        float c  = MAT_AT(NN_OUTPUT(nn), 0, 2);
        printf("%d + %d = %f, carry = %f\n", n1, n2, res, c);
    }
}

int main() {
    // PREPARE THE TRAINING SET FOR ADDER
    int rows = (1<<2*BITS);
    int cols = (3*BITS + 1);
    float arr[rows*cols];
    make_train(arr);
    Mat ti = {
        .rows = rows,
        .cols = 2*BITS,
        .stride = cols,
        .es = arr,
    };

    Mat to = {
        .rows = rows,
        .cols = BITS + 1,
        .stride = cols,
        .es = arr + 2*BITS,
    };

    srand(23);
    float rate = 1e-1;

    size_t arch[] = {4, 6, 4, 6, 3};
    NN nn = nn_alloc(arch, ARRAY_LEN(arch));
    NN g  = nn_alloc(arch, ARRAY_LEN(arch));
    nn_rand(nn, 0, 1);

    // 100_000 EPOCHS
    for (size_t i = 0; i < 100*1000; ++i) {
        nn_backprop(nn, g, ti, to);
        nn_learn(nn, g, rate);
    }

    NN_PRINT(nn);
    adder_test(nn, ti);
    printf("COST = %f\n", nn_cost(nn, ti, to));
    return 0;
}
