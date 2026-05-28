#include <stdio.h>
#include <time.h>
#include <assert.h>

#define NN_IMPLEMENTATION
#include "nn.h"

float xor_td[] = {
    0, 0, 0,
    0, 1, 1,
    1, 0, 1,
    1, 1, 0,
};

float or_td[] = {
    0, 0, 0,
    0, 1, 1,
    1, 0, 1,
    1, 1, 1,
};

float adder_train[]= {
// in1, in2, in3, res, carry
    0, 0, 0, 0, 0,
    0, 0, 1, 1, 0,
    0, 1, 0, 1, 0,
    0, 1, 1, 0, 1,
    1, 0, 0, 1, 0,
    1, 0, 1, 0, 1,
    1, 1, 0, 0, 1,
    1, 1, 1, 1, 1,
};

Mat xor = { .rows = 4, .cols = 3, .stride = 3, .es = xor_td, };
Mat or = { .rows = 4, .cols = 3, .stride = 3, .es = or_td, };
Mat adder = { .rows = 8, .cols = 5, .stride = 5, .es = adder_train, };

void xor_test(NN nn) {
    for (size_t i = 0; i < 2; i++) {
        for (size_t j = 0; j < 2; j++) {
            MAT_AT(NN_INPUT(nn), 0, 0) = i;
            MAT_AT(NN_INPUT(nn), 0, 1) = j;
            nn_forward(nn);
            float y =  MAT_AT(NN_OUTPUT(nn), 0, 0);
            printf("%zu ^ %zu = %f\n", i, j, y);
        }
    }
}

void adder_test(NN nn) {
    for (size_t i = 0; i < 2; i++) {
        for (size_t j = 0; j < 2; j++) {
            for (size_t k = 0; k < 2; k++) {
                MAT_AT(NN_INPUT(nn), 0, 0) = i;
                MAT_AT(NN_INPUT(nn), 0, 1) = j;
                MAT_AT(NN_INPUT(nn), 0, 2) = k;
                nn_forward(nn);
                float sum   =  MAT_AT(NN_OUTPUT(nn), 0, 0);
                float carry =  MAT_AT(NN_OUTPUT(nn), 0, 1);
                printf("%zu + %zu + %zu = %f, Carry = %f\n", i, j, k, sum, carry);
            }
        }
    }
}


int main() {
    srand(23);
    size_t arch[] = {2, 2, 1};
    NN nn = nn_alloc(arch, ARRAY_LEN(arch));
    NN g  = nn_alloc(arch, ARRAY_LEN(arch));
    nn_rand(nn, 0, 1);

    Mat m = xor;

    Mat ti = mat_sub(m, COORDINATE(0, 0), COORDINATE(3, 1));
    Mat to = mat_sub(m, COORDINATE(0, 2), COORDINATE(3, 2));

    float rate = 1e-1;

    // THIS IS CALLED 40-000 EPOCHS
    for (int i = 0; i < 100*1000; i++) {
        nn_backprop(nn, g, ti, to);
        nn_learn(nn, g, rate);
    }

    printf("Cost = %f\n", nn_cost(nn, ti, to));
    NN_PRINT(nn);
    xor_test(nn);
}
