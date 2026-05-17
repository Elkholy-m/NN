#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

const float and_train[][3] = {
    {0, 0, 0},
    {0, 1, 0},
    {1, 0, 0},
    {1, 1, 1},
};

const float or_train[][3] = {
    {0, 0, 0},
    {0, 1, 1},
    {1, 0, 1},
    {1, 1, 1},
};

const float nand_train[][3] = {
    {0, 0, 1},
    {0, 1, 1},
    {1, 0, 1},
    {1, 1, 0},
};

const float nor_train[][3] = {
    {0, 0, 1},
    {0, 1, 0},
    {1, 0, 0},
    {1, 1, 0},
};

typedef float Sample[3];
#define SAMPLE_SIZE  4

float cost(float w1, float w2, float bias);
float rand_float();
float sigmoidf(float in);

const Sample* train = or_train;

int main () {
    srand(time(0));
    float eps  = 1e-1;
    float rate = 1e-1;

    float bias = rand_float();
    float w1   = rand_float();
    float w2   = rand_float();

    // printf("----------TRAIN-----------\n");
    for (int i = 0; i < 100*1000; i++) {
        float c = cost(w1, w2, bias);
        float w1_der =   (cost(w1 + eps, w2, bias) - c) / eps;
        float w2_der =   (cost(w1, w2 + eps, bias) - c) / eps;
        float bias_der = (cost(w1, w2, bias + eps) - c) / eps;

        w1 -= rate*w1_der;
        w2 -= rate*w2_der;
        bias -= rate*bias_der;

        c = cost(w1, w2, bias);
        // printf("cost: %f\n", c);
    }
    // printf("--------------------------\n\n");

    printf("----------OR-TEST--------\n");
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++ ) {
            printf("(%d | %d) = %f\n", i, j, sigmoidf(i*w1 + j*w2 + bias));
        }
    }

    printf("--------------------------\n\n");
    printf("w1: %f, w2: %f,b: %f\n", w1, w2, bias);
    printf("cost: %f\n", cost(w1, w2, bias));

    return 0;
}

float cost(float w1, float w2, float bias) {
    float comm_err = 0.f;

    for (size_t i = 0; i < SAMPLE_SIZE; i++) {
        float x1 = train[i][0];
        float x2 = train[i][1];
        float act_op = sigmoidf(w1 * x1 + w2 * x2 + bias);
        float pred_op = train[i][2];
        float diff = act_op - pred_op;
        comm_err += diff * diff;
    }

    return comm_err;
}

float rand_float() {
    return (float)rand() / RAND_MAX;
}

float sigmoidf(float in) {
    return (1.f / (1.f + expf(-in)));
}
