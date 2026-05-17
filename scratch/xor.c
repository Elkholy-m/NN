#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define SAMPLE_SIZE 4

const float adder_train[][3] = {
// in1, in2, res
    {0, 0, 0},
    {0, 1, 1},
    {1, 0, 1},
    {1, 1, 0},
};

typedef struct {
    float params[9];
} Xor;

typedef float Sample[3];

const Sample* train = adder_train;

float sigmoidf(float in) {
    return (1.f / (1.f + expf(-in)));
}

float forward_xor(Xor m, float x1, float x2) {
     float inter_1 = sigmoidf(m.params[0] * x1 + m.params[1] * x2 + m.params[2]);
     float inter_2 = sigmoidf(m.params[3] * x1 + m.params[4] * x2 + m.params[5]);

    return sigmoidf(m.params[6] * inter_1 + m.params[7] * inter_2 + m.params[8]);
}

float rand_float() {
    return (float)rand() / RAND_MAX;
}

Xor rand_xor() {
    Xor m;
    for (int i = 0; i < 9; i++) {
        m.params[i] = rand_float();
    }
    return m;
}

void print_xor(Xor m) {
    printf("----------PRINT---------\n");
    for (int i = 0; i < 9; i++) {
        printf("params[%d] = %f\n",i , m.params[i]);
    }
    printf("------------------------\n\n");
}

float cost_xor(Xor m) {
    float comm_err = 0.f;

    for (size_t i = 0; i < SAMPLE_SIZE; i++) {
        float act_op = forward_xor(m, train[i][0], train[i][1]);
        float pred_op = train[i][2];
        float diff = act_op - pred_op;
        comm_err += diff * diff;
    }

    return comm_err;
}

Xor gredient_xor(Xor m, float eps) {
    Xor g;
    float cost = cost_xor(m);
    for (int i = 0; i < 9; i++) {
        float saved = m.params[i];
        m.params[i] += eps;
        g.params[i] = (cost_xor(m) - cost) / eps;
        m.params[i] = saved;
    }

    return g;
}

Xor learn(Xor m, Xor g, float rate) {
    Xor res;

    for (int i = 0; i < 9; i++) {
        res.params[i] = m.params[i] - rate * g.params[i];
    }

    return res;
}

void test(Xor m) {
    printf("----------TEST----------\n");
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            printf("(%d ^ %d) = %f\n", i, j, forward_xor(m, i, j));
        }
    }
    printf("------------------------\n\n");
}

int main() {
    srand(23);
    float eps  = 1e-1;
    float rate = 1e-1;
    Xor m = rand_xor();

    for (int i = 0; i < 100*1000; i++) {
        Xor g = gredient_xor(m, eps);
        m = learn(m, g, rate);
    }

    test(m);
    printf("FINAL COST = %f\n", cost_xor(m));
    return 0;
}


