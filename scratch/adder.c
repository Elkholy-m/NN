#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

const float adder_train[][5] = {
// in1, in2, in3, res, carry
    {0, 0, 0, 0, 0},
    {0, 0, 1, 1, 0},
    {0, 1, 0, 1, 0},
    {0, 1, 1, 0, 1},
    {1, 0, 0, 1, 0},
    {1, 0, 1, 0, 1},
    {1, 1, 0, 0, 1},
    {1, 1, 1, 1, 1},
};

#define SAMPLE_SIZE 8

typedef float Sample[5];

typedef struct {
    float res;
    float carry;
} Output;

typedef struct {
    float x1;
    float x2;
    float x3;
} Input;

typedef struct {
    float params[20];
} Adder;

const Sample* train = adder_train;

float sigmoidf(float in) {
    return (1.f / (1.f + expf(-in)));
}

Output forward(Adder add, Input in) {
    Output out;

    float inter1 = sigmoidf(in.x1 * add.params[0] + in.x2 * add.params[1] + in.x3 * add.params[2] + add.params[3]);
    float inter2 = sigmoidf(in.x1 * add.params[4] + in.x2 * add.params[5] + in.x3 * add.params[6] + add.params[7]);
    float inter3 = sigmoidf(in.x1 * add.params[8] + in.x2 * add.params[9] + in.x3 * add.params[10] + add.params[11]);

    out.res =  sigmoidf(
        inter1 * add.params[12] +
        inter2 * add.params[13] +
        inter3 * add.params[14] +
        add.params[15]);

    out.carry = sigmoidf(
        inter1 * add.params[16] +
        inter2 * add.params[17] +
        inter3 * add.params[18] +
        add.params[19]);

    return out;
}

float cost(Adder add) {
    float comm_err = 0.f;

    for (size_t i = 0; i < SAMPLE_SIZE; i++) {
        Input in = {
            .x1 = train[i][0],
            .x2 = train[i][1],
            .x3 = train[i][2],
        };

        Output act_op = forward(add, in);
        float pred_op_res   = train[i][3];
        float pred_op_carry = train[i][4];
        float res_diff      = act_op.res - pred_op_res;
        float carry_diff    = act_op.carry - pred_op_carry;

        comm_err += (res_diff * res_diff + carry_diff * carry_diff) / 2;
    }

    return comm_err;
}

Adder gredient(Adder add, float eps) {
    Adder g;
    float c = cost(add);

    for (int i = 0; i < 20; i++) {
        float saved = add.params[i];
        add.params[i] += eps;
        g.params[i] = (cost(add) - c) / eps;
        add.params[i] = saved;
    }

    return g;
}

Adder learn(Adder add, Adder g, float rate) {
    Adder res;
    for (int i = 0; i < 20; i++) {
        res.params[i] = add.params[i] - (rate * g.params[i]);
    }

    return res;
}

float rand_float() {
    return (float)rand() / RAND_MAX;
}

Adder rand_adder() {
    Adder add;
    for (int i = 0; i < 20; i++) {
        add.params[i] = rand_float();
    }
    return add;
}

void print_adder(Adder add) {
    printf("----------PRINT---------\n");
    for (int i = 0; i < 20; i++) {
        printf("params[%d] = %f\n",i , add.params[i]);
    }
    printf("------------------------\n\n");
}

void test(Adder add) {
    printf("----------TEST----------\n");
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            for(int k = 0; k < 2; k++) {
                Input in = {
                    .x1 = i,
                    .x2 = j,
                    .x3 = k,
                };

                Output out = forward(add, in);
                printf("(%d + %d + %d) => res = %f, carry = %f\n", i, j, k, out.res, out.carry);
            }
        }
    }
    printf("------------------------\n\n");
}


int main() {
    srand(23);
    float eps  = 1e-1;
    float rate = 1e-1;
    Adder add = rand_adder();

    printf("Cost: %f\n", cost(add));

    // train 
    for (int i = 0; i < 500*1000; i++) {
        Adder g = gredient(add, eps);
        add = learn(add, g, rate);
    }

    test(add);
    printf("Cost: %f\n", cost(add));

    return 0;
}
