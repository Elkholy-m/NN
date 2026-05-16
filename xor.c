#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

// XOR FOUMULA: ``A XOR B = (A NAND B) AND (A OR B)``
/*
    XOR NEEDS AT LEAST 2 LAYERS WITH 3 NEURONS 
    EACH NEURON HAVE 2 INPUTS + BIAS
    SO THE NUMBER OF PARAMETERS IS GONE BE 9
*/

// THIS IS SHITTY CODE I KNOW 
// THESE VARIABLES PROPABLY SHOULD BE ARRAY OF FLOATS

typedef struct {
    float in1_and;
    float in2_and;
    float bi_and;

    float in1_or;
    float in2_or;
    float bi_or;

    float in1_nand;
    float in2_nand;
    float bi_nand;
} Xor;

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

const float xor_train[][3] = {
    {0, 0, 0},
    {0, 1, 1},
    {1, 0, 1},
    {1, 1, 0},
};

typedef float Sample[3];
#define SAMPLE_SIZE  4

float cost(Xor m);
float rand_float();
float sigmoidf(float in);
float forward(Xor m, float x1, float x2);
void print_xor(Xor m);
Xor rand_xor();
Xor gredient(Xor m, float eps);
Xor learn(Xor m, Xor g, float rate);
void test(Xor m);

const Sample* train = xor_train;

int main() {
    srand(time(0));
    const float eps  = 1e-1;
    const float rate = 1e-1;
    Xor m = rand_xor();

    printf("BEFORE:\n");
    print_xor(m);

    // train
    for (int i = 0; i < 100 * 1000; i++) {
        Xor g = gredient(m, eps);
        m = learn(m, g, rate);
    }

    printf("AFTER:\n");
    print_xor(m);
    printf("FINAL COST = %f\n", cost(m));

    // test
    test(m);

    return 0;
}

float cost(Xor m) {
    float comm_err = 0.f;

    for (size_t i = 0; i < SAMPLE_SIZE; i++) {
        float act_op = forward(m, train[i][0], train[i][1]);
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

float forward(Xor m, float x1, float x2) {
    float inter_1   = sigmoidf(x1 * m.in1_or + x2 * m.in2_or + m.bi_or);
    float inter_2   = sigmoidf(x1 * m.in1_nand + x2 * m.in2_nand + m.bi_nand);
    float actual_op = sigmoidf(inter_1 * m.in1_and + inter_2 * m.in2_and + m.bi_and);
    return actual_op;
}

Xor gredient(Xor m, float eps) {
    Xor g; 

    float saved = m.in1_or;
    float c = cost(m);
    m.in1_or += eps;
    g.in1_or = (cost(m) - c) / eps;
    m.in1_or = saved;

    saved = m.in2_or;
    c = cost(m);
    m.in2_or += eps;
    g.in2_or = (cost(m) - c) / eps;
    m.in2_or = saved;

    saved = m.bi_or;
    c = cost(m);
    m.bi_or += eps;
    g.bi_or = (cost(m) - c) / eps;
    m.bi_or = saved;

    saved = m.in1_and;
    c = cost(m);
    m.in1_and += eps;
    g.in1_and = (cost(m) - c) / eps;
    m.in1_and = saved;

    saved = m.in2_and;
    c = cost(m);
    m.in2_and += eps;
    g.in2_and = (cost(m) - c) / eps;
    m.in2_and = saved;

    saved = m.bi_and;
    c = cost(m);
    m.bi_and += eps;
    g.bi_and = (cost(m) - c) / eps;
    m.bi_and = saved;

    saved = m.in1_nand;
    c = cost(m);
    m.in1_nand += eps;
    g.in1_nand = (cost(m) - c) / eps;
    m.in1_nand = saved;

    saved = m.in2_nand;
    c = cost(m);
    m.in2_nand += eps;
    g.in2_nand = (cost(m) - c) / eps;
    m.in2_nand = saved;

    saved = m.bi_nand;
    c = cost(m);
    m.bi_nand += eps;
    g.bi_nand = (cost(m) - c) / eps;
    m.bi_nand = saved;

    return g;
}

Xor learn(Xor m, Xor g, float rate) {
    Xor res;

    res.in1_and = m.in1_and - rate* g.in1_and;
    res.in2_and = m.in2_and - rate * g.in2_and;
    res.bi_and = m.bi_and - rate * g.bi_and;
    res.in1_or = m.in1_or - rate * g.in1_or;
    res.in2_or = m.in2_or - rate * g.in2_or;
    res.bi_or = m.bi_or - rate * g.bi_or;
    res.in1_nand = m.in1_nand - rate * g.in1_nand;
    res.in2_nand = m.in2_nand - rate * g.in2_nand;
    res.bi_nand = m.bi_nand - rate * g.bi_nand;

    return res;
}

Xor rand_xor() {
    Xor m;

    m.in1_and = rand_float();
    m.in2_and = rand_float();
    m.bi_and = rand_float();
    m.in1_or = rand_float();
    m.in2_or = rand_float();
    m.bi_or = rand_float();
    m.in1_nand = rand_float();
    m.in2_nand = rand_float();
    m.bi_nand = rand_float();

    return m;
}

void print_xor(Xor m) {
    printf("-------------------------\n");
    printf("in1_and  = %f\n", m.in1_and);
    printf("in2_and  = %f\n", m.in2_and);
    printf("bi_and   = %f\n", m.bi_and);
    printf("in1_or   = %f\n", m.in1_or);
    printf("in2_or   = %f\n", m.in2_or);
    printf("bi_or    = %f\n", m.bi_or);
    printf("in1_nand = %f\n", m.in1_nand);
    printf("in2_nand = %f\n", m.in2_nand);
    printf("bi_nand  = %f\n", m.bi_nand);
    printf("-------------------------\n\n");
}

void test(Xor m) {
    printf("----------TEST-----------\n");
    for (size_t i = 0; i < 2; i++) {
        for (size_t j = 0; j < 2; j++) {
            printf("(%zu ^ %zu) = %f\n", i, j, forward(m, i, j));
        }
    }
}
