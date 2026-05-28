#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

const float xor_train[][3] = {
    {0, 0, 0},
    {0, 1, 1},
    {1, 0, 1},
    {1, 1, 0},
};

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
void gcost(Sample p, Sample g);
float rand_float();
float sigmoidf(float in);

const Sample* train = nand_train;

int main () {
    srand(time(0));
    float rate = 1e-1;

    float b = rand_float();
    float w1   = rand_float();
    float w2   = rand_float();

    // printf("----------TRAIN-----------\n");
    for (int i = 0; i < 100*1000; i++) {
        float w1_der;
        float w2_der;
        float b_der;
#if 0
        float c = cost(w1, w2, b);
        float eps  = 1e-1;
        w1_der =   (cost(w1 + eps, w2, b) - c) / eps;
        w2_der =   (cost(w1, w2 + eps, b) - c) / eps;
        b_der = (cost(w1, w2, b + eps) - c) / eps;
#else
        Sample p = {w1, w2, b};
        Sample g = {0};
        gcost(p, g);
        w1_der = g[0];
        w2_der = g[1];
        b_der  = g[2];
#endif

        w1 -= rate*w1_der;
        w2 -= rate*w2_der;
        b -= rate*b_der;

        // c = cost(w1, w2, b);
        // printf("cost: %f\n", c);
    }
    // printf("--------------------------\n\n");

    printf("-----------TEST--------\n");
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++ ) {
            printf("(%d | %d) = %f\n", i, j, sigmoidf(i*w1 + j*w2 + b));
        }
    }

    printf("--------------------------\n\n");
    printf("w1: %f, w2: %f,b: %f\n", w1, w2, b);
    printf("cost: %f\n", cost(w1, w2, b));

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

void gcost(Sample p, Sample g) {
    g[0] = 0;
    g[1] = 0;
    g[2] = 0;

    size_t n = SAMPLE_SIZE ;
    float w1 = p[0];
    float w2 = p[1];
    float b  = p[2];
        
    for (size_t i = 0; i < n; i++) {
        float xi = train[i][0];
        float yi = train[i][1];
        float zi = train[i][2];
        float ai = sigmoidf(xi*w1 + yi*w2 + b);

        // calc the gredient here
        g[0] += 2*(ai - zi)*(ai* (1 - ai) * xi);
        g[1] += 2*(ai - zi)*(ai* (1 - ai) * yi);
        g[2] += 2*(ai - zi)*(ai* (1 - ai));
    }

    for (int i = 0; i < 3; i++) {
        g[i] /= n;
    }
}
