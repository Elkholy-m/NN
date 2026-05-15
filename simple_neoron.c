#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/* 
  EQUATION THAT MODEL THE INPUTS TO THE OUTPUTS
  Y = M*X WHERE M = 2
  THA MODEL SHOULD GET THE VALUE CLOSE TO 2
  USING MEAN SQUARE ERROR COST FUNCTION
*/

const float train[][2] = {
    {0, 0},
    {1, 2},
    {2, 4},
    {3, 6},
    {4, 8},
};

#define TRAIN_COUNT  sizeof(train) / sizeof(train[0])

float cost(float weight);
float rand_float();

int main() {
    srand(23);
    float weight = rand_float() * 10.f;
    float eps  = 1e-3;
    float rate = 1e-3;

    printf("--------------------------\n");
    for (int i = 0; i < 1000; i++) {
        float c = cost(weight);
        float der = (cost(weight + eps) - c) / eps;
        weight -= rate*der;
        c = cost(weight);
        printf("cost: %f\n", c);
    }
    printf("--------------------------\n");
    printf("the final weight value = %f\n", weight);

    return 0;
}

float cost(float weight) {
    float comm_err = 0.f;

    for (size_t i = 0; i < TRAIN_COUNT; i++) {
        float input = train[i][0];
        float act_op = weight * input;
        float pred_op = train[i][1];
        float diff = act_op - pred_op;
        comm_err += diff * diff;
    }

    return comm_err;
}

float rand_float() {
    return ((float)rand() / RAND_MAX);
}
