#include <time.h>

#define NN_IMPLEMENTATION 
#include "nn.h"

#define BITS 2

int main() {
    // PREPARE THE TRAINING SET FOR ADDER
    int n = (1<<BITS);
    int rows = n*n;
    Mat ti = mat_alloc(rows, 2*BITS);
    Mat to = mat_alloc(rows, BITS+1);
    for (int i = 0; i < rows; i++) {
        int x = i/n; 
        int y = i%n; 
        int z = x + y;
        for (int j = 0; j < BITS; j++) {
            MAT_AT(ti, i, j)      = (x>>(BITS-1-j)) & 1;
            MAT_AT(ti, i, j+BITS) = (y>>(BITS-1-j)) & 1;
            MAT_AT(to, i, j)      = (z>>(BITS-1-j)) & 1;
        }
        MAT_AT(to, i, BITS) = z >= n;
    }

    // MODEL LEARNING
    srand(time(0));
    float rate = 1;

    size_t arch[] = {2*BITS, 4*BITS, BITS+1};
    NN nn = nn_alloc(arch, ARRAY_LEN(arch));
    NN g  = nn_alloc(arch, ARRAY_LEN(arch));
    nn_rand(nn, 0, 1);

    // 1000 EPOCHS NOW IS ENOUGH
    for (size_t i = 0; i < 1000; ++i) {
        nn_backprop(nn, g, ti, to);
        nn_learn(nn, g, rate);
    }
    printf("COST = %f\n", nn_cost(nn, ti, to));

    // TESTING MODEL
    int fails = 0;
    for (int x = 0; x < n; x++) {
        for (int y = 0; y < n; y++) {
            int z = x + y;
            for (int j = 0; j < BITS; j++) {
                MAT_AT(NN_INPUT(nn), 0, j)      = x>>(BITS-1-j) & 1;
                MAT_AT(NN_INPUT(nn), 0, j+BITS) = y>>(BITS-1-j) & 1;
            }
            nn_forward(nn);

            int a = 0;
            for (int j = 0; j < BITS; j++) {
                a |= (MAT_AT(NN_OUTPUT(nn), 0, j) >  0.5f)<<(BITS-1-j);
            }
            int c = MAT_AT(NN_OUTPUT(nn), 0, BITS)>0.5f;

            // COMPARE THE MODEL FORWARDING WITH THE EXPECTED
            int ea  = z&(n-1);
            int ec = z >= n;
            if (a != ea || ec != c) {
                printf("%d + %d =  %d, c = %d\tExpected = %d, ec = %d\n"
                       , x, y, a, c, ea, ec);
                fails++;
            }
        }
    }
    if (fails == 0) printf("NICE MODEL :)\n");

    return 0;
}
