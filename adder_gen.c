#define NN_IMPLEMENTATION
#include "nn.h"

#define BITS 4

int main(void)
{
    int n = (1<<BITS);
    int rows = n*n;

    Mat t  = mat_alloc(rows, 3*BITS+1);
    Mat ti = {
        .rows   = t.rows,
        .cols   = 2*BITS,
        .stride = t.stride,
        .es     = t.es,
    };
    Mat to = {
        .rows   = t.rows,
        .cols   = BITS+1,
        .stride = t.stride,
        .es     = t.es + 2*BITS,
    };

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

    const char* pathname = "adder.mat";
    FILE * out_file = fopen(pathname, "wb");
    if (out_file == NULL) {
        fprintf(stderr, "Can't oper file %s\n", pathname);
        return 1;
    }
    mat_save(t, out_file);
    printf("Generated file %s\n", pathname);
    fclose(out_file);
    return 0;
}
