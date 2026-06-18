#define NN_IMPLEMENTATION
#include "nn.h"

int main(void)
{
    Mat t = mat_alloc(4, 3);
    Mat ti = {
        .rows   = t.rows,
        .cols   = 2,
        .stride = t.stride,
        .es     = t.es,
    };

    Mat to = {
        .rows   = t.rows,
        .cols   = 1,
        .stride = t.stride,
        .es     = t.es+2,
    };

    for (size_t i = 0; i < 2; i++) {
        for (size_t j = 0; j < 2; j++) {
            MAT_AT(ti, i*2+j, 0) = i;
            MAT_AT(ti, i*2+j, 1) = j;
            MAT_AT(to, i*2+j, 0) = i^j;
        }
    }

    const char* out_file = "xor.mat";
    FILE* out = fopen(out_file, "wb");
    if (out == NULL) {
        fprintf(stderr, "ERROR: Couldn't write to file: %s", out_file);
        return 1;
    }

    mat_save(t, out);
    fclose(out);
    printf("Generated file: %s", out_file);
    return 0;
}
