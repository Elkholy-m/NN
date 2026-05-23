#ifndef NN_H_
#define NN_H_
#define ARRAY_LEN(arr) (sizeof(arr)/sizeof(arr[0]))
#define MAT_AT(mat, i, j) (mat).es[(i) * (mat).stride + (j)]
#define MAT_PRINT(mat) mat_print(mat, #mat, 0)

#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <sys/param.h>

#ifndef NN_ASSERT
#include <assert.h>
#define NN_ASSERT assert
#endif // !NN_ASSERT

#ifndef NN_MALLOC
#include <stdlib.h>
#define NN_MALLOC malloc
#endif // !NN_MALLOC

float rand_float();
float sigmoidf(float x);

#define COORDINATE(i, j) (Coord){.x = (i), .y = (j)}
typedef struct {
    size_t x;
    size_t y;
} Coord;

typedef struct {
    size_t rows;
    size_t cols;
    size_t stride;
    float* es;
} Mat;

Mat mat_alloc(size_t rows, size_t cols);
void mat_dot(Mat des, Mat a, Mat b);
void mat_sum(Mat des, Mat a);
Mat mat_row(Mat mat, size_t row);
Mat mat_sub(Mat mat, Coord start, Coord end);
void mat_copy(Mat dest, Mat src);
void mat_rand(Mat mat, float min, float max);
void mat_fill(Mat mat, float x);
void mat_sig(Mat mat);
void mat_print(Mat mat, const char* name, size_t padding);

#define NN_INPUT(nn)  (nn).as[0]
#define NN_OUTPUT(nn) (nn).as[(nn).count]
#define NN_PRINT(nn) nn_print(nn, #nn)

typedef struct {
    size_t count;
    Mat* ws;
    Mat* bs;
    Mat* as; // activations => count + 1
} NN;

NN nn_alloc(size_t* arch, size_t arch_count);
void nn_print(NN nn, const char* name);
void nn_rand(NN nn, size_t low, size_t high);
void nn_forward(NN nn);
float nn_cost(NN nn, Mat ti, Mat to);
void nn_gredient(NN nn, NN g, Mat ti, Mat to, float eps);
void nn_learn(NN nn, NN g, float rate);

#endif // !NN_H_

#ifdef NN_IMPLEMENTATION
float rand_float() {
    return (float)rand() / (float)RAND_MAX;
}

float sigmoidf(float x) {
    return 1.f / (1.f + expf(-x));
}

Mat mat_alloc(size_t rows, size_t cols) {
    Mat m;
    m.rows = rows,
    m.cols = cols,
    m.stride = cols,
    m.es = NN_MALLOC(sizeof(*m.es)*rows*cols);

    NN_ASSERT(m.es != NULL);
    return m;
}

void mat_dot(Mat des, Mat a, Mat b) {
    NN_ASSERT(a.cols == b.rows);
    NN_ASSERT(des.rows == a.rows);
    NN_ASSERT(des.cols == b.cols);

    size_t n = a.cols;
    for (size_t i = 0; i < des.rows; i++) {
        for (size_t j = 0; j < des.cols; j++) {
            MAT_AT(des, i, j) = 0;
            for (size_t k = 0; k < n; k++) {
                MAT_AT(des, i, j) += MAT_AT(a, i, k) * MAT_AT(b, k, j);
            }
        }
    }
}

void mat_sum(Mat des, Mat a) {
    NN_ASSERT(des.rows == a.rows);
    NN_ASSERT(des.cols == a.cols);

    for (size_t i = 0; i < des.rows; i++) {
        for (size_t j = 0; j < des.cols; j++) {
            MAT_AT(des, i, j) += MAT_AT(a, i, j);
        }
    }
}

Mat mat_row(Mat mat, size_t row) {
    return (Mat) {
        .rows = 1,
        .cols = mat.cols,
        .stride = mat.stride,
        .es = &MAT_AT(mat, row, 0)
    };
}

Mat mat_sub(Mat mat, Coord start, Coord end) {
    // ASSERTIONS
    NN_ASSERT(start.x < mat.rows);
    NN_ASSERT(end.x   < mat.rows);
    NN_ASSERT(start.y < mat.cols);
    NN_ASSERT(end.y   < mat.cols);

    return (Mat) {
        .rows = abs((int)(end.x - start.x)) + 1,
        .cols = abs((int)(end.y - start.y)) + 1,
        .stride = mat.stride,
        .es = &(MAT_AT(mat, (size_t)MIN(end.x, start.x), (size_t)MIN(end.y, start.y))),
    };
}

void mat_copy(Mat dest, Mat src) {
    NN_ASSERT(dest.rows == src.rows);
    NN_ASSERT(dest.cols == src.cols);

    for (size_t i = 0; i < dest.rows; i++) {
        for (size_t j = 0; j < dest.cols; j++) {
            MAT_AT(dest, i, j) = MAT_AT(src, i, j);
        }
    }
}

void mat_rand(Mat mat, float low, float high) {
    for (size_t i = 0; i < mat.rows; i++) {
        for (size_t j = 0; j < mat.cols; j++) {
            MAT_AT(mat, i, j) = rand_float() * (high - low) + low;
        }
    }
}

void mat_fill(Mat mat, float x) {
    for (size_t i = 0; i < mat.rows; i++) {
        for (size_t j = 0; j < mat.cols; j++) {
            MAT_AT(mat, i, j) = x;
        }
    }
}

void mat_sig(Mat mat) {
    for (size_t i = 0; i < mat.rows; i++) {
        for (size_t j = 0; j < mat.cols; j++) {
            MAT_AT(mat, i, j) = sigmoidf(MAT_AT(mat, i, j));
        }
    }
}

void mat_print(Mat mat, const char* name, size_t padding) {
    printf("%*s%s: [\n", (int) padding, "", name);
    for (size_t i = 0; i < mat.rows; i++) {
        for (size_t j = 0; j < mat.cols; j++) {
            printf("%*s", (int) padding, "");
            printf("    %f", MAT_AT(mat, i, j));
        }
        printf("\n");
    }
    printf("%*s]\n", (int) padding, "");
}


NN nn_alloc(size_t* arch, size_t arch_count) {
    NN_ASSERT(arch_count > 0);

    NN nn;
    nn.count = arch_count - 1;
    nn.ws = NN_MALLOC(sizeof(*nn.ws) * nn.count);
    nn.bs = NN_MALLOC(sizeof(*nn.bs) * nn.count);
    nn.as = NN_MALLOC(sizeof(*nn.bs) * arch_count);
    
    NN_INPUT(nn) = mat_alloc(1, arch[0]);
    for (size_t i = 1; i < arch_count; i++) {
        nn.as[i]     = mat_alloc(1, arch[i]);
        nn.ws[i - 1] = mat_alloc(arch[i - 1], arch[i]);
        nn.bs[i - 1] = mat_alloc(1, arch[i]);
    }

    return nn;
}

void nn_print(NN nn, const char* name) {
    char buff[256];
    printf("%s: [\n", name);
    for (size_t i = 0; i < nn.count; i++) {
        snprintf(buff, sizeof(buff), "w%zu::[%zu]x[%zu]", i, nn.ws[i].rows, nn.ws[i].cols);
        mat_print(nn.ws[i], buff, 4);
        snprintf(buff, sizeof(buff), "b%zu::[%zu]x[%zu]", i, nn.bs[i].rows, nn.bs[i].cols);
        mat_print(nn.bs[i], buff, 4);
    }
    printf("]\n");
}

void nn_rand(NN nn, size_t low, size_t high) {
    for (size_t i = 0; i < nn.count; i++) {
        mat_rand(nn.ws[i], low, high);
        mat_rand(nn.bs[i], low, high);
    }
}

void nn_forward(NN nn) {
    for (size_t i = 0; i < nn.count; i++) {
        mat_dot(nn.as[i + 1], nn.as[i], nn.ws[i]);
        mat_sum(nn.as[i + 1], nn.bs[i]);
        mat_sig(nn.as[i + 1]);
    }
}

float nn_cost(NN nn, Mat ti, Mat to) {
    NN_ASSERT(ti.rows == to.rows);
    NN_ASSERT(ti.cols == NN_INPUT(nn).cols);
    NN_ASSERT(to.cols == NN_OUTPUT(nn).cols);

    float c = 0;
    size_t n = ti.rows;

    for (size_t i = 0; i < n; i++) {
        Mat x = mat_row(ti, i);
        Mat y = mat_row(to, i);
        mat_copy(NN_INPUT(nn), x);
        nn_forward(nn);

        size_t q = to.cols;
        for (size_t j = 0; j < q; j++) {
            float diff = MAT_AT(y, 0, j)  - MAT_AT(NN_OUTPUT(nn), 0, j);
            c += diff*diff;
        }
    }

    return c/n;
}

void nn_gredient(NN nn, NN g, Mat ti, Mat to, float eps) {
    NN_ASSERT(nn.count == g.count);
    NN_ASSERT(ti.rows == to.rows);
    NN_ASSERT(to.cols == NN_OUTPUT(nn).cols);

    float saved;
    float c = nn_cost(nn, ti, to);
    for (size_t i = 0; i < nn.count; i++) {
        for (size_t j = 0; j < nn.ws[i].rows; j++) {
            for (size_t k = 0; k < nn.ws[i].cols; k++) {
                NN_ASSERT(nn.ws[i].rows == g.ws[i].rows);
                NN_ASSERT(nn.ws[i].cols == g.ws[i].cols);

                saved = MAT_AT(nn.ws[i], j, k);
                MAT_AT(nn.ws[i], j, k) += eps;
                MAT_AT(g.ws[i], j, k)   = (nn_cost(nn, ti, to) - c)/eps;
                MAT_AT(nn.ws[i], j, k)  = saved;
            }
        }
    }

    for (size_t i = 0; i < nn.count; i++) {
        for (size_t j = 0; j < nn.bs[i].rows; j++) {
            for (size_t k = 0; k < nn.bs[i].cols; k++) {
                NN_ASSERT(nn.bs[i].rows == g.bs[i].rows);
                NN_ASSERT(nn.bs[i].cols == g.bs[i].cols);

                saved = MAT_AT(nn.bs[i], j, k);
                MAT_AT(nn.bs[i], j, k) += eps;
                MAT_AT(g.bs[i], j, k)   = (nn_cost(nn, ti, to) - c)/eps;
                MAT_AT(nn.bs[i], j, k)  = saved;
            }
        }
    }
}

void nn_learn(NN nn, NN g, float rate) {
    for (size_t i = 0; i < nn.count; i++) {
        for (size_t j = 0; j < nn.ws[i].rows; j++) {
            for (size_t k = 0; k < nn.ws[i].cols; k++) {
                MAT_AT(nn.ws[i], j, k) -= rate*MAT_AT(g.ws[i], j, k);
            }
        }
    }

    for (size_t i = 0; i < nn.count; i++) {
        for (size_t j = 0; j < nn.bs[i].rows; j++) {
            for (size_t k = 0; k < nn.bs[i].cols; k++) {
                MAT_AT(nn.bs[i], j, k) -= rate*MAT_AT(g.bs[i], j, k);
            }
        }
    }
}
#endif // NN_IMPLEMENTATION
