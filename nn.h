#ifndef NN_H_
#define NN_H_

#define MAT_AT(mat, i, j) (mat).data[(i) * (mat).cols + (j)]
#define MAT_PRINT(mat) mat_print(mat, #mat)

#include <stddef.h>
#include <stdio.h>

#ifndef NN_ASSERT
#include <assert.h>
#define NN_ASSERT assert
#endif // !NN_ASSERT

#ifndef NN_MALLOC
#include <stdlib.h>
#define NN_MALLOC malloc
#endif // !NN_MALLOC

typedef struct {
    size_t rows;
    size_t cols;
    float* data;
} Mat;

float rand_float();

Mat mat_alloc(size_t rows, size_t cols);
void mat_rand(Mat mat, float min, float max);
void mat_fill(Mat mat, float x);
void mat_dot(Mat des, Mat a, Mat b);
void mat_sum(Mat des, Mat a);
void mat_print(Mat mat, const char* name);

#endif // !NN_H_

#ifdef NN_IMPLEMENTATION
float rand_float() {
    return (float)rand() / (float)RAND_MAX;
}

Mat mat_alloc(size_t rows, size_t cols) {
    Mat m;
    m.rows = rows,
    m.cols = cols,
    m.data = NN_MALLOC(sizeof(*m.data)*rows*cols);

    NN_ASSERT(m.data != NULL);
    return m;
}

void mat_rand(Mat mat, float min, float max) {
    for (size_t i = 0; i < mat.rows; i++) {
        for (size_t j = 0; j < mat.cols; j++) {
            MAT_AT(mat, i, j) = rand_float() * (max - min) + min;
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

void mat_print(Mat mat, const char* name) {
    printf("%s: [\n", name);
    for (size_t i = 0; i < mat.rows; i++) {
        for (size_t j = 0; j < mat.cols; j++) {
            printf("    %f", MAT_AT(mat, i, j));
        }
        printf("\n");
    }
    printf("]\n");
}
#endif // NN_IMPLEMENTATION
