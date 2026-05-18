#ifndef NN_H_
#define NN_H_

#define MAT_AT(mat, i, j) (mat).es[(i) * (mat).stride + (j)]
#define MAT_PRINT(mat) mat_print(mat, #mat)

#include <stddef.h>
#include <stdio.h>
#include <math.h>

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
    size_t stride;
    float* es;
} Mat;

float rand_float();
float sigmoidf(float x);

Mat mat_alloc(size_t rows, size_t cols);
void mat_dot(Mat des, Mat a, Mat b);
void mat_sum(Mat des, Mat a);
Mat mat_row(Mat mat, size_t row);
void mat_copy(Mat dest, Mat src);
void mat_rand(Mat mat, float min, float max);
void mat_fill(Mat mat, float x);
void mat_sig(Mat mat);
void mat_print(Mat mat, const char* name);

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

void mat_copy(Mat dest, Mat src) {
    NN_ASSERT(dest.rows == src.rows);
    NN_ASSERT(dest.cols == src.cols);

    for (size_t i = 0; i < dest.rows; i++) {
        for (size_t j = 0; j < dest.cols; j++) {
            MAT_AT(dest, i, j) = MAT_AT(src, i, j);
        }
    }
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

void mat_sig(Mat mat) {
    for (size_t i = 0; i < mat.rows; i++) {
        for (size_t j = 0; j < mat.cols; j++) {
            MAT_AT(mat, i, j) = sigmoidf(MAT_AT(mat, i, j));
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
