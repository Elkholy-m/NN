#include <stdio.h>
#include <time.h>
#include <assert.h>

#define NN_IMPLEMENTATION
#include "nn.h"

typedef struct {
    Mat a0;
    Mat w1, b1, a1;
    Mat w2, b2, a2;
} Xor;

Xor xor_init() {
    Xor m;
    m.a0 = mat_alloc(1, 2);
    m.w1 = mat_alloc(2, 2);
    m.b1 = mat_alloc(1, 2);
    m.a1 = mat_alloc(1, 2);
    m.w2 = mat_alloc(2, 1);
    m.b2 = mat_alloc(1, 1);
    m.a2 = mat_alloc(1, 1);

    mat_rand(m.w1, 0, 1);
    mat_rand(m.b1, 0, 1);
    mat_rand(m.w2, 0, 1);
    mat_rand(m.b2, 0, 1);

    return m;
}

void xor_forward(Xor m) {

    mat_dot(m.a1, m.a0, m.w1);
    mat_sum(m.a1,  m.b1);
    mat_sig(m.a1);

    mat_dot(m.a2, m.a1, m.w2);
    mat_sum(m.a2, m.b2);
    mat_sig(m.a2);
}

float xor_cost(Xor m, Mat ti, Mat to) {
    assert(ti.rows == to.rows);
    assert(to.cols == m.a2.cols);

    size_t n = ti.rows;
    size_t q = to.cols;
    float err = 0;

    for (size_t i = 0; i < n; i++) {
        Mat x = mat_row(ti, i);
        Mat y = mat_row(to, i);
        mat_copy(m.a0, x);
        xor_forward(m);
        for (size_t j = 0; j < q; j++) {
            float diff =  MAT_AT(m.a2, 0, j) - MAT_AT(y, 0, j);
            err += diff*diff;
        }
    }

    return err/n;
}

void xor_finite_diff(Xor m, Xor g, Mat ti, Mat to, float eps) {
    float cost = xor_cost(m, ti, to);

    for (size_t i = 0; i < m.w1.rows; i++) {
        for (size_t j = 0; j < m.w1.cols; j++) {
            float saved = MAT_AT(m.w1, i, j);
            MAT_AT(m.w1, i, j) += eps;
            MAT_AT(g.w1, i, j) = (xor_cost(m, ti, to) - cost) / eps;
            MAT_AT(m.w1, i, j) = saved;
        }
    }

    for (size_t i = 0; i < m.w2.rows; i++) {
        for (size_t j = 0; j < m.w2.cols; j++) {
            float saved = MAT_AT(m.w2, i, j);
            MAT_AT(m.w2, i, j) += eps;
            MAT_AT(g.w2, i, j) = (xor_cost(m, ti, to) - cost) / eps;
            MAT_AT(m.w2, i, j) = saved;
        }
    }

    for (size_t i = 0; i < m.b1.rows; i++) {
        for (size_t j = 0; j < m.b1.cols; j++) {
            float saved = MAT_AT(m.b1, i, j);
            MAT_AT(m.b1, i, j) += eps;
            MAT_AT(g.b1, i, j) = (xor_cost(m, ti, to) - cost) / eps;
            MAT_AT(m.b1, i, j) = saved;
        }
    }

    for (size_t i = 0; i < m.b2.rows; i++) {
        for (size_t j = 0; j < m.b2.cols; j++) {
            float saved = MAT_AT(m.b2, i, j);
            MAT_AT(m.b2, i, j) += eps;
            MAT_AT(g.b2, i, j) = (xor_cost(m, ti, to) - cost) / eps;
            MAT_AT(m.b2, i, j) = saved;
        }
    }
}

void xor_learn(Xor m, Xor g, float rate) {
    for (size_t i = 0; i < m.w1.rows; i++) {
        for (size_t j = 0; j < m.w1.cols; j++) {
            MAT_AT(m.w1, i, j) -= rate*MAT_AT(g.w1, i, j);
        }
    }

    for (size_t i = 0; i < m.w2.rows; i++) {
        for (size_t j = 0; j < m.w2.cols; j++) {
            MAT_AT(m.w2, i, j) -= rate*MAT_AT(g.w2, i, j);
        }
    }

    for (size_t i = 0; i < m.b1.rows; i++) {
        for (size_t j = 0; j < m.b1.cols; j++) {
            MAT_AT(m.b1, i, j) -= rate*MAT_AT(g.b1, i, j);
        }
    }

    for (size_t i = 0; i < m.b2.rows; i++) {
        for (size_t j = 0; j < m.b2.cols; j++) {
            MAT_AT(m.b2, i, j) -= rate*MAT_AT(g.b2, i, j);
        }
    }
}

void xor_truth_taple(Xor m) {
    for (size_t i = 0; i < 2; i++) {
        for (size_t j = 0; j < 2; j++) {
            MAT_AT(m.a0, 0, 0) = i;
            MAT_AT(m.a0, 0, 1) = j;
            xor_forward(m);
            float y = *m.a2.es;
            printf("%zu ^ %zu = %f\n", i, j , y);
        }
    }
}

float td[] = {
    0, 0, 0,
    0, 1, 1,
    1, 0, 1,
    1, 1, 0,
};

int main() {
    srand(time(0));

    float eps  = 1e-1;
    float rate = 1e-1;

    Xor m = xor_init();
    Mat ti = {
        .rows = 4,
        .cols = 2,
        .stride = 3,
        .es = td,
    };

    Mat to = {
        .rows = 4,
        .cols = 1,
        .stride = 3,
        .es = td + 2,
    };

    Xor g = xor_init();
    for (size_t i = 0; i < 100*1000; i++) {
        xor_finite_diff(m, g, ti, to, eps);
        xor_learn(m, g, rate);
    }
    printf("Cost = %f\n", xor_cost(m, ti, to));
    xor_truth_taple(m);
    return 0;
}
