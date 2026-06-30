#ifndef NN_H_
#define NN_H_
#define ARRAY_LEN(arr) (sizeof(arr)/sizeof(arr[0]))
#define MAT_AT(mat, i, j) (mat).es[(i) * (mat).stride + (j)]
#define MAT_PRINT(mat) mat_print(mat, #mat, 0)

/* #define TRADITIONAL_APPROACH */
#ifndef NN_ACT
#define NN_ACT ACT_SIG
#endif //NN_ACT

#ifndef NN_RELU_PARAM
#define NN_RELU_PARAM 0.005f
#endif //NN_RELU_PARAM

#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <float.h>
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
float reluf(float x);

#define COORDINATE(i, j) (Coord){.x = (i), .y = (j)}
typedef enum {
    ACT_SIG,
    ACT_RELU,
    ACT_TANH,
    ACT_SIN,
} Act;

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
void mat_shuffle(Mat mat);
void mat_save(Mat mat, FILE* out);
Mat mat_load(FILE* in);
void mat_dot(Mat des, Mat a, Mat b);
void mat_sum(Mat des, Mat a);
Mat mat_row(Mat mat, size_t row);
Mat mat_sub(Mat mat, Coord start, Coord end);
void mat_copy(Mat dest, Mat src);
void mat_rand(Mat mat, float min, float max);
void mat_fill(Mat mat, float x);
void mat_act(Mat mat);
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
void nn_rand(NN nn, float low, float high);
void nn_forward(NN nn);
float nn_cost(NN nn, Mat ti, Mat to);
void nn_finite_diff(NN nn, NN g, Mat ti, Mat to, float eps);
void nn_backprop(NN nn, NN g, Mat ti, Mat to);
void nn_learn(NN nn, NN g, float rate);
void nn_zero(NN nn);

#ifdef NN_ENABLE_GYM

#ifndef WINDOW_FACTOR
#define WINDOW_FACTOR 80
#endif

#ifndef WINDOW_WIDTH
#define WINDOW_WIDTH  (16*WINDOW_FACTOR)
#endif

#ifndef WINDOW_HEIGHT
#define WINDOW_HEIGHT (9*WINDOW_FACTOR)
#endif

#include <raylib.h>
#include <raymath.h>

#define DA_INIT_CAPACITY 256
#define BORDER_VPAD 50
#define BORDER_HPAD 50

#define DA_APPEND(da, item)                                                            \
    do {                                                                               \
        if ((da)->count >= (da)->capacity) {                                           \
            (da)->capacity = (da)->capacity == 0? DA_INIT_CAPACITY : (da)->capacity*2; \
            (da)->items = realloc((da)->items, sizeof(*(da)->items)*(da)->capacity);   \
            assert((da)->items != NULL && "Fail to realloc the dynamic array");        \
        }                                                                              \
        (da)->items[(da)->count++] = item;                                             \
    } while(0)

typedef struct {
    float* items;
    size_t capacity;
    size_t count;
} Costs;

void gym_nn_render(NN nn, int x, int y, int w, int h);
void gym_cost_render(Costs costs, int x, int y, int w, int h);
#endif // !NN_ENABLE_GYM_


#endif // !NN_H_


#ifdef NN_IMPLEMENTATION
float rand_float() {
    return (float)rand() / (float)RAND_MAX;
}

float sigmoidf(float x) {
    return 1.f / (1.f + expf(-x));
}

float reluf(float x) {
    return (x > 0) ? x : x*NN_RELU_PARAM;
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

void mat_shuffle(Mat mat)
{
    for (size_t i = 0; i < mat.rows; i++) {
        size_t swapidx = i + rand() % (mat.rows - i);
        if (i != swapidx) {
            for (size_t j = 0; j < mat.cols; j++) {
                float tmp = MAT_AT(mat, i, j);
                MAT_AT(mat, i, j) = MAT_AT(mat, swapidx, j);
                MAT_AT(mat, swapidx, j) = tmp;
            }    
        }
        
    }
}

void mat_save(Mat mat, FILE* out)
{
    const char* magic = "nn.h.mat";
    fwrite(magic, strlen(magic), 1, out);
    fwrite(&mat.rows, sizeof(mat.rows), 1, out);
    fwrite(&mat.cols, sizeof(mat.cols), 1, out);

    for (size_t i = 0; i < mat.rows; ++i) {
        size_t n = fwrite(&MAT_AT(mat, i, 0), sizeof(*mat.es), mat.cols, out);
        while (n < mat.cols && !ferror(out)) {
            size_t k = fwrite(&MAT_AT(mat, i+n, 0), sizeof(*mat.es), mat.cols-n, out);
            n += k;
        }
    }
}

Mat mat_load(FILE* in)
{
    uint64_t magic;
    size_t rows, cols;

    // THIS SHOULD WARN US IF THE SYSTEM IS LITTLE OR BIG ENDIAN
    // THE (void)! JUST FOR SILENT COMPILER WARNING
    (void)!fread(&magic , sizeof(magic), 1, in);
    NN_ASSERT(magic == 0x74616d2e682e6e6e);

    (void)!fread(&rows  , sizeof(rows) , 1, in);
    (void)!fread(&cols  , sizeof(cols) , 1, in);

    Mat mat = mat_alloc(rows, cols);
    size_t n = fread(mat.es, sizeof(*mat.es), rows*cols, in);
    while (n < rows*cols && !ferror(in)) {
        size_t k = fread(mat.es+n, sizeof(*mat.es), rows*cols-n, in);
        n += k;
    }

    return mat;
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
            MAT_AT(mat, i, j) = (rand_float() * (high - low)) + low;
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

void mat_act(Mat mat) {
    for (size_t i = 0; i < mat.rows; i++) {
        for (size_t j = 0; j < mat.cols; j++) {
            switch(NN_ACT) {
            case ACT_SIG:
                MAT_AT(mat, i, j) = sigmoidf(MAT_AT(mat, i, j));
                break;
            case ACT_RELU:
                MAT_AT(mat, i, j) = reluf(MAT_AT(mat, i, j));
                break;
            case ACT_TANH:
                MAT_AT(mat, i, j) = tanhf(MAT_AT(mat, i, j));
                break;
            case ACT_SIN:
                MAT_AT(mat, i, j) = sinf(MAT_AT(mat, i, j));
                break;
            default:
                NN_ASSERT(0 && "UNREACHABLE");
                break;
            }
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

void nn_rand(NN nn, float low, float high) {
    for (size_t i = 0; i < nn.count; i++) {
        mat_rand(nn.ws[i], low, high);
        mat_rand(nn.bs[i], low, high);
    }
}

void nn_forward(NN nn) {
    for (size_t i = 0; i < nn.count; i++) {
        mat_dot(nn.as[i + 1], nn.as[i], nn.ws[i]);
        mat_sum(nn.as[i + 1], nn.bs[i]);
        mat_act(nn.as[i + 1]);
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

void nn_finite_diff(NN nn, NN g, Mat ti, Mat to, float eps) {
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

void nn_backprop(NN nn, NN g, Mat ti, Mat to) {
    NN_ASSERT(nn.count == g.count);
    NN_ASSERT(ti.rows == to.rows);
    NN_ASSERT(NN_INPUT(nn).cols == ti.cols);
    NN_ASSERT(NN_OUTPUT(nn).cols == to.cols);

    // I -> CURRENT SAMPLE
    // L -> CURRENT LAYER
    // J -> CURRENT ACTIVATION
    // K -> PREVIOUS ACTIVATION

    nn_zero(g);
    size_t n = ti.rows;
    for (size_t i = 0; i < n; i++) {
        // THIS  IS THE FEED FORWARD OPERATION
        mat_copy(NN_INPUT(nn), mat_row(ti, i));
        nn_forward(nn);

        for (size_t j = 0; j <= nn.count; j++) {
            mat_fill(g.as[j], 0);
        }

        for (size_t j = 0; j < to.cols; j++) {
#ifdef TRADITIONAL_APPROACH
            MAT_AT(NN_OUTPUT(g), 0, j) = 2*MAT_AT(NN_OUTPUT(nn), 0, j) - MAT_AT(to, i, j);
#else
            MAT_AT(NN_OUTPUT(g), 0, j) = MAT_AT(NN_OUTPUT(nn), 0, j) - MAT_AT(to, i, j);
#endif //TRADITIONAL_APPROACH
        }

        // THIS  IS THE BACK PROPAGATION OPERATION
        for (size_t l = nn.count; l > 0; l--) {
            for (size_t j = 0; j < nn.as[l].cols; j++) {
                float a = MAT_AT(nn.as[l], 0, j);
                float da = MAT_AT(g.as[l], 0, j);
                float q;
                switch(NN_ACT) {
                case ACT_SIG:
                    q = a*(1 - a);
                    break;
                case ACT_RELU:
                    q = (a >= 0) ? 1 : NN_RELU_PARAM;
                    break;
                case ACT_TANH:
                    q = 1 - a*a;
                    break;
                case ACT_SIN:
                    q = cosf(asinf(a));
                    break;
                default:
                    NN_ASSERT(0 && "UNREACHABLE");
                    break;
                };

#ifdef TRADITIONAL_APPROACH
                float s = 1;
#else
                float s = 2;
#endif //TRADITIONAL_APPROACH

                MAT_AT(g.bs[l-1], 0, j) += s*da*q;
                for (size_t k = 0; k < nn.as[l-1].cols; k++) {
                    float pa = MAT_AT(nn.as[l-1], 0, k);
                    float w  = MAT_AT(nn.ws[l-1], k, j);
                    MAT_AT(g.ws[l-1], k, j) += s*da*q*pa;
                    MAT_AT(g.as[l-1], 0, k) += s*da*q*w;
                }
            }
        }
    }

    for (size_t i = 0; i < g.count; i++) {
        for (size_t j = 0; j < g.ws[i].rows; j++) {
            for (size_t k = 0; k < g.ws[i].cols; k++) {
                MAT_AT(g.ws[i], j, k) /= n;
            }
        }

        for (size_t j = 0; j < g.bs[i].rows; j++) {
            for (size_t k = 0; k < g.bs[i].cols; k++) {
                MAT_AT(g.bs[i], j, k) /= n;
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

        for (size_t j = 0; j < nn.bs[i].rows; j++) {
            for (size_t k = 0; k < nn.bs[i].cols; k++) {
                MAT_AT(nn.bs[i], j, k) -= rate*MAT_AT(g.bs[i], j, k);
            }
        }
    }
}

void nn_zero(NN nn) {
    for (size_t i = 0; i < nn.count; i++) {
        mat_fill(nn.ws[i], 0);
        mat_fill(nn.bs[i], 0);
        mat_fill(nn.as[i], 0);
    }
    mat_fill(nn.as[nn.count], 0);
}

#ifdef NN_ENABLE_GYM
void gym_nn_render(NN nn, int x, int y, int w, int h)
{
    unsigned int neuron_raduis = h*((float)25/WINDOW_HEIGHT);
    Color low__color       = MAROON;
    Color high_color       = DARKGREEN;

    size_t nn_x = w - 2*BORDER_HPAD;
    size_t nn_y = h - 2*BORDER_VPAD;

    size_t no_of_layers = nn.count + 1;
    int hpad = nn_x/no_of_layers;
    
    for (size_t i = 0; i < no_of_layers; i++) {
        int vpad_1 = nn_y/nn.as[i].cols;
        for (size_t j = 0; j < nn.as[i].cols; j++) {
            int cx1 = x + BORDER_HPAD + i*hpad + hpad/2;
            int cy1 = y +BORDER_VPAD + j*vpad_1 + vpad_1/2;
            if (i < (no_of_layers - 1)) {
                int vpad_2 = nn_y/nn.as[i+1].cols;
                for (size_t k = 0; k < nn.as[i+1].cols; k++) {
                    int cx2 = x + BORDER_HPAD + (i+1)*hpad + hpad/2;
                    int cy2 = y +BORDER_VPAD + k*vpad_2 + vpad_2/2;
                    
                    float alpha = sigmoidf(MAT_AT(nn.ws[i], j, k));;
                    float thick = h*(3.0f/WINDOW_HEIGHT);
                    Color connection_color = ColorAlphaBlend(low__color, high_color, RAYWHITE);
                    high_color.a = floorf(255.f*alpha);
                    DrawLineEx((Vector2){cx1, cy1}, (Vector2){cx2, cy2}, thick, connection_color);
                }
            }
            if (i == 0) {
                DrawCircle(cx1, cy1, neuron_raduis, GRAY);
            } else {
                high_color.a = floorf(255.f*sigmoidf(MAT_AT(nn.bs[i-1], 0, j)));
                Color neuron_color = ColorAlphaBlend(low__color, high_color, RAYWHITE);
                DrawCircle(cx1, cy1, neuron_raduis, neuron_color);
            }
        }
    }
}

void gym_cost_render(Costs costs, int x, int y, int w, int h)
{
    char buffer[256];
    float font_size = h*(50.0f/WINDOW_HEIGHT);

    snprintf(buffer, sizeof(buffer), "COST-FUNCTION");
    int tw =  MeasureText(buffer, font_size);
    DrawText(buffer, x+(w/2-tw/2), y+h, font_size, RAYWHITE);

    w -= BORDER_HPAD;
    h -= BORDER_VPAD;

    float min = FLT_MAX, max = FLT_MIN;

    for (size_t i = 0; i < costs.count; i++) {
        if (costs.items[i] < min) min = costs.items[i];
        if (costs.items[i] > max) max = costs.items[i];
    }

    if (min > 0) min = 0;
    size_t n = costs.count;
    if (n < 1000) n = 1000;

    // DRAW THE AXIS LINES
    int thick = h*(5.0f/WINDOW_HEIGHT);
    int lx = x+BORDER_HPAD/2-thick;
    int ly = y+BORDER_HPAD/2-thick;
    float tri_val = (h*10.f/WINDOW_HEIGHT);

    // X-AXIS 
    DrawLineEx((Vector2) {lx, ly+h}, (Vector2) {lx+w, ly+h}, thick, RAYWHITE);
    DrawTriangle((Vector2) {lx+w, ly+h-tri_val}, (Vector2) {lx+w, ly+h+tri_val}, (Vector2) {lx+w+tri_val, ly+h}, RAYWHITE);

    // Y-AXIS
    DrawLineEx((Vector2) {lx, ly}, (Vector2) {lx, ly+h}, thick, RAYWHITE);
    DrawTriangle((Vector2) {lx-tri_val, ly}, (Vector2) {lx+tri_val, ly}, (Vector2) {lx, ly-tri_val}, RAYWHITE);

    snprintf(buffer, sizeof(buffer), "0");
    DrawText(buffer, lx+5, ly+h+5, font_size, RAYWHITE);


    for (size_t i = 0; (i+1) < costs.count; i++) {
        int x1 = x + BORDER_HPAD/2 + (float)w/n*i;
        int y1 = y + BORDER_VPAD/2 + (1 - (costs.items[i]-min) / (max-min))*h;

        int x2 = x + BORDER_HPAD/2 + (float)w/n*(i+1);
        int y2 = y + BORDER_VPAD/2 + (1 - (costs.items[i+1]-min) / (max-min))*h;

        float thick = h*(5.0f/WINDOW_HEIGHT);
        DrawLineEx((Vector2) {x1, y1}, (Vector2) {x2, y2}, thick, MAROON);
    }
}

#endif // !NN_ENABLE_GYM_

#endif // NN_IMPLEMENTATION
