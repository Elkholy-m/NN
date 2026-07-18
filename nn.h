#ifndef NN_H_
#define NN_H_

#ifndef NN_ACT
#define NN_ACT ACT_SIG
#endif //NN_ACT

#ifndef NN_RELU_PARAM
#define NN_RELU_PARAM 0.005f
#endif //NN_RELU_PARAM

#include <math.h>
#include <stddef.h>
#include <stdbool.h>
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

#define NN_INPUT(nn)  (nn).as[0]
#define NN_OUTPUT(nn) (nn).as[(nn).arch_count-1]
#define NN_PRINT(nn) nn_print(nn, #nn)
#define ARRAY_LEN(arr) (sizeof(arr)/sizeof(arr[0]))
#define MAT_AT(mat, i, j) (mat).es[(i) * (mat).stride + (j)]
#define MAT_PRINT(mat) mat_print(mat, #mat, 0)
#define COORDINATE(i, j) (Coord){.x = (i), .y = (j)}

float rand_float();
float sigmoidf(float x);
float reluf(float x);

typedef struct {
    size_t capacity;
    size_t size;
    char*  data;
} Region;

Region region_alloc_alloc(size_t capacity_bytes);
void* region_alloc(Region* r, size_t size_bytes);
#define region_reset(r) (NN_ASSERT(r != NULL), (r)->size = 0)

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

Mat mat_alloc(Region* r, size_t rows, size_t cols);
void mat_shuffle(Mat mat);
void mat_save(Mat mat, FILE* out);
Mat mat_load(Region* r, FILE* in);
void mat_dot(Mat des, Mat a, Mat b);
void mat_sum(Mat des, Mat a);
Mat mat_row(Mat mat, size_t row);
Mat mat_sub(Mat mat, Coord start, Coord end);
void mat_copy(Mat dest, Mat src);
void mat_rand(Mat mat, float min, float max);
void mat_fill(Mat mat, float x);
void mat_act(Mat mat);
void mat_print(Mat mat, const char* name, size_t padding);

typedef struct {
    size_t* arch;
    size_t arch_count;
    Mat* ws; // weights     => arch_count-1
    Mat* bs; // biases      => arch_count-1
    Mat* as; // activations => arch_count
} NN;

typedef struct { 
    size_t start;
    float cost;
    bool end;
} Batch;

NN nn_alloc(Region* r, size_t* arch, size_t arch_count);
void nn_print(NN nn, const char* name);
void nn_rand(NN nn, float low, float high);
void nn_forward(NN nn);
float nn_cost(NN nn, Mat ti, Mat to);
NN nn_finite_diff(Region* r, NN nn, Mat ti, Mat to, float eps);
NN nn_backprop(Region* r, NN nn, Mat ti, Mat to);
void nn_learn(NN nn, NN g, float rate);
void nn_zero(NN nn);
void nn_batch(Region* r, Batch* batch, NN nn, Mat t, float rate, size_t batch_size);

#ifdef NN_ENABLE_GYM

#ifndef low__color
#define low__color MAROON
#endif
#ifndef high_color
#define high_color DARKGREEN
#endif

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

#define BORDER_VPAD 50
#define BORDER_HPAD 50

#define DA_INIT_CAPACITY 256
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

typedef struct {
    size_t x;
    size_t y;
    size_t w;
    size_t h;
} LayoutRect;

typedef enum {
    LO_HORZ,
    LO_VERT,
} LayoutOrient;

typedef struct {
    LayoutRect rect;
    LayoutOrient orient;
    size_t i;
    size_t count;
    size_t gap;
} Layout;

typedef struct {
    Layout* items;
    size_t count;
    size_t capacity;
} LayoutStack;

typedef enum {
    WEIGHT,
    ACT,
} HeatmapKind;

#define layout_stack_slot(ls) layout_stack_slot_imp(ls, __FILE__, __LINE__)

LayoutRect rect_constructor(size_t x, size_t y, size_t w, size_t h);
Layout layout_constructor(LayoutRect rect, LayoutOrient orient, size_t count, size_t gap);
void layout_stack_push(LayoutStack* ls, LayoutRect rect, LayoutOrient orient, size_t count, size_t gap);
void layout_stack_pop(LayoutStack* ls);
LayoutRect layout_slot(Layout *l, const char* file, const int line);
LayoutRect layout_stack_slot_imp(LayoutStack* ls, const char* file, const int line);
void widget(LayoutRect rect, Color c);

void gym_nn_render(NN nn, LayoutRect r);
void gym_heatmap_render(NN nn, LayoutRect r, HeatmapKind kind);
void gym_cost_render(Costs costs, LayoutRect r);
void gym_status_line_render(int h, int rw, size_t epoch, size_t max_epoch, float rate, float cost, Region* r);
void gym_slider_render(Vector2 rect_corner, Vector2 rect_size, Vector2 slider_center, float slider_raduis, bool* slider_clicked, float* scroll);
#endif // !NN_ENABLE_GYM

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

Region region_alloc_alloc(size_t capacity_bytes)
{
    return (Region) {
        .capacity = capacity_bytes,
        .size = 0,
        .data = NN_MALLOC(capacity_bytes),
    };
}

void* region_alloc(Region* r, size_t size_bytes)
{
    if (r == NULL) return NN_MALLOC(size_bytes);

    // RUN TIME CHECK
    NN_ASSERT((r->size + size_bytes) <= r->capacity);
    // COMPILE TIME CHECK SO IF THE USER CHANGE NN_ASSERT() IT STILL HANDLE IT
    if ((r->size + size_bytes) > r->capacity) return NULL;

    void* result = &r->data[r->size];
    r->size += size_bytes;
    return result;
}

Mat mat_alloc(Region* r, size_t rows, size_t cols) {
    Mat m;
    m.rows = rows,
    m.cols = cols,
    m.stride = cols,
    m.es = region_alloc(r, sizeof(*m.es)*rows*cols);

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

Mat mat_load(Region* r, FILE* in)
{
    uint64_t magic;
    size_t rows, cols;

    // THIS SHOULD WARN US IF THE SYSTEM IS LITTLE OR BIG ENDIAN
    // THE (void)! JUST FOR SILENT COMPILER WARNING
    (void)!fread(&magic , sizeof(magic), 1, in);
    NN_ASSERT(magic == 0x74616d2e682e6e6e);

    (void)!fread(&rows  , sizeof(rows) , 1, in);
    (void)!fread(&cols  , sizeof(cols) , 1, in);

    Mat mat = mat_alloc(r, rows, cols);
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


NN nn_alloc(Region* r, size_t* arch, size_t arch_count) {
    NN_ASSERT(arch_count > 0);

    NN nn;
    nn.arch = arch;
    nn.arch_count = arch_count;
    nn.ws = region_alloc(r, sizeof(*nn.ws) * nn.arch_count-1);
    nn.bs = region_alloc(r, sizeof(*nn.bs) * nn.arch_count-1);
    nn.as = region_alloc(r, sizeof(*nn.bs) * nn.arch_count);
    
    NN_INPUT(nn) = mat_alloc(r, 1, arch[0]);
    for (size_t i = 1; i < arch_count; i++) {
        nn.as[i]     = mat_alloc(r, 1, arch[i]);
        nn.ws[i - 1] = mat_alloc(r, arch[i - 1], arch[i]);
        nn.bs[i - 1] = mat_alloc(r, 1, arch[i]);
    }

    return nn;
}

void nn_print(NN nn, const char* name) {
    char buff[256];
    printf("%s: [\n", name);
    for (size_t i = 0; i < nn.arch_count-1; i++) {
        snprintf(buff, sizeof(buff), "w%zu::[%zu]x[%zu]", i, nn.ws[i].rows, nn.ws[i].cols);
        mat_print(nn.ws[i], buff, 4);
        snprintf(buff, sizeof(buff), "b%zu::[%zu]x[%zu]", i, nn.bs[i].rows, nn.bs[i].cols);
        mat_print(nn.bs[i], buff, 4);
    }
    printf("]\n");
}

void nn_rand(NN nn, float low, float high) {
    for (size_t i = 0; i < nn.arch_count-1; i++) {
        mat_rand(nn.ws[i], low, high);
        mat_rand(nn.bs[i], low, high);
    }
}

void nn_forward(NN nn) {
    for (size_t i = 0; i < nn.arch_count-1; i++) {
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

NN nn_finite_diff(Region* r, NN nn, Mat ti, Mat to, float eps)
{
    // NN_ASSERT(nn.count == g.count);
    NN_ASSERT(ti.rows == to.rows);
    NN_ASSERT(to.cols == NN_OUTPUT(nn).cols);

    NN g = nn_alloc(r, nn.arch, nn.arch_count);

    float saved;
    float c = nn_cost(nn, ti, to);
    for (size_t i = 0; i < nn.arch_count-1; i++) {
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

    for (size_t i = 0; i < nn.arch_count-1; i++) {
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

    return g;
}

NN nn_backprop(Region* r, NN nn, Mat ti, Mat to)
{
    // NN_ASSERT(nn.count == g.count);
    NN_ASSERT(ti.rows == to.rows);
    NN_ASSERT(NN_INPUT(nn).cols == ti.cols);
    NN_ASSERT(NN_OUTPUT(nn).cols == to.cols);

    // I -> CURRENT SAMPLE
    // L -> CURRENT LAYER
    // J -> CURRENT ACTIVATION
    // K -> PREVIOUS ACTIVATION

    NN g = nn_alloc(r, nn.arch, nn.arch_count);

    nn_zero(g);
    size_t n = ti.rows;
    for (size_t i = 0; i < n; i++) {
        // THIS  IS THE FEED FORWARD OPERATION
        mat_copy(NN_INPUT(nn), mat_row(ti, i));
        nn_forward(nn);

        for (size_t j = 0; j <= nn.arch_count-1; j++) {
            mat_fill(g.as[j], 0);
        }

        for (size_t j = 0; j < to.cols; j++) {
#ifdef NN_TRADITIONAL_APPROACH
            MAT_AT(NN_OUTPUT(g), 0, j) = 2*MAT_AT(NN_OUTPUT(nn), 0, j) - MAT_AT(to, i, j);
#else
            MAT_AT(NN_OUTPUT(g), 0, j) = MAT_AT(NN_OUTPUT(nn), 0, j) - MAT_AT(to, i, j);
#endif //NN_TRADITIONAL_APPROACH
        }

        // THIS  IS THE BACK PROPAGATION OPERATION
        for (size_t l = nn.arch_count-1; l > 0; l--) {
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

#ifdef NN_TRADITIONAL_APPROACH
                float s = 1;
#else
                float s = 2;
#endif //NN_TRADITIONAL_APPROACH

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

    for (size_t i = 0; i < g.arch_count-1; i++) {
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

    return g;
}

void nn_learn(NN nn, NN g, float rate) {
    for (size_t i = 0; i < nn.arch_count-1; i++) {
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
    for (size_t i = 0; i < nn.arch_count-1; i++) {
        mat_fill(nn.ws[i], 0);
        mat_fill(nn.bs[i], 0);
        mat_fill(nn.as[i], 0);
    }
    mat_fill(nn.as[nn.arch_count-1], 0);
}

// batch size, t training data, rate, epoch, costs
void nn_batch(Region* r, Batch* batch, NN nn, Mat t, float rate, size_t batch_size)
{
    if (batch->end){
        batch->end = false;
        batch->start = 0;       
        batch->cost = 0.f;       
    }

    size_t size = batch_size;
    
    if ((batch->start+batch_size) >= t.rows) size = t.rows - batch->start;

    Mat batch_ti = {
        .rows = size,
        .cols = NN_INPUT(nn).cols,
        .stride = t.stride,
        .es = &MAT_AT(t, batch->start, 0),
    };

    Mat batch_to = {
        .rows = size,
        .cols = NN_OUTPUT(nn).cols,
        .stride = t.stride,
        .es = &MAT_AT(t, batch->start, batch_ti.cols),
    };


    NN g = nn_backprop(r, nn, batch_ti, batch_to);
    nn_learn(nn, g, rate);
    batch->cost += nn_cost(nn, batch_ti, batch_to);
    batch->start += batch_size;

    if ((batch->start) >= t.rows) {
        batch->cost /= (t.rows + batch_size - 1)/batch_size;
        batch->end = true;
    }
}

#ifdef NN_ENABLE_GYM

LayoutRect rect_constructor(size_t x, size_t y, size_t w, size_t h)
{
    return (LayoutRect) {.x = x, .y = y, .w = w, .h = h};
}

Layout layout_constructor(LayoutRect rect, LayoutOrient orient, size_t count, size_t gap)
{
    return (Layout) {.rect = rect, .orient = orient, .count = count, .gap = gap};
}

void layout_stack_push(LayoutStack* ls, LayoutRect rect, LayoutOrient orient, size_t count, size_t gap)
{
    Layout l = layout_constructor(rect, orient, count, gap);
    DA_APPEND(ls, l);
}

void layout_stack_pop(LayoutStack* ls)
{
    assert(ls->count > 0);
    ls->count--;
}

LayoutRect layout_slot(Layout *l, const char* file, const int line)
{
    if (l->i >= l->count) {
        fprintf(stderr, "%s:%d:\terror: widget overfloaw\n", file, line);
        exit(EXIT_FAILURE);
    }

    LayoutRect r = {0};
    switch(l->orient) {
    case LO_HORZ:
        r.w = l->rect.w/l->count;
        r.h = l->rect.h;
        r.x = l->rect.x+l->i*r.w;
        r.y = l->rect.y;

        if (l->i == 0) { // first
            r.w -= l->gap/2;
        } else if (l->i == l->count-1) { // last
            r.x += l->gap/2;
            r.w -= l->gap/2;
        } else { //middle
            r.x += l->gap/2;
            r.w -= l->gap;
        }
        break;
    case LO_VERT:
        r.w = l->rect.w;
        r.h = l->rect.h/l->count;
        r.x = l->rect.x;
        r.y = l->rect.y+l->i*r.h;

        if (l->i == 0) { // first
            r.h -= l->gap/2;
        } else if (l->i == l->count-1) { // last
            r.y += l->gap/2;
            r.h -= l->gap/2;
        } else { //middle
            r.y += l->gap/2;
            r.h -= l->gap;
        }
        break;
    default:
        assert(0 && "unreachable");
    }
    l->i++;

    return r;
}

LayoutRect layout_stack_slot_imp(LayoutStack* ls, const char* file, const int line)
{
    assert(ls->count > 0);
    return layout_slot(&ls->items[ls->count-1], file, line);
}



void widget(LayoutRect rect, Color c)
{
    Rectangle rec = {
        .x = rect.x,
        .y = rect.y,
        .width = rect.w,
        .height = rect.h,
    };

    if(CheckCollisionPointRec(GetMousePosition(), rec)) {
        c = ColorBrightness(c, 0.65);
    }

    DrawRectangleRec(rec,  c);
}

void gym_nn_render(NN nn, LayoutRect r)
{
    unsigned int neuron_raduis = r.h*((float)25/WINDOW_HEIGHT);

    size_t nn_x = r.w - 2*BORDER_HPAD;
    size_t nn_y = r.h - 2*BORDER_VPAD;

    size_t no_of_layers = nn.arch_count;
    int hpad = nn_x/no_of_layers;
    
    for (size_t i = 0; i < no_of_layers; i++) {
        int vpad_1 = nn_y/nn.as[i].cols;
        for (size_t j = 0; j < nn.as[i].cols; j++) {
            int cx1 = r.x + BORDER_HPAD + i*hpad + hpad/2;
            int cy1 = r.y +BORDER_VPAD + j*vpad_1 + vpad_1/2;
            if (i < (no_of_layers - 1)) {
                int vpad_2 = nn_y/nn.as[i+1].cols;
                for (size_t k = 0; k < nn.as[i+1].cols; k++) {
                    int cx2 = r.x + BORDER_HPAD + (i+1)*hpad + hpad/2;
                    int cy2 = r.y +BORDER_VPAD + k*vpad_2 + vpad_2/2;
                    
                    float thick = r.h*(5.0f/WINDOW_HEIGHT);
                    Color connection_color = ColorLerp(low__color, high_color, sigmoidf(MAT_AT(nn.ws[i], j, k)));
                    DrawLineEx((Vector2){cx1, cy1}, (Vector2){cx2, cy2}, thick, connection_color);
                }
            }
            if (i == 0) {
                DrawCircle(cx1, cy1, neuron_raduis, GRAY);
            } else {
                Color neuron_color = ColorLerp(low__color, high_color, sigmoidf(MAT_AT(nn.bs[i-1], 0, j)));
                DrawCircle(cx1, cy1, neuron_raduis, neuron_color);
            }
        }
    }
}

void gym_heatmap_render(NN nn, LayoutRect r, HeatmapKind kind)
{
    size_t max_cols = 0;
    size_t total_rows = 0;
    size_t gap = r.h*0.03;
    size_t no_of_spacing;
    switch (kind) {
    case WEIGHT:
        no_of_spacing = nn.arch_count - 2;
        for (size_t i = 0; i < nn.arch_count-1; ++i) {
            if (max_cols < nn.ws[i].cols) max_cols += nn.ws[i].cols;
            total_rows += nn.ws[i].rows;
        }
        break;
    case ACT:
        no_of_spacing = nn.arch_count-1;
        for (size_t i = 0; i < nn.arch_count; ++i) {
            if (max_cols < nn.as[i].cols) max_cols += nn.as[i].cols;
            total_rows += nn.as[i].rows;
        }
        break;
    default:
        NN_ASSERT(0 && "Unreachable");
    }

    
    char buffer[256];
    float font_size = r.h*(50.0f/WINDOW_HEIGHT);
    int label_height = r.h*0.02;
    int tw;

    int cell_width   = r.w/max_cols;
    int cell_height  = (r.h-(no_of_spacing*gap+label_height))/total_rows;
    size_t accumelator = 0;

    switch (kind) {
    case WEIGHT:
        for (size_t i = 0; i < nn.arch_count-1; i++) {
            for (size_t j = 0; j < nn.ws[i].rows; j++) {
                size_t no_of_cols = nn.ws[i].cols;
                size_t cntrx = r.w/2-no_of_cols*cell_width/2;
                for (size_t k = 0; k < nn.ws[i].cols; k++) {
                    Color connection_color = ColorLerp(low__color, high_color, sigmoidf(MAT_AT(nn.ws[i], j, k)));
                    DrawRectangle(r.x+k*cell_width+cntrx, r.y+accumelator*cell_height+i*gap, cell_width, cell_height, connection_color);
                }
                accumelator += 1;
            }
        }


        snprintf(buffer, sizeof(buffer), "WEIGHTS");
        tw =  MeasureText(buffer, font_size);
        DrawText(buffer, r.x+(r.w/2-tw/2), r.y+r.h, font_size, RAYWHITE);
        break;
    case ACT:
        for (size_t i = 0; i < nn.arch_count; i++) {
            for (size_t j = 0; j < nn.as[i].rows; j++) {
                size_t no_of_cols = nn.as[i].cols;
                size_t cntrx = r.w/2-no_of_cols*cell_width/2;
                for (size_t k = 0; k < nn.as[i].cols; k++) {
                    Color connection_color = ColorLerp(low__color, high_color, sigmoidf(MAT_AT(nn.as[i], j, k)));
                    DrawRectangle(r.x+k*cell_width+cntrx, r.y+accumelator*cell_height+i*gap, cell_width, cell_height, connection_color);
                }
                accumelator += 1;
            }
        }

        snprintf(buffer, sizeof(buffer), "ACTIVATIONS");
        tw =  MeasureText(buffer, font_size);
        DrawText(buffer, r.x+(r.w/2-tw/2), r.y+r.h, font_size, RAYWHITE);
        break;
    default:
        NN_ASSERT(0 && "Unreachable");
    }

}

void gym_cost_render(Costs costs, LayoutRect r)
{
    char buffer[256];
    float font_size = r.h*(50.0f/WINDOW_HEIGHT);

    snprintf(buffer, sizeof(buffer), "COST-FUNCTION");
    int tw =  MeasureText(buffer, font_size);
    DrawText(buffer, r.x+(r.w/2-tw/2), r.y+r.h, font_size, RAYWHITE);

    r.w -= BORDER_HPAD;
    r.h -= BORDER_VPAD;

    float min = FLT_MAX, max = FLT_MIN;

    for (size_t i = 0; i < costs.count; i++) {
        if (costs.items[i] < min) min = costs.items[i];
        if (costs.items[i] > max) max = costs.items[i];
    }

    if (min > 0) min = 0;
    size_t n = costs.count;
    if (n < 1000) n = 1000;

    // DRAW THE AXIS LINES
    int thick = r.h*(7.0f/WINDOW_HEIGHT);
    int lx = r.x+BORDER_HPAD/2-thick;
    int ly = r.y+BORDER_HPAD/2-thick;
    float tri_val = (r.h*10.f/WINDOW_HEIGHT);

    // X-AXIS 
    DrawLineEx((Vector2) {lx, ly+r.h}, (Vector2) {lx+r.w, ly+r.h}, thick, RAYWHITE);
    DrawTriangle((Vector2) {lx+r.w, ly+r.h-tri_val}, (Vector2) {lx+r.w, ly+r.h+tri_val}, (Vector2) {lx+r.w+tri_val, ly+r.h}, RAYWHITE);

    // Y-AXIS
    DrawLineEx((Vector2) {lx, ly}, (Vector2) {lx, ly+r.h}, thick, RAYWHITE);
    DrawTriangle((Vector2) {lx-tri_val, ly}, (Vector2) {lx+tri_val, ly}, (Vector2) {lx, ly-tri_val}, RAYWHITE);

    snprintf(buffer, sizeof(buffer), "0");
    DrawText(buffer, lx+5, ly+r.h+5, font_size, RAYWHITE);


    for (size_t i = 0; (i+1) < costs.count; i++) {
        int x1 = r.x + BORDER_HPAD/2 + (float)r.w/n*i;
        int y1 = r.y + BORDER_VPAD/2 + (1 - (costs.items[i]-min) / (max-min))*r.h;

        int x2 = r.x + BORDER_HPAD/2 + (float)r.w/n*(i+1);
        int y2 = r.y + BORDER_VPAD/2 + (1 - (costs.items[i+1]-min) / (max-min))*r.h;

        float thick = r.h*(7.0f/WINDOW_HEIGHT);
        DrawLineEx((Vector2) {x1, y1}, (Vector2) {x2, y2}, thick, MAROON);
    }
}

void gym_status_line_render(int h, int rw, size_t epoch, size_t max_epoch, float rate, float cost, Region* r)
{
    char buffer[256];
    snprintf(buffer, sizeof(buffer),
    "Epoch: %zu/%zu\t\tRate: %.5f\t\tCost: %f\t\t Memory Usage: %zu",
    epoch, max_epoch, rate, cost, r->size);
    
    float font_size = h*(50.0f/WINDOW_HEIGHT);
    int tw =  MeasureText(buffer, font_size);
    DrawText(buffer, rw/2-tw/2, 30, font_size, WHITE);
    
}

void gym_slider_render(Vector2 rect_corner, Vector2 rect_size, Vector2 slider_center, float slider_raduis, bool* slider_clicked, float* scroll)
{
        DrawRectangleV(rect_corner, rect_size, RAYWHITE);
        DrawCircleV(slider_center, slider_raduis, MAROON);

        if (*slider_clicked) {
            float x = GetMousePosition().x;

            if (x <= rect_corner.x) x = rect_corner.x;
            if (x >= rect_corner.x+rect_size.x) x =rect_corner.x+rect_size.x;

            *scroll = (x - rect_corner.x) / rect_size.x;
        }

        if (IsMouseButtonPressed(MOUSE_LEFT_BUTTON)) {
            float dist = Vector2Distance(GetMousePosition(), slider_center);
            if (dist <= slider_raduis) {
                *slider_clicked = true;
            }
        }

        if (IsMouseButtonReleased(MOUSE_LEFT_BUTTON)) {
            *slider_clicked = false;
        }
}

#endif // !NN_ENABLE_GYM

#endif // NN_IMPLEMENTATION
