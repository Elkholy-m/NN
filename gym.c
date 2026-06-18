#include <stdio.h>
#include <time.h>
#include <ctype.h>
#include <string.h>
#include <float.h>
#include <stdint.h>
#include "raylib.h"

#define NN_IMPLEMENTATION
#include "nn.h"

#define DA_INIT_CAPACITY 256
#define IMG_FACTOR 80
#define IMG_WIDTH  (16*IMG_FACTOR)
#define IMG_HEIGHT (9*IMG_FACTOR)

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
    size_t* items;
    size_t capacity;
    size_t count;
} Arch;

typedef struct {
    float* items;
    size_t capacity;
    size_t count;
} Costs;

void nn_render(NN nn, int x, int y, int w, int h)
{
    unsigned int neuron_raduis = h*((float)25/IMG_HEIGHT);
    Color low__color       = {.a =0xFF, .b =0xFF, .g =0x00, .r =0xFF};
    Color high_color       = {.a =0xFF, .b =0x00, .g =0xFF, .r =0x00};
    

    size_t border_vpad = 50;
    size_t border_hpad = 50;

    size_t nn_x = w - 2*border_hpad;
    size_t nn_y = h - 2*border_vpad;

    size_t no_of_layers = nn.count + 1;
    int hpad = nn_x/no_of_layers;
    
    for (size_t i = 0; i < no_of_layers; i++) {
        int vpad_1 = nn_y/nn.as[i].cols;
        for (size_t j = 0; j < nn.as[i].cols; j++) {
            int cx1 = x + border_hpad + i*hpad + hpad/2;
            int cy1 = y +border_vpad + j*vpad_1 + vpad_1/2;
            if (i < (no_of_layers - 1)) {
                int vpad_2 = nn_y/nn.as[i+1].cols;
                for (size_t k = 0; k < nn.as[i+1].cols; k++) {
                    int cx2 = x + border_hpad + (i+1)*hpad + hpad/2;
                    int cy2 = y +border_vpad + k*vpad_2 + vpad_2/2;
                    
                    float alpha = sigmoidf(MAT_AT(nn.ws[i], j, k));;
                    float thick = h*(3.0f/IMG_HEIGHT);
                    Color connection_color = ColorAlphaBlend(low__color, high_color, WHITE);
                    high_color.a = floorf(255.f*alpha);
                    DrawLineEx((Vector2){cx1, cy1}, (Vector2){cx2, cy2}, thick, connection_color);
                }
            }
            if (i == 0) {
                DrawCircle(cx1, cy1, neuron_raduis, GRAY);
            } else {
                high_color.a = floorf(255.f*sigmoidf(MAT_AT(nn.bs[i-1], 0, j)));
                Color neuron_color = ColorAlphaBlend(low__color, high_color, WHITE);
                DrawCircle(cx1, cy1, neuron_raduis, neuron_color);
            }
        }
    }
}

void get_cost_minmax(Costs costs, float* min, float* max)
{
    // cheet idea the max always is the first cost
    // cheet idea the min always is the last cost
    *min = FLT_MAX;
    *max = FLT_MIN;

    for (size_t i = 0; i < costs.count; i++) {
        if (costs.items[i] < *min) *min = costs.items[i];
        if (costs.items[i] > *max) *max = costs.items[i];
    }
}

void cost_render(Costs costs, int x, int y, int w, int h)
{
    size_t boarder_hpad = 50;
    size_t boarder_vpad = 50;

    w -= boarder_hpad;
    h -= boarder_vpad;

    float min, max;
    get_cost_minmax(costs, &min, &max);
    size_t n = costs.count;

    if (min > 0) min = 0;
    /* if (n < 1000) n = 1000; */

    for (size_t i = 0; (i+1) < n; i++) {
        int x1 = x + boarder_hpad/2 + (float)w/n*i;
        int y1 = y + boarder_vpad/2 + (1 - (costs.items[i]-min) / (max-min))*h;

        int x2 = x + boarder_hpad/2 + (float)w/n*(i+1);
        int y2 = y + boarder_vpad/2 + (1 - (costs.items[i+1]-min) / (max-min))*h;

        float thick = h*(5.0f/IMG_HEIGHT);
        DrawLineEx((Vector2) {x1, y1}, (Vector2) {x2, y2}, thick, RED);
    }
}

Arch get_arch_from_file(const char* arch_file_path)
{
    Arch arch = {0};
    int file_len = 0;
    unsigned char * buffer = LoadFileData(arch_file_path, &file_len);
    if (buffer == NULL) {
        fprintf(stderr, "Can't read file %s\n", arch_file_path);
    }
    // choping by delimeter `SPC` AND NEW-LINE
    char delim[] = " \n";
    char* token = strtok((char*)buffer, delim);

    // get the integers
    while (token != NULL) {
            size_t tokenlen = strlen(token);
            size_t i = 0;
            uint32_t res = 0;
            while (i < tokenlen && isdigit(token[i])) {
                res = res*10 + token[i] - '0';
                i++;
            }

            if (res > 0) {
                DA_APPEND(&arch, res);
            }
        token = strtok(NULL, delim);
    }
    return arch;

}

char* shift_args(int* argc, char*** argv)
{
    char* result = *argv[0];
    *argc -= 1;
    *argv += 1;
    return result; 
}

int main(int argc, char** argv)
{
    // GETTING THE ARGS OF THE PROGRAM
    char* program = shift_args(&argc, &argv);
    if (argc <= 0) {
        fprintf(stderr, "Missing arch file: Usage %s <arch-file> <data-file>", program);
        return 1;
    }
    char* arch_file_path = shift_args(&argc, &argv);

    if (argc <= 0) {
        fprintf(stderr, "Missing data file: Usage %s <arch-file> <data-file>", program);
        return 1;
    }
    char* mat_file_path = shift_args(&argc, &argv);

    // GETTING THE ARCH FROM arch_file_path
    Arch arch = get_arch_from_file(arch_file_path);

    // GETTING THE  MAT FROM mat_file_path
    FILE* in = fopen(mat_file_path, "rb");
    if (in == NULL) {
        fprintf(stderr, "ERROR: Can't read file: %s", mat_file_path);
        return 1;
    }
    Mat t = mat_load(in);
    fclose(in);
    NN_ASSERT(arch.count > 1);
    NN_ASSERT((arch.items[0]+arch.items[arch.count-1]) == t.cols);
    Mat ti = {
        .rows   = t.rows,
        .cols   = arch.items[0],
        .stride = t.stride,
        .es     = &MAT_AT(t, 0, 0),
    };

    Mat to = {
        .rows   = t.rows,
        .cols   = arch.items[arch.count-1],
        .stride = t.stride,
        .es     = &MAT_AT(t, 0, arch.items[0]),
    };

    // MODEL LEARNING
    NN nn = nn_alloc(arch.items, arch.count);
    NN g  = nn_alloc(arch.items, arch.count);
    nn_rand(nn, 0, 1);

    SetConfigFlags(FLAG_WINDOW_RESIZABLE);
    InitWindow(IMG_WIDTH, IMG_HEIGHT, "GYM");
    SetTargetFPS(60);

    srand(time(0));
    float rate = 1;
    size_t epoch = 0;
    size_t max_epoch = 5000;
    Costs costs = {0};
    while (!WindowShouldClose()) {
        for (size_t j = 0; j < 7 && epoch < max_epoch; j++) {
            nn_backprop(nn, g, ti, to);
            nn_learn(nn, g, rate);
            DA_APPEND(&costs, nn_cost(nn, ti, to));
            epoch++;
        }

        Color background_color = {.a =0xFF, .b =0x18, .g =0x18, .r =0x18};
        BeginDrawing();
        ClearBackground(background_color);
        int x, y, w, h;

        int rw = GetRenderWidth();
        int rh = GetRenderHeight();

        w = rw/2;
        h = rh*9/16;
        x = rw-w;
        y = rh/2-h/2;
        nn_render(nn, x, y, w, h);

        x = 0;
        cost_render(costs, x, y, w, h);

        h = rh;
        char buffer[256];
        float font_size = h*(30.0f/IMG_HEIGHT);
        snprintf(buffer, sizeof(buffer), "Epoch: %zu/%zu\t\tRate: %.2f\t\tCost: %f", epoch, max_epoch, rate, nn_cost(nn, ti, to));
        
        int tw =  MeasureText(buffer, font_size);
        DrawText(buffer, rw/2-tw/2, 30, font_size, WHITE);
        EndDrawing();
    }

    CloseWindow();
    return 0;
}
