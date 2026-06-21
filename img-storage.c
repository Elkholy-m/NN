#include <stdio.h>
#include <time.h>
#include <float.h>
#include <raylib.h>
#include "stb_image.h"
#include "stb_image_write.h"


#define NN_IMPLEMENTATION
#include "nn.h"

#define DA_INIT_CAPACITY 256
#define WINDOW_FACTOR 80
#define WINDOW_WIDTH  (16*WINDOW_FACTOR)
#define WINDOW_HEIGHT (9*WINDOW_FACTOR)
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

void nn_render(NN nn, int x, int y, int w, int h)
{
    unsigned int neuron_raduis = h*((float)25/WINDOW_HEIGHT);
    Color low__color       = {.a =0xFF, .b =0xFF, .g =0x00, .r =0xFF};
    Color high_color       = {.a =0xFF, .b =0x00, .g =0xFF, .r =0x00};

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
    char buffer[256];
    float font_size = h*(50.0f/WINDOW_HEIGHT);

    snprintf(buffer, sizeof(buffer), "COST-FUNCTION");
    int tw =  MeasureText(buffer, font_size);
    DrawText(buffer, x+(w/2-tw/2), y+h, font_size, WHITE);

    w -= BORDER_HPAD;
    h -= BORDER_VPAD;

    float min, max;
    get_cost_minmax(costs, &min, &max);
    size_t n = costs.count;

    if (min > 0) min = 0;
    /* TODO: MAKE THE PLOT MORE STABLE BUT THERE IS STILL A BUG */
    /* if (n < 1000) n = 1000; */

    // DRAW THE AXIS LINES
    int thick = h*(5.0f/WINDOW_HEIGHT);
    int lx = x+BORDER_HPAD/2-thick;
    int ly = y+BORDER_HPAD/2-thick;
    float tri_val = (h*10.f/WINDOW_HEIGHT);

    // X-AXIS
    DrawLineEx((Vector2) {lx, ly+h}, (Vector2) {lx+w, ly+h}, thick, WHITE);
    DrawTriangle((Vector2) {lx+w, ly+h-tri_val}, (Vector2) {lx+w, ly+h+tri_val}, (Vector2) {lx+w+tri_val, ly+h}, WHITE);

    // Y-AXIS
    DrawLineEx((Vector2) {lx, ly}, (Vector2) {lx, ly+h}, thick, WHITE);
    DrawTriangle((Vector2) {lx-tri_val, ly}, (Vector2) {lx+tri_val, ly}, (Vector2) {lx, ly-tri_val}, WHITE);

    snprintf(buffer, sizeof(buffer), "0");
    DrawText(buffer, lx+5, ly+h+5, font_size, WHITE);


    for (size_t i = 0; (i+1) < n; i++) {
        int x1 = x + BORDER_HPAD/2 + (float)w/n*i;
        int y1 = y + BORDER_VPAD/2 + (1 - (costs.items[i]-min) / (max-min))*h;

        int x2 = x + BORDER_HPAD/2 + (float)w/n*(i+1);
        int y2 = y + BORDER_VPAD/2 + (1 - (costs.items[i+1]-min) / (max-min))*h;

        float thick = h*(5.0f/WINDOW_HEIGHT);
        DrawLineEx((Vector2) {x1, y1}, (Vector2) {x2, y2}, thick, RED);
    }
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
    char* program = shift_args(&argc, &argv);
    if (argc <= 0) {
        fprintf(stderr, "Usage: %s <img-file-path>\n", program);
        return 1;
    }

    char* img_file_path = shift_args(&argc, &argv);
    int img_width, img_height, img_comp;
    uint8_t* img_pixels = (uint8_t*)stbi_load(img_file_path, &img_width, &img_height, &img_comp, 0);
    if (img_pixels == NULL) {
        fprintf(stderr, "ERROR: Couldn't open file: %s\n", img_file_path);
        return 1;
    }

    if (img_comp != 1) {
        fprintf(stderr, "ERROR: The image is not grey scaled it has %d components\n", img_comp);
        return 1;
    }

    // INTIALIZING THE NURAL NETWORK
    size_t arch[] = {2, 7, 4, 1};
    size_t rows = img_width*img_height;
    size_t cols = arch[0] + arch[ARRAY_LEN(arch)-1];
    Mat t = mat_alloc(rows, cols);
    for (int y = 0; y < img_height; y++) {
        for (int x = 0; x < img_width; x++) {
            int i = y*img_width+x;
            MAT_AT(t, i, 0) = (float)x/(img_width-1);
            MAT_AT(t, i, 1) = (float)y/(img_height-1);
            MAT_AT(t, i, 2) = img_pixels[i]/255.f;
        }
    }

    Mat ti = {
        .rows = t.rows,
        .cols = arch[0],
        .stride = t.stride,
        .es = &MAT_AT(t, 0, 0),
    };

    Mat to = {
        .rows = t.rows,
        .cols = arch[ARRAY_LEN(arch)-1],
        .stride = t.stride,
        .es = &MAT_AT(t, 0, ti.cols),
    };

    NN nn = nn_alloc(arch, ARRAY_LEN(arch));
    NN g  = nn_alloc(arch, ARRAY_LEN(arch));
    nn_rand(nn, -1, 1);

    SetConfigFlags(FLAG_WINDOW_RESIZABLE);
    InitWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "IMAGE-STORAGE");;
    SetTargetFPS(60);

    srand(time(0));
    float rate = 1;
    size_t epoch = 0;
    size_t max_epoch = 100000;
    size_t epochs_per_frame = 103;
    Costs costs = {0};
    bool paused = true;

    Image img_prev  = GenImageColor(img_width, img_height, BLACK);
    Texture2D training_prev  = LoadTextureFromImage(img_prev);
    Texture2D original_prev = LoadTextureFromImage(img_prev);
    while (!WindowShouldClose()) {
        if(IsKeyPressed(KEY_R)) {
            paused = true;
            epoch = 0;
            nn_rand(nn, -1, 1);
            costs.count = 0;
        }

        if (IsKeyPressed(KEY_SPACE)) paused = !paused;
        for (size_t j = 0; j < epochs_per_frame && epoch < max_epoch && !paused; j++) {
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

        x = 0;
        w = rw/3;
        h = rh*9/16;
        y = rh/2-h/2;

        cost_render(costs, x, y, w, h);
        
        x += w;
        nn_render(nn, x, y, w, h);

        x += w;
        float scale = h*13.f/WINDOW_HEIGHT;

        for (int y = 0; y < img_height; ++y) {
            for (int x =  0; x < img_width; ++x) {
                MAT_AT(NN_INPUT(nn), 0, 0) = (float)x/(img_width-1);
                MAT_AT(NN_INPUT(nn), 0, 1) = (float)y/(img_height-1);

                nn_forward(nn);
                uint8_t pixel = MAT_AT(NN_OUTPUT(nn), 0, 0)*255;
                ImageDrawPixel(&img_prev, x, y, (Color) {pixel, pixel, pixel, 255});
            }
        }

        Vector2 texture_position = (Vector2) {x+w/2-training_prev.width/2*scale, y+h/2-training_prev.height*scale};
        UpdateTexture(training_prev, img_prev.data);
        DrawTextureEx(training_prev, texture_position, 0, scale, WHITE);

        for (int y = 0; y < img_height; ++y) {
            for (int x =  0; x < img_width; ++x) {
                uint8_t pixel = img_pixels[y*img_width+x];
                ImageDrawPixel(&img_prev, x, y, (Color) {pixel, pixel, pixel, 255});
            }
        }

        texture_position = (Vector2) {x+w/2-training_prev.width/2*scale, y+h/2};
        DrawTextureEx(original_prev, texture_position, 0, scale, WHITE);
        UpdateTexture(original_prev, img_prev.data);


        char buffer[256];
        snprintf(buffer, sizeof(buffer),
                 "Epoch: %zu/%zu\t\tRate: %.2f\t\tCost: %f",
                 epoch, max_epoch, rate, nn_cost(nn, ti, to));
        
        float font_size = h*(50.0f/WINDOW_HEIGHT);
        int tw =  MeasureText(buffer, font_size);
        DrawText(buffer, rw/2-tw/2, 30, font_size, WHITE);

        EndDrawing();
    }

    char screenshot_file_path[256];
    snprintf(screenshot_file_path, sizeof(screenshot_file_path),
             "./nn_screen_shots/%s", img_file_path);
    TakeScreenshot(screenshot_file_path);
    CloseWindow();
    
    size_t out_width = 512;
    size_t out_height = 512;
    uint8_t *out_pixels = malloc(sizeof(*out_pixels)*out_width*out_height);
    assert(out_pixels != NULL);

    for (size_t y = 0; y < out_height; ++y) {
        for (size_t x = 0; x < out_width; ++x) {
            MAT_AT(NN_INPUT(nn), 0, 0) = (float)x/(out_width - 1);
            MAT_AT(NN_INPUT(nn), 0, 1) = (float)y/(out_height - 1);
            nn_forward(nn);
            uint8_t pixel = MAT_AT(NN_OUTPUT(nn), 0, 0)*255.f;
            out_pixels[y*out_width + x] = pixel;
        }
    }

    const char *out_file_path = "upscaled.png";
    if (!stbi_write_png(out_file_path, out_width, out_height, 1, out_pixels, out_width*sizeof(*out_pixels))) {
        fprintf(stderr, "ERROR: could not save image %s\n", out_file_path);
        return 1;
    }

    printf("Generated %s from %s\n", out_file_path, img_file_path);
    return 0;
}
