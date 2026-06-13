#include <stdio.h>
#include <time.h>
#include "raylib.h"

#define NN_IMPLEMENTATION
#include "nn.h"

#define IMG_WIDTH   800
#define IMG_HEIGHT  600

void nn_render_raylib(NN nn) {
    unsigned int neuron_raduis = 25;
    Color background_color = {.a =0xFF, .b =0x18, .g =0x18, .r =0x18};
    Color low__color       = {.a =0xFF, .b =0xFF, .g =0x00, .r =0xFF};
    Color high_color       = {.a =0xFF, .b =0x00, .g =0xFF, .r =0x00};
    
    BeginDrawing();
    ClearBackground(background_color);

    size_t border_vpad = 50;
    size_t border_hpad = 50;

    size_t nn_x = IMG_WIDTH -2*border_hpad;
    size_t nn_y = IMG_HEIGHT-2*border_vpad;

    size_t no_of_layers = nn.count + 1;
    int hpad = nn_x/no_of_layers;
    
    for (size_t i = 0; i < no_of_layers; i++) {
        int vpad_1 = nn_y/nn.as[i].cols;
        for (size_t j = 0; j < nn.as[i].cols; j++) {
            int cx1 = border_hpad + i*hpad + hpad/2;
            int cy1 = border_vpad + j*vpad_1 + vpad_1/2;
            if (i < (no_of_layers - 1)) {
                int vpad_2 = nn_y/nn.as[i+1].cols;
                for (size_t k = 0; k < nn.as[i+1].cols; k++) {
                    int cx2 = border_hpad + (i+1)*hpad + hpad/2;
                    int cy2 = border_vpad + k*vpad_2 + vpad_2/2;
                    
                    low__color.a = floorf(255.f*sigmoidf(MAT_AT(nn.ws[i], j, k)));
                    Color connection_color = ColorAlphaBlend(high_color, low__color, WHITE);
                    DrawLine(cx1, cy1, cx2, cy2, connection_color);
                }
            }
            if (i == 0) {
                Color input_color = {.a =0xFF, .b =0x80, .g =0x80, .r =0x80};
                DrawCircle(cx1, cy1, neuron_raduis, input_color);
            } else {
                low__color.a = floorf(255.f*sigmoidf(MAT_AT(nn.bs[i-1], 0, j)));
                Color neuron_color = ColorAlphaBlend(high_color, low__color, WHITE);
                DrawCircle(cx1, cy1, neuron_raduis, neuron_color);
            }
        }
    }

    EndDrawing();
}

int32_t pixels[IMG_WIDTH*IMG_HEIGHT];

float xor_td[] = {
    0, 0, 0,
    0, 1, 1,
    1, 0, 1,
    1, 1, 0,
};

Mat xor = { .rows = 4, .cols = 3, .stride = 3, .es = xor_td, };

void xor_test(NN nn) {
    for (size_t i = 0; i < 2; i++) {
        for (size_t j = 0; j < 2; j++) {
            MAT_AT(NN_INPUT(nn), 0, 0) = i;
            MAT_AT(NN_INPUT(nn), 0, 1) = j;
            nn_forward(nn);
            float y =  MAT_AT(NN_OUTPUT(nn), 0, 0);
            printf("%zu ^ %zu = %f\n", i, j, y);
        }
    }
}

int main() {
    srand(time(0));
    size_t arch[] = {2, 2, 1};
    NN nn = nn_alloc(arch, ARRAY_LEN(arch));
    NN g  = nn_alloc(arch, ARRAY_LEN(arch));
    nn_rand(nn, 0, 1);

    Mat m = xor;

    Mat ti = mat_sub(m, COORDINATE(0, 0), COORDINATE(3, 1));
    Mat to = mat_sub(m, COORDINATE(0, 2), COORDINATE(3, 2));

    float rate = 1e-1;

    InitWindow(IMG_WIDTH, IMG_HEIGHT, "XOR_NN");
    SetTargetFPS(60);

    size_t i = 0;
    while (!WindowShouldClose()) {
        if (i < 5000) {
            nn_backprop(nn, g, ti, to);
            nn_learn(nn, g, rate);
            printf("%zu: Cost = %f\n", i, nn_cost(nn, ti, to));
            i++;
        }

        nn_render_raylib(nn);
    }

    CloseWindow();
    NN_PRINT(nn);
    xor_test(nn);
}
