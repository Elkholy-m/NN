#include <stdio.h>
#include <time.h>
#include "raylib.h"

#define NN_IMPLEMENTATION
#include "nn.h"
#define IMG_FACTOR 80
#define IMG_WIDTH  (16*IMG_FACTOR)
#define IMG_HEIGHT (9*IMG_FACTOR)

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

#define BITS 4

int main() {
    // PREPARE THE TRAINING SET FOR ADDER
    int n = (1<<BITS);
    int rows = n*n;
    Mat ti = mat_alloc(rows, 2*BITS);
    Mat to = mat_alloc(rows, BITS+1);
    for (int i = 0; i < rows; i++) {
        int x = i/n; 
        int y = i%n; 
        int z = x + y;
        for (int j = 0; j < BITS; j++) {
            MAT_AT(ti, i, j)      = (x>>(BITS-1-j)) & 1;
            MAT_AT(ti, i, j+BITS) = (y>>(BITS-1-j)) & 1;
            MAT_AT(to, i, j)      = (z>>(BITS-1-j)) & 1;
        }
        MAT_AT(to, i, BITS) = z >= n;
    }

    // MODEL LEARNING
    srand(time(0));
    float rate = 1;

    size_t arch[] = {2*BITS, 4*BITS, BITS+1};
    NN nn = nn_alloc(arch, ARRAY_LEN(arch));
    NN g  = nn_alloc(arch, ARRAY_LEN(arch));
    nn_rand(nn, 0, 1);

    InitWindow(IMG_WIDTH, IMG_HEIGHT, "ADDER_NN");
    SetTargetFPS(60);

    size_t i = 0;
    while (!WindowShouldClose()) {
        if (i < 3000) {
            nn_backprop(nn, g, ti, to);
            nn_learn(nn, g, rate);
            printf("%zu: COST = %f\n", i, nn_cost(nn, ti, to));
            i++;
        }

        nn_render_raylib(nn);
    }

    CloseWindow();

    // TESTING MODEL
    int fails = 0;
    for (int x = 0; x < n; x++) {
        for (int y = 0; y < n; y++) {
            int z = x + y;
            for (int j = 0; j < BITS; j++) {
                MAT_AT(NN_INPUT(nn), 0, j)      = x>>(BITS-1-j) & 1;
                MAT_AT(NN_INPUT(nn), 0, j+BITS) = y>>(BITS-1-j) & 1;
            }
            nn_forward(nn);

            int a = 0;
            for (int j = 0; j < BITS; j++) {
                a |= (MAT_AT(NN_OUTPUT(nn), 0, j) >  0.5f)<<(BITS-1-j);
            }
            int c = MAT_AT(NN_OUTPUT(nn), 0, BITS)>0.5f;

            // COMPARE THE MODEL FORWARDING WITH THE EXPECTED
            int ea  = z&(n-1);
            int ec = z >= n;
            if (a != ea || ec != c) {
                printf("%2d + %2d = %2d, c = %2d\tExpected = %2d, ec = %2d\n"
                       , x, y, a, c, ea, ec);
                fails++;
            }
        }
    }
    if (fails == 0) printf("NICE MODEL :)\n");

    return 0;
}
