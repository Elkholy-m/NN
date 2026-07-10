#include <stdio.h>
#include <time.h>
#include "raylib.h"

#define NN_IMPLEMENTATION
#define NN_ENABLE_GYM
#include "nn.h"

#define BITS 4
size_t arch[] = {2*BITS, 4*BITS, BITS+1};

int main() {
    // PREPARE THE TRAINING SET FOR ADDER
    int n = (1<<BITS);
    int rows = n*n;
    int maxwidth = (n == 0) ? 1 : (int)log10(n) + 1;
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
    NN nn = nn_alloc(arch, ARRAY_LEN(arch));
    NN g  = nn_alloc(arch, ARRAY_LEN(arch));
    nn_rand(nn, -1, 1);

    Costs costs = {0};
    bool paused = true;
    float rate = 0.5f;

    LayoutStack ls = {0};
    LayoutRect r = {0};
    size_t max_epochs = 100*1000;
    size_t epoch = 0;

    SetConfigFlags(FLAG_WINDOW_RESIZABLE);
    InitWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "ADDER_NN");
    SetTargetFPS(60);

    while (!WindowShouldClose()) {
        if(IsKeyPressed(KEY_R)) {
            paused = true;
            epoch = 0;
            nn_rand(nn, -1, 1);
            costs.count = 0;
        }

        if (IsKeyPressed(KEY_SPACE)) paused = !paused;

        for (size_t i = 0; i < 10 && epoch < max_epochs && !paused; ++i) {
            nn_backprop(nn, g, ti, to);
            nn_learn(nn, g, rate);
            DA_APPEND(&costs, nn_cost(nn, ti, to));
            epoch++;
        }

        Color background_color = {.a =0xFF, .b =0x18, .g =0x18, .r =0x18};
        BeginDrawing();
        ClearBackground(background_color);

        int rw = GetRenderWidth();
        int rh = GetRenderHeight();
        size_t frame = rh*0.25;
        size_t gap = rh*0.03;

        layout_stack_push(&ls, rect_constructor(0, frame, rw, rh-2*frame), LO_HORZ, 3, gap);
            gym_cost_render(costs, layout_stack_slot(&ls));
            gym_nn_render(nn, layout_stack_slot(&ls));

        // RENDERING THE VERIFICATION SLOT
            r =layout_stack_slot(&ls);
            float fontSize = r.h*(20.0f/WINDOW_HEIGHT);
            char buffer[256];
            float spacing = r.h*0.05;
            float cntrx = r.w/2-(n+1)*spacing/2;
            float cntry = r.h/2-(n+1)*spacing/2;

            for (int x = 0; x < n; x++) {
                snprintf(buffer, sizeof(buffer), "%*d\t", maxwidth, x);
                DrawText(buffer, r.x+(x*spacing)+cntrx, r.y-spacing+cntry, fontSize, WHITE);

                snprintf(buffer, sizeof(buffer), "%*d\t", maxwidth, x);
                DrawText(buffer, r.x-spacing+cntrx, r.y+(x*spacing)+cntry, fontSize, WHITE);

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
                    if (c == 1) DrawRectangle(r.x+(x*spacing)+cntrx, r.y+(y*spacing)+cntry, ceilf(spacing), ceilf(spacing), DARKGRAY);

                    // COMPARE THE MODEL FORWARDING WITH THE EXPECTED
                    Color tr = GREEN;
                    int ea  = z&(n-1);
                    int ec = z >= n;
                    if (a != ea || ec != c) tr = RED;

                    snprintf(buffer, sizeof(buffer), "%*d\t", maxwidth, a);
                    DrawText(buffer, r.x+(x*spacing)+cntrx, r.y+(y*spacing)+cntry, fontSize, tr);
                }
            }

            gym_status_line_render(r.h, rw, epoch, max_epochs, rate, costs.count > 0 ? costs.items[costs.count - 1] : 0);
        layout_stack_pop(&ls);
        EndDrawing();

        // ENSURE NO MEMORY LEAKS INSIDE THE LOOP
        assert(ls.count == 0);
    }

    CloseWindow();
    return 0;
}
