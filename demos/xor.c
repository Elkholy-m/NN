#include <stdio.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#define HELPER_IMPLEMENTATION
#include "helper.h"

#define NN_IMPLEMENTATION
#define NN_ENABLE_GYM
#define TRADITIONAL_APPROACH
#define NN_ACT ACT_SIN
#include "nn.h"

size_t arch[] = {2, 2, 1};

float td[] = {
    0, 0, 0,
    0, 1, 1,
    1, 0, 1,
    1, 1, 0,
};

Mat xor = { .rows = 4, .cols = 3, .stride = 3, .es = td, };

int main()
{
    Mat ti = mat_sub(xor, COORDINATE(0, 0), COORDINATE(3, 1));
    Mat to = mat_sub(xor, COORDINATE(0, 2), COORDINATE(3, 2));

    srand(time(0));
    NN nn = nn_alloc(arch, ARRAY_LEN(arch));
    NN g  = nn_alloc(arch, ARRAY_LEN(arch));
    nn_rand(nn, -1, 1);

    Costs costs = {0};
    bool paused = true;
    float rate = 0.1;
    float eps = 1e-2;

    LayoutStack ls = {0};
    LayoutRect r = {0};
    size_t max_epochs = 5000;
    size_t epoch = 0;

    SetConfigFlags(FLAG_WINDOW_RESIZABLE);
    InitWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "XOR_NN");
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
            nn_finite_diff(nn, g, ti, to, eps);
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
        float fontSize = r.h*(25.0f/WINDOW_HEIGHT);
        int n = 1;
        char buffer[256];
        float spacing = r.h*0.1;
        float cntrx = r.w/2-(n+1)*spacing/2;
        float cntry = r.h/2-(n+1)*spacing/2;

        for (size_t i = 0; i < 2; i++) {
            snprintf(buffer, sizeof(buffer), "%zu\t", i);
            DrawText(buffer, r.x+(i*spacing)+cntrx, r.y-spacing+cntry, fontSize, WHITE);
            
            snprintf(buffer, sizeof(buffer), "%zu\t", i);
            DrawText(buffer, r.x-spacing+cntrx, r.y+(i*spacing)+cntry, fontSize, WHITE);
            for (size_t j = 0; j < 2; j++) {
                MAT_AT(NN_INPUT(nn), 0, 0) = i;
                MAT_AT(NN_INPUT(nn), 0, 1) = j;
                nn_forward(nn);
                int y =  (MAT_AT(NN_OUTPUT(nn), 0, 0) >= 0.5f) ? 1 : 0;

                Color tr = GREEN;
                int exp  = i^j;
                if (y != exp) tr = RED;
                snprintf(buffer, sizeof(buffer), "%d\t", y);
                
                DrawText(buffer, r.x+(i*spacing)+cntrx, r.y+(j*spacing)+cntry, fontSize, tr);
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
