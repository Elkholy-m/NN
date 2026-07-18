#include <stdarg.h>
#include <stdio.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#define HELPER_IMPLEMENTATION
#include "helper.h"
#include "stb_image.h"
#include "stb_image_write.h"

#define NN_IMPLEMENTATION
#define NN_ENABLE_GYM
#include "nn.h"

#define STR2(s) #s
#define STR(s) STR2(s)
#define READ_END 0
#define WRITE_END 1
#define FPS 30

#define out_width  256
#define out_height 256
uint32_t out_pixels[out_width*out_height];

size_t arch[] = {3, 14, 14, 11, 1};

void original_img_construct(Image* img_prev, uint8_t* pixels)
{
    for (int y = 0; y < img_prev->height; ++y) {
        for (int x =  0; x < img_prev->width; ++x) {
            uint8_t pixel = pixels[y*img_prev->width+x];
            ImageDrawPixel(img_prev, x, y, (Color) {pixel, pixel, pixel, 255});
        }
    }
}

void verify_img_construct(Image* img_prev, NN nn, float scroll)
{
    for (int y = 0; y < img_prev->height; ++y) {
        for (int x =  0; x < img_prev->width; ++x) {
            MAT_AT(NN_INPUT(nn), 0, 0) = (float)x/(img_prev->width-1);
            MAT_AT(NN_INPUT(nn), 0, 1) = (float)y/(img_prev->height-1);
            MAT_AT(NN_INPUT(nn), 0, 2) = scroll;

            nn_forward(nn);
            float activation = MAT_AT(NN_OUTPUT(nn), 0, 0);
            if (activation < 0) activation = 0;
            if (activation > 1) activation = 1;
            uint8_t pixel = activation*255;
            ImageDrawPixel(img_prev, x, y, (Color) {pixel, pixel, pixel, 255});
        }
    }
}

void verification_render(Image* img_prev, NN nn,
                         int x, int y, int w, int h,
                         float scale,
                         float* scroll1, bool* slider_clicked1,
                         float* scroll2, bool* slider_clicked2,
                         Texture2D training_prev1, Texture2D training_prev2, Texture2D training_prev3,
                         Texture2D original_prev1, Texture2D original_prev2)
{
    // DRAWING THE DOWN LEFT TEXTURE (TRAINING OF FIRST IMAGE)
    verify_img_construct(img_prev, nn, 0.0f);
    Vector2 texture_position = {x+w/2-training_prev1.width*scale, y+h/2-training_prev1.height*0.5f*scale};
    UpdateTexture(training_prev1, img_prev->data);
    DrawTextureEx(training_prev1, texture_position, 0, scale, RAYWHITE);

    // DRAWING THE UP LEFT TEXTURE (ORIGINAL OF FIRST IMAGE)
    texture_position = (Vector2) {x+w/2-training_prev1.width*scale, y+h/2-training_prev1.height*1.5f*scale};
    DrawTextureEx(original_prev1, texture_position, 0, scale, RAYWHITE);

    // DRAWING THE DOWN RIGHT TEXTURE (TRAINING OF SECOND IMAGE)
    verify_img_construct(img_prev, nn, 1.0f);
    texture_position = (Vector2) {x+w/2, y+h/2-training_prev2.height*0.5f*scale};
    UpdateTexture(training_prev2, img_prev->data);
    DrawTextureEx(training_prev2, texture_position, 0, scale, RAYWHITE);

    // DRAWING THE UP RIGHT TEXTURE (ORIGINAL OF SECOND IMAGE)
    texture_position = (Vector2) {x+w/2, y+h/2-training_prev2.height*1.5f*scale};
    DrawTextureEx(original_prev2, texture_position, 0, scale, RAYWHITE);

    // DRAWING THE IN BETWEEN TEXTURE 
    verify_img_construct(img_prev, nn, *scroll1);
    texture_position = (Vector2) {x+w/2-training_prev3.width/2*scale, y+h/2+training_prev3.height*0.5f*scale};
    UpdateTexture(training_prev3, img_prev->data);
    DrawTextureEx(training_prev3, texture_position, 0, scale, RAYWHITE);

    // RENDERING THE TOP SLIDER
    Vector2 rect_corner = (Vector2) {x+w/2-training_prev1.width*scale, y+h/2+training_prev1.height*2.0f*scale};
    Vector2 rect_size   = (Vector2) {2*training_prev1.width*scale, h*0.007};
    Vector2 slider_center = (Vector2) {
        (*scroll2*2*training_prev1.width*scale)+ x+w/2-training_prev1.width*scale,
        y+h/2+training_prev1.height*2.0f*scale};
    float slider_raduis = h*((float)19/WINDOW_HEIGHT);
    gym_slider_render(rect_corner, rect_size, slider_center, slider_raduis, slider_clicked2, scroll2);
    float font_size = h*(35.0f/WINDOW_HEIGHT);
    DrawText("Rate:", rect_corner.x-0.35*h, rect_corner.y-rect_size.y,  font_size, WHITE);

    // RENDERING THE BOTTOM SLIDER
    rect_corner = (Vector2) {x+w/2-training_prev1.width*scale, y+h/2+training_prev1.height*1.70f*scale};
    rect_size   = (Vector2) {2*training_prev1.width*scale, h*0.007};
    slider_center = (Vector2) {
        (*scroll1*2*training_prev1.width*scale)+ x+w/2-training_prev1.width*scale,
        y+h/2+training_prev1.height*1.70f*scale};
    slider_raduis = h*((float)19/WINDOW_HEIGHT);
    gym_slider_render(rect_corner, rect_size, slider_center, slider_raduis, slider_clicked1, scroll1);
    DrawText("Interpolation:", rect_corner.x-0.35*h, rect_corner.y-rect_size.y,  font_size, WHITE);
}

void render_single_frame(NN nn, float scroll)
{
    for (size_t y = 0; y < out_height; ++y) {
        for (size_t x = 0; x < out_width; ++x) {
            MAT_AT(NN_INPUT(nn), 0, 0) = (float)x/(out_width - 1);
            MAT_AT(NN_INPUT(nn), 0, 1) = (float)y/(out_height - 1);
            MAT_AT(NN_INPUT(nn), 0, 2) = scroll;

            nn_forward(nn);
            float activation = MAT_AT(NN_OUTPUT(nn), 0, 0);
            if (activation < 0) activation = 0;
            if (activation > 1) activation = 1;
            uint32_t bright = activation*255.f;
            uint32_t pixel = 0xFF000000|bright|(bright<<8)|(bright<<16);
            out_pixels[y*out_width + x] = pixel;
        }
    }   
}

void render_image_snapshot(NN nn, float scroll, const char* img_file_path)
{
    char out_file_path[256];
    assert(out_pixels != NULL);
    render_single_frame(nn, scroll);
    snprintf(out_file_path, sizeof(out_file_path), "upscaled/%s", get_file_name(img_file_path));
    check(!stbi_write_png(out_file_path, out_width, out_height, 4, out_pixels, out_width*sizeof(*out_pixels)),
          "could not save image %s\n", out_file_path);
    printf("INFO: Generated %s\n", out_file_path);
}

void render_ffmpeg_video(NN nn, float duration, const char* out_video_path)
{
    int wstatus, res;
    pid_t cpid, wpid;
    int pipefd[2];
    res = pipe(pipefd);
    check(res < 0, "pipe2() failure");

    cpid = fork();
    check(cpid < 0, "fork() failure");

    if (cpid == 0) {
        close(pipefd[WRITE_END]);
        res = dup2(pipefd[READ_END], STDIN_FILENO);
        check(res < 0, "dup2() failure");

        res = execlp("ffmpeg",
                     "ffmpeg",
                     "-loglevel", "verbose",
                     "-y",
                     "-f", "rawvideo",
                     "-pix_fmt", "rgba",
                     "-s", STR(out_width) "x" STR(out_height),
                     "-r", STR(FPS),
                     "-an",
                     "-i", "-",
                     "-c:v", "libx264",
                     out_video_path,
                     NULL);

        check(res < 0, "execlp() failure");
        close(pipefd[READ_END]);

        exit(EXIT_FAILURE);
    }

    close(pipefd[READ_END]);

    // ADDING SOME EASE-IN EASE-OUT LOGIC
    Segment segments[] = {
        {0, 0},
        {0, 1},
        {1, 1},
        {1, 0},
        {0, 0},
    };
    size_t no_segments = ARRAY_LEN(segments);
    // SENDING THE DATA TO THE FFMPEG STDING USING PIPES
    size_t frame_count = duration*FPS;
    for (size_t i = 0; i < frame_count; i++) {
        float a = (float)i/frame_count;
        float segment_len = 1.f/no_segments;
        size_t segment_index = a/segment_len;
        float segment_progress = a/segment_len - floorf(a/segment_len);
        float b = segments[segment_index].start + segment_progress*(segments[segment_index].end - segments[segment_index].start);

        render_single_frame(nn, sqrtf(b));
        size_t pixels_size = sizeof(*out_pixels)*out_width*out_height;
        int x = write(pipefd[WRITE_END], out_pixels, pixels_size);
        check (x < (int)pixels_size, "cant write %zu bytes only write %d", pixels_size, x);
    }

    close(pipefd[WRITE_END]);

    // INSPECTION OF THE CHILD PROCESS
    do {
        wpid = waitpid(cpid, &wstatus, WUNTRACED | WCONTINUED);
        check(wpid < 0, "waitpid() failure");

        if (WIFEXITED(wstatus)) {
            printf("exited,                     status=%d\n", WEXITSTATUS(wstatus));
        } else if (WIFSIGNALED(wstatus)) {
            printf("killed       by       signal       %d\n", WTERMSIG(wstatus));
        } else if (WIFSTOPPED(wstatus)) {
            printf("stopped       by      signal       %d\n", WSTOPSIG(wstatus));
        } else if (WIFCONTINUED(wstatus)) {
            printf("continued\n");
        }
    } while (!WIFEXITED(wstatus) && !WIFSIGNALED(wstatus));
    printf("INFO: Generate %s video successfully\n", out_video_path);
}

int main(int argc, char** argv)
{
    char* program = shift_args(&argc, &argv);
    check(argc <= 0, "Usage: %s <image_1> <image_2>\n", program); 

    char* img_file_path1 = shift_args(&argc, &argv);
    check(argc <= 0, "Usage: %s <image_1> <image_2>\n", program); 

    int img_width1, img_height1, img_comp1;
    uint8_t* img_pixels1 = (uint8_t*)stbi_load(img_file_path1, &img_width1, &img_height1, &img_comp1, 0);

    check(img_pixels1 == NULL, "Couldn't open file: %s\n", img_file_path1);
    check(img_comp1 != 1, "The image: %s is not grey scaled it has %d components\n", img_file_path1, img_comp1);

    int img_width2, img_height2, img_comp2;
    char* img_file_path2 = shift_args(&argc, &argv);
    uint8_t* img_pixels2 = (uint8_t*)stbi_load(img_file_path2, &img_width2, &img_height2, &img_comp2, 0);
    check(img_pixels2 == NULL, "Couldn't open file: %s\n", img_file_path2);
    check(img_comp2 != 1, "The image: %s is not grey scaled it has %d components\n", img_file_path2, img_comp2);

    Region temp = region_alloc_alloc(64*1024*1024);

    size_t rows = img_width1*img_height1 + img_height2*img_width2;
    size_t cols = arch[0] + arch[ARRAY_LEN(arch)-1];
    Mat t = mat_alloc(NULL, rows, cols);
    for (int y = 0; y < img_height1; y++) {
        for (int x = 0; x < img_width1; x++) {
            int i = y*img_width1+x;
            MAT_AT(t, i, 0) = (float)x/(img_width1-1);
            MAT_AT(t, i, 1) = (float)y/(img_height1-1);
            MAT_AT(t, i, 2) = 0.0f;
            MAT_AT(t, i, 3) = img_pixels1[i]/255.f;
        }
    }

    for (int y = 0; y < img_height2; y++) {
        for (int x = 0; x < img_width2; x++) {
            // THE INPUT HERE IS SHIFTED BY THE DIMENTION OF IMAGE 1
            int i = img_width1*img_height1+y*img_width2+x;
            MAT_AT(t, i, 0) = (float)x/(img_width2-1);
            MAT_AT(t, i, 1) = (float)y/(img_height2-1);
            MAT_AT(t, i, 2) = 1.0f;

            MAT_AT(t, i, 3) = img_pixels2[y*img_width2+x]/255.f;
        }
    }

    NN nn = nn_alloc(NULL, arch, ARRAY_LEN(arch));
    size_t epoch = 0;
    size_t max_epoch = 100000;
    Costs costs = {0};
    LayoutStack ls = {0};
    LayoutRect r = {0};
    bool paused = true;

    size_t batch_size = 28;
    size_t batch_per_frame = 103;
    Batch batch = {0};

    float scroll1 = 0.5f;
    // this is the scroll2;
    float rate = 1;
    bool slider_clicked1 = false;
    bool slider_clicked2 = false;

    srand(time(0));
    nn_rand(nn, -1, 1);

    SetConfigFlags(FLAG_WINDOW_RESIZABLE);
    InitWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "IMAGE_NN");;
    SetTargetFPS(60);

    // LET THIS ASSERTION FOR NOW
    // IT HELP ME IN RENDERING THE VERIFICATION SECTION
    // IN FUTURE I WILL REMOVE IT
    assert(img_width1 == img_width2 && img_height1 == img_height2);

    Image img_prev = GenImageColor(img_width1, img_height1, BLACK);
    Texture2D training_prev1  = LoadTextureFromImage(img_prev);
    Texture2D training_prev2  = LoadTextureFromImage(img_prev);
    Texture2D training_prev3  = LoadTextureFromImage(img_prev);

    original_img_construct(&img_prev, img_pixels1);
    Texture2D original_prev1 = LoadTextureFromImage(img_prev);

    original_img_construct(&img_prev, img_pixels2);
    Texture2D original_prev2 = LoadTextureFromImage(img_prev);

    while (!WindowShouldClose()) {
        // RESET THE NURAL NETWORK
        if(IsKeyPressed(KEY_R)) {
            paused = true;
            epoch = 0;
            nn_rand(nn, -1, 1);
            costs.count = 0;
        }

        // TAKE SNAP SHOT OF THE BOTTOM TEXTURE
        if (IsKeyPressed(KEY_U)) {
            time_t sec = time(0);
            struct tm* now = localtime(&sec);
            char buffer[256];

            check(strftime(buffer, sizeof(buffer), "%Y-%m-%d_%H-%M-%S.png", now) < 1,
                  "strftime() ==> Couldn't format the time");
            render_image_snapshot(nn, scroll1, buffer);
        }

        if (IsKeyPressed(KEY_SPACE)) paused = !paused;
        if (IsKeyPressed(KEY_P)) NN_PRINT(nn);
        if (IsKeyPressed(KEY_S)) TakeScreenshot("demos_screenshots/img2nn.png");
        if (IsKeyPressed(KEY_V)) render_ffmpeg_video(nn, 5, "transition.mp4");

        // LEARNING PROCESS
        for (size_t j = 0; j < batch_per_frame && epoch < max_epoch && !paused; j++) {
            nn_batch(&temp, &batch, nn, t, rate, batch_size);
           
            if (batch.end) {
                DA_APPEND(&costs, batch.cost);
                mat_shuffle(t);
                epoch++;
            }
        }    

        Color background_color = {.a =0xFF, .b =0x18, .g =0x18, .r =0x18};
        BeginDrawing();
        ClearBackground(background_color);

        int rw = GetRenderWidth();
        int rh = GetRenderHeight();

        size_t frame = rh*0.25;
        size_t gap = rh*0.03;

        layout_stack_push(&ls, rect_constructor(0, frame, rw, rh-2*frame), LO_HORZ, 3, gap);
            gym_nn_render(nn, layout_stack_slot(&ls));

            layout_stack_push(&ls, layout_stack_slot(&ls), LO_VERT, 2, gap);
            gym_heatmap_render(nn, layout_stack_slot(&ls), WEIGHT);
            gym_cost_render(costs, layout_stack_slot(&ls));
            layout_stack_pop(&ls);

        r = layout_stack_slot(&ls);
        float scale = r.h*8.f/WINDOW_HEIGHT;

        verification_render(&img_prev, nn,
                            r.x, r.y, r.w, r.h, scale,
                            &scroll1, &slider_clicked1,
                            &rate, &slider_clicked2,
                            training_prev1, training_prev2, training_prev3,
                            original_prev1, original_prev2);

        gym_status_line_render(r.h, rw, epoch, max_epoch, rate, costs.count > 0 ? costs.items[costs.count - 1] : 0, &temp);
        layout_stack_pop(&ls);
        EndDrawing();

        assert(ls.count == 0);
        region_reset(&temp);
    }
    
    CloseWindow();
    return 0;
}
