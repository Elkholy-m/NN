#include <time.h>
#define NN_IMPLEMENTATION
#include "nn.h"

#define OLIVEC_IMPLEMENTATION
#include "olive.c"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"


#define IMG_WIDTH   800
#define IMG_HEIGHT  600

uint32_t pixels[IMG_WIDTH*IMG_HEIGHT];

void nn_render(Olivec_Canvas img, NN nn)
{
    uint32_t frame_thic = 10;
    uint32_t neuron_rad = 25;

    uint32_t background_color = 0xFF181818;
    uint32_t frame_color      = 0xFFAAAAAA;
    uint32_t low__color       = 0x00FF00FF;
    uint32_t high_color       = 0x0000FF00;
    
    olivec_fill(img, background_color);
    olivec_frame(img, 0, 0, img.width-1, img.height-1, frame_thic, frame_color);

    size_t border_vpad = 50;
    size_t border_hpad = 50;
    size_t nn_x = img.width -2*border_hpad;
    size_t nn_y = img.height-2*border_vpad;
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

                    
                    uint32_t alpha = floorf(255.f*sigmoidf(MAT_AT(nn.ws[i], j, k)));
                    uint32_t connection_color = low__color|0xFF000000;
                    olivec_blend_color(&connection_color, (alpha<<24) | high_color);
                    olivec_line(img, cx1, cy1, cx2, cy2, connection_color);
                }
            }
            if (i == 0) {
                olivec_circle(img, cx1, cy1, neuron_rad, 0xFF808080);
            } else {
                uint32_t alpha = floorf(255.f*sigmoidf(MAT_AT(nn.bs[i-1], 0, j)));
                uint32_t neuron_color = low__color|0xFF000000;
                olivec_blend_color(&neuron_color, (alpha<<24) | high_color);
                olivec_circle(img, cx1, cy1, neuron_rad, neuron_color);
            }
        }
    }

}

int main(void)
{
    srand(90);
    size_t arch[] = {4, 4, 2, 1};
    NN nn = nn_alloc(arch, ARRAY_LEN(arch));
    nn_rand(nn, -1, 1);
    NN_PRINT(nn);

    Olivec_Canvas img = olivec_canvas(pixels, IMG_WIDTH, IMG_HEIGHT, IMG_WIDTH);
    nn_render(img, nn);

    const char* filename = "nn.png";
    int res = stbi_write_png(filename, img.width, img.height, 4, pixels, img.stride*sizeof(uint32_t));
    if (!res) {
        printf("ERROR: couldn't save file %s.\n", filename);
        return 1;
    }
    printf("Save file successfully %s.\n", filename);

    return 0;
}
