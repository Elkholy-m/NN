#include <assert.h>
#include <raylib.h>
#include <stdio.h>
#include <stdlib.h>

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

#define FACTOR  80
#define WIDTH   16*FACTOR
#define HEIGHT  9*FACTOR

typedef struct {
    size_t x;
    size_t y;
    size_t w;
    size_t h;
} LayoutRect;

LayoutRect rect_constructor(size_t x, size_t y, size_t w, size_t h)
{
    return (LayoutRect) {.x = x, .y = y, .w = w, .h = h};
}

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

Layout layout_constructor(LayoutRect rect, LayoutOrient orient, size_t count, size_t gap)
{
    return (Layout) {.rect = rect, .orient = orient, .count = count, .gap = gap};
}

typedef struct {
    Layout* items;
    size_t count;
    size_t capacity;
} LayoutStack;

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
        fprintf(stderr, "%s:%d:\tERROR: Widget overfloaw\n", file, line);
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
        assert(0 && "Unreachable");
    }
    l->i++;

    return r;
}

LayoutRect layout_stack_slot_imp(LayoutStack* ls, const char* file, const int line)
{
    assert(ls->count > 0);
    return layout_slot(&ls->items[ls->count-1], file, line);
}

#define layout_stack_slot(ls) layout_stack_slot_imp(ls, __FILE__, __LINE__)


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

int main(void)
{
    LayoutStack ls = {0};
    SetConfigFlags(FLAG_WINDOW_RESIZABLE);
    InitWindow(WIDTH, HEIGHT, "WIDGETS");;
    SetTargetFPS(60);

    while(!WindowShouldClose()) {
        BeginDrawing();
        ClearBackground(BLACK);
        int w = GetRenderWidth();
        int h = GetRenderHeight();
        size_t frame = h*0.1;
        size_t gap = h*0.015;
        layout_stack_push(&ls, rect_constructor(0, frame, w, h-2*frame), LO_HORZ, 3, gap);
            widget(layout_stack_slot(&ls), BLUE);
            widget(layout_stack_slot(&ls), GREEN);
            layout_stack_push(&ls, layout_stack_slot(&ls), LO_VERT, 3, gap);
                layout_stack_push(&ls, layout_stack_slot(&ls), LO_HORZ, 2, gap);
                    layout_stack_push(&ls, layout_stack_slot(&ls), LO_VERT, 2, gap);
                    widget(layout_stack_slot(&ls), SKYBLUE);
                        layout_stack_push(&ls, layout_stack_slot(&ls), LO_HORZ, 2, gap);
                        widget(layout_stack_slot(&ls), SKYBLUE);
                        widget(layout_stack_slot(&ls), SKYBLUE);
                        layout_stack_pop(&ls);
                    layout_stack_pop(&ls);
                widget(layout_stack_slot(&ls), RED);
                layout_stack_pop(&ls);
                layout_stack_push(&ls, layout_stack_slot(&ls), LO_HORZ, 3, gap);
                widget(layout_stack_slot(&ls), YELLOW);
                widget(layout_stack_slot(&ls), YELLOW);
                widget(layout_stack_slot(&ls), YELLOW);
                layout_stack_pop(&ls);
                widget(layout_stack_slot(&ls), MAGENTA);
                layout_stack_pop(&ls);
            layout_stack_pop(&ls);
        EndDrawing();

        // PREVENTION OF MEMORY LEAK
        assert(ls.count == 0);
    }
    CloseWindow();
    return 0;
}
