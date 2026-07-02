#include <assert.h>
#include <raylib.h>
#include <stdio.h>

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
} Layout;

Layout layout_constructor(LayoutRect rect, LayoutOrient orient, size_t count)
{
    return (Layout) {.rect = rect, .orient = orient, .count = count};
}

typedef struct {
    Layout* items;
    size_t count;
    size_t capacity;
} LayoutStack;

void layout_stack_push(LayoutStack* ls, Layout l)
{
    DA_APPEND(ls, l);
}

void layout_stack_pop(LayoutStack* ls)
{
    assert(ls->count > 0);
    ls->count--;
}


LayoutRect layout_slot(Layout *l)
{
    LayoutRect r = {0};
    switch(l->orient) {
    case LO_HORZ:
        r.w = l->rect.w/l->count;
        r.h = l->rect.h;
        r.x = l->rect.x+l->i*r.w;
        r.y = l->rect.y;
        break;
    case LO_VERT:
        r.w = l->rect.w;
        r.h = l->rect.h/l->count;
        r.x = l->rect.x;
        r.y = l->rect.y+l->i*r.h;
        break;
    default:
        assert(0 && "Unreachable");
    }
    l->i++;

    return r;
}

LayoutRect layout_stack_slot(LayoutStack* ls)
{
    assert(ls->count > 0);
    return layout_slot(&ls->items[ls->count-1]);
}

void widget(LayoutRect rect, Color c)
{
    DrawRectangle(rect.x, rect.y, rect.w, rect.h, c);
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
        layout_stack_push(&ls, layout_constructor(rect_constructor(0, 0, w, h), LO_HORZ, 3));
            widget(layout_stack_slot(&ls), BLUE);
            widget(layout_stack_slot(&ls), GREEN);
                layout_stack_push(&ls, layout_constructor(layout_stack_slot(&ls), LO_VERT, 3));
                widget(layout_stack_slot(&ls), RED);
                widget(layout_stack_slot(&ls), WHITE);
                widget(layout_stack_slot(&ls), BLACK);
                layout_stack_pop(&ls);
            layout_stack_pop(&ls);
        EndDrawing();

        // PREVENTION OF MEMORY LEAK
        assert(ls.count == 0);
    }
    CloseWindow();
    return 0;
}
