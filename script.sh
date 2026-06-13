#!/usr/bin/bash

set -xe

CFLAGS="-O3 -Wall -Wextra -ggdb -I./thirdparty/ -I. `pkg-config --cflags raylib`"
LIBS="-lm `pkg-config --libs raylib` -lglfw -ldl -lpthread -lrt -lX11 -lGL"

# gcc $CFLAGS -o out/xor xor.c $LIBS
gcc $CFLAGS -o out/adder_2 adder.c $LIBS
# gcc  -Wextra -o out/dump_nn dump_nn.c -lm
