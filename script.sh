#!/usr/bin/bash

set -xe

CFLAGS="-O3 -Wall -Wextra -ggdb -I./thirdparty/ -I. `pkg-config --cflags raylib`"
LIBS="-lm `pkg-config --libs raylib` -lglfw -ldl -lpthread -lrt -lX11 -lGL"

# gcc $CFLAGS -o out/adder_gen adder_gen.c $LIBS
# gcc $CFLAGS -o out/xor_gen xor_gen.c $LIBS
# gcc $CFLAGS -o out/gym gym.c $LIBS
gcc $CFLAGS -o out/img-storage img-storage.c $LIBS
