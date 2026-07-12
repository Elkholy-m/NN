#!/usr/bin/bash

set -xe

CFLAGS="-O3 -Wall -Wextra -ggdb -I./thirdparty/ -I. `pkg-config --cflags raylib`"
LIBS="`pkg-config --libs raylib` -lglfw -ldl -lpthread -lrt -lX11 -lGL -lm"

gcc $CFLAGS -o out/xor    demos/xor.c    $LIBS
gcc $CFLAGS -o out/adder  demos/adder.c  $LIBS
gcc $CFLAGS -o out/img2nn demos/img2nn.c $LIBS
