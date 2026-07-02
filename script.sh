#!/usr/bin/bash

set -xe

CFLAGS="-O3 -Wall -Wextra -ggdb -I./thirdparty/ -I. `pkg-config --cflags raylib`"
LIBS="`pkg-config --libs raylib` -lglfw -ldl -lpthread -lrt -lX11 -lGL -lm"

# gcc $CFLAGS -o out/img-storage img-storage.c $LIBS
gcc $CFLAGS -o out/layout layout.c $LIBS
