#!/usr/bin/bash

set -xe

# gcc -Wextra -o out/xor xor.c -lm
gcc -Wextra -o out/adder_2 adder.c -lm
# gcc  -Wextra -o out/dump_nn dump_nn.c -lm
