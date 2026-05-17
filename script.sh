#!/usr/bin/bash

set -xe

gcc -Wall -Wextra -o out/nn nn.c -lm
