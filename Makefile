CC     := gcc
OUTDIR := out
CFLAGS := -Wextra -Wall
LINKS  := -lm
BINS   := simple_neoron gates xor

all: $(BINS)

%: %.c
	@$(CC) -o $(OUTDIR)/$@ $< $(CFLAGS) $(LINKS)

clean:
	rm -rf $(OUTDIR)/*
