CC     := gcc
OUTDIR := out
CFLAGS := -Wextra -Wall
LINKS  := -lm
BINS   := simple_neoron gates xor adder
SCRDIR := scratch

all: $(BINS)

%: $(SCRDIR)/%.c
	@$(CC) -o $(OUTDIR)/$@ $< $(CFLAGS) $(LINKS)

clean:
	rm -rf $(OUTDIR)/*
