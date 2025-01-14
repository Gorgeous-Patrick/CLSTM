# Compiler and flags
CC = gcc
CFLAGS = -O2 -Wall -Wextra
LDFLAGS = -lopenblas -lm

# Source files
SRCS = multihead_attention.c

# Output binary
TARGET = multihead_attention

# Default target
all: $(TARGET)

# Rule to build the target
$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) $(SRCS) -o $@ $(LDFLAGS)

clean:
	rm -f $(TARGET)