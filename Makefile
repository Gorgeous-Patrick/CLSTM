# Compiler and flags
CC = icx
CFLAGS = -O2 -Wall -Wextra -g3
LDFLAGS = -lopenblas -lm

# Source files
SRCS = multihead_attention.c

# Output binary
TARGET = multihead_attention
ADVI_RES = advi_results

# Default target
all: $(TARGET)

# Rule to build the target
$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) $(SRCS) -o $@ $(LDFLAGS)

$(ADVI_RES): $(TARGET)
	advisor --collect=roofline --project-dir=./$(ADVI_RES) -- ./$(TARGET)

report.html: $(ADVI_RES)
	advisor --report=roofline --project-dir=./$(ADVI_RES) --report-output=report.html

clean:
	rm -rf $(TARGET) $(ADVI_RES) config report.html
