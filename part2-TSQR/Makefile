# Case studies - Q2 - Makefile for TSQR (Seagull, mpicc + MKL)

CC      = mpicc
CFLAGS  = -O2 -Wall -I$(MKLROOT)/include

LDFLAGS = -L$(MKLROOT)/lib/intel64 \
          -lmkl_intel_lp64 -lmkl_core -lmkl_sequential \
          -lpthread -lm -ldl

TARGET  = TSQR_test
SRCS    = TSQR.c
OBJS    = $(SRCS:.c=.o)

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $<

run: $(TARGET)
	mpirun -np 4 ./$(TARGET)

clean:
	rm -f $(TARGET) $(OBJS)
