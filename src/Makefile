# Makefile for CUDA Tic-Tac-Toe
CXX = nvcc
TARGET = tic_tac_toe.exe
SRC = tic_tac_toe.cu

CXXFLAGS = --std=c++17 -Wno-deprecated-gpu-targets
LDFLAGS = -lcuda -lcudart

all: clean build

build: $(SRC)
	$(CXX) $(SRC) -o $(TARGET) $(CXXFLAGS) $(LDFLAGS)

run:
	./$(TARGET) $(ARGS)

clean:
	rm -f $(TARGET) output*.txt
