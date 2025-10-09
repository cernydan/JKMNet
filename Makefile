# Author: Petr Maca
# Last revision: Oct 6 2025
# TODO: replace with pattern rules; mkdir for folders

TARGET = bin/JKMNet
CC = g++
CPPFLAGS = -std=c++20 -Wall -pedantic -Iinclude -fopenmp
LDFLAGS := -Llib

# Object files
OBJ = obj/main.o \
      obj/JKMNet.o \
      obj/MLP.o \
      obj/Layer.o \
      obj/Data.o \
      obj/Metrics.o \
      obj/CNNLayer.o \
      obj/PSO.o \
      obj/HyperparamObjective.o \
      obj/HyperparamOptimizer.o

.PHONY: all clean run dirs

all: dirs $(TARGET)

dirs:
	@mkdir -p obj bin

obj/main.o: src/main.cpp
	$(CC) -c $< -o $@ $(CPPFLAGS)

obj/JKMNet.o: src/JKMNet.cpp
	$(CC) -c $< -o $@ $(CPPFLAGS)

obj/MLP.o: src/MLP.cpp
	$(CC) -c $< -o $@ $(CPPFLAGS)

obj/Layer.o: src/Layer.cpp
	$(CC) -c $< -o $@ $(CPPFLAGS)

obj/Data.o: src/Data.cpp
	$(CC) -c $< -o $@ $(CPPFLAGS)

obj/Metrics.o: src/Metrics.cpp
	$(CC) -c $< -o $@ $(CPPFLAGS)

obj/CNNLayer.o: src/CNNLayer.cpp
	$(CC) -c $< -o $@ $(CPPFLAGS)

obj/PSO.o: src/PSO.cpp
	$(CC) -c $< -o $@ $(CPPFLAGS)

obj/HyperparamObjective.o: src/HyperparamObjective.cpp
	$(CC) -c $< -o $@ $(CPPFLAGS)

obj/HyperparamOptimizer.o: src/HyperparamOptimizer.cpp
	$(CC) -c $< -o $@ $(CPPFLAGS)

$(TARGET): $(OBJ)
	$(CC) -o $@ $(OBJ) $(CPPFLAGS) $(LDFLAGS)

run: $(TARGET)
	./$(TARGET)

clean:
	rm -rfv obj/* bin/JKMNet
