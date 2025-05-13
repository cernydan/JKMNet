# Author: Petr Maca \
Last revision: May 6 2025 \
TODO: replace with pattern rules; mkdir for folders 

TARGET = bin/JKMNet
CC = g++ 	# compiler
CPPFLAGS = -std=c++20 -Wall -pedantic -Iinclude -Iinclude/eigen-3.4 -fopenmp # bad code warnings and include header folder
OBJ = obj/main.o obj/JKMNet.o  obj/MLP.o obj/Layer.o # TODO
LDFLAGS := -Llib 

.PHONY: all clean

all: $(TARGET)

obj/main.o: src/main.cpp
	$(CC) -c $< -o obj/main.o $(CPPFLAGS)

obj/JKMNet.o: src/JKMNet.cpp
	$(CC) -c $< -o obj/JKMNet.o $(CPPFLAGS)

obj/MLP.o: src/MLP.cpp
	$(CC) -c $< -o obj/MLP.o $(CPPFLAGS)

obj/Layer.o: src/Layer.cpp
	$(CC) -c $< -o obj/Layer.o $(CPPFLAGS)

$(TARGET): $(OBJ)
	$(CC) -o $(TARGET) $(OBJ) $(CPPFLAGS) $(LDFLAGS)

clean:
	rm -r -f -v obj/*
	rm -f -v $(TARGET)