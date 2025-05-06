# Author: Petr Maca \
Last revision: May 6 2025 \
TODO: replace with pattern rules; mkdir for folders 



TARGET = bin/JKMNet
CC = g++ 	# compiler
CPPFLAGS = -std=c++14 -Wall -pedantic -Iinclude -fopenmp # bad code warnings and include header folder
OBJ = obj/main.o # TODO
LDFLAGS := -Llib 


.PHONY: all clean

all: $(TARGET)

obj/main.o: src/main.cpp
	$(CC) -c $< -o obj/main.o $(CPPFLAGS)

obj/data_HB_1d.o: src/JKMNet.cpp
	$(CC) -c $< -o obj/JKMNet.o $(CPPFLAGS)

$(TARGET): $(OBJ)
	$(CC) -o $(TARGET) $(OBJ) $(CPPFLAGS) $(LDFLAGS)

clean:
	rm -r -f -v obj/*
	rm -f -v $(TARGET)