UNAME_S:=$(shell uname -s)
ifeq ($(UNAME_S),Linux)
CFLAGS=-I/usr/local/include -I/usr/include/eigen3 -I/usr/local/include/opencv4 -Wall -Wextra -std=c++1z
else ifeq ($(UNAME_S),Darwin)
CFLAGS=-I/usr/local/include -I/usr/local/include/eigen3 -I/usr/local/include/opencv4 -Wall -Wextra -std=c++1z
else
echo "Unknown platform"
endif

ifeq ($(BUILD), debug)
CFLAGS += -g -O0
else
CFLAGS += -O3 -DNDEBUG
endif

LDFLAGS = -L/usr/local/lib -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_imgcodecs

CC = clang++
EXE = "colorize"
TEST_EXE = "test"

all: src/main.cpp | output_directory
	$(CC) $(CFLAGS) $(LDFLAGS) src/main.cpp -o "bin/$(EXE)"

test: src/test_eigen.cpp | output_directory
	$(CC) `pkg-config --cflags eigen3` -o "bin/$(TEST_EXE)" src/test_eigen.cpp

output_directory:
	@mkdir -p bin

clean:
	rm -rf bin/
