cc = clang++

all: linux

mac:
	$(cc) -Wall -Wextra -std=c++1z -I/usr/local/include/eigen3 -I/usr/local/include -I/usr/local/include/opencv4 -L/usr/local/lib -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_imgcodecs main.cpp -o colorize

linux:
	$(cc) -Wall -Wextra `pkg-config --cflags eigen3` main.cpp -o colorize

clean:
	rm -f colorize
