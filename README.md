# Colorization using Optimization
---------------------------------

This repository contains an implementation of the algorithm described in 
A. Levin D. Lischinski and Y. Weiss Colorization using Optimization. ACM Transactions on Graphics, Aug 2004
for coloring grayscale images.

## Compile

### Requirements

1. [OpenCV](https://opencv.org/)
2. [Eigen3](https://eigen.tuxfamily.org/)
3. [CMake](https://cmake.org/) for generating the Makefile.

Run the following commands under the root directory of the project to compile the project in the terminal.

```bash
mkdir build
cd build
cmake ..
make
```

## Run

The compiled binary is located in the `build/bin` directory

The compiled binary takes in a grayscale image, an image with line scribbles, and the path to output the colored image. The following 

```bash
cd bin
./colorize ../data/man.bmp ../data/man_marked.bmp ../data/man_res.png
```

## Sample images and markings

Sample images can be found in the `data` directory. 
Images ending with `_marked.bmp` correspond to color scribbles
and those ending with `_res.png` are sample results.
See the example above
