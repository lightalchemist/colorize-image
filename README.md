# Colorization using Optimization

This repository contains an implementation of the algorithm described in 
A. Levin D. Lischinski and Y. Weiss Colorization using Optimization. ACM Transactions on Graphics, Aug 2004
for coloring grayscale images.

## Compile

### Requirements

1. [OpenCV](https://opencv.org/)
2. [Eigen3](https://eigen.tuxfamily.org/)
3. [CMake](https://cmake.org/) optionally for generating Makefile.

### Compiling with CMake

This project includes a `CMakeLists.txt` to help locate the required libraries and their header files and generate the Makefile. If the above requirements are met, the following will generate the binary `colorize`.

```bash
mkdir build
cd build
cmake ..
make
```

### Compiling without CMake

Edit the `Makefile` under the root directory so that the compiler can find the required libraries and header files. Then run `make` in the terminal to compile the project.

```bash
make
```

## Run

The compiled binary is located in the `bin/` directory.
It takes in a grayscale image, an image with line scribbles, and the path to output the colored image

```bash
./colorize ../data/man.bmp ../data/man_marked.bmp ../data/man_res.png
```

| Original                       | Scribbles                              | Result                           |
| :-------------:                | :-------------:                        | :-----:                          |
| ![Original](data/man.bmp)      | ![Scribbles](data/man_marked.bmp)      | ![Result](data/man_res.png)      |
| ![Original](data/casual.bmp)   | ![Scribbles](data/casual_marked.bmp)   | ![Result](data/casual_res.png)   |
| ![Original](data/example.bmp)  | ![Scribbles](data/example_marked.bmp)  | ![Result](data/example_res.png)  |
| ![Original](data/example3.bmp) | ![Scribbles](data/example3_marked.bmp) | ![Result](data/example3_res.png) |

## Sample images and markings

Sample images can be found in the `data` directory. 
Images ending with `_marked.bmp` correspond to color scribbles
and those ending with `_res.png` are sample results.
