# edgeMatching
> Pyramid search matching based on edge gradient cosine similarity

This project provides functionality for template matching using OpenCV. It includes shared libraries for template creation and finding, as well as executables for training and inference.



> Performance Tests (CPU I7-10700)

| Test  | Resolution | Metrics                                                      | Image                                              |
| ----- | ---------- | ------------------------------------------------------------ | -------------------------------------------------- |
| Test1 | 3648*3648  | **Execution Time**:<br/>207ms<br/><br />**Inference Parameters**:<br/>f_angle_start=-5<br/>f_angle_stop=5<br/>f_min_score=0.7<br/>f_matche_numbers=200<br />f_max_overlap=0.5 | ![src2_result](./assert/.md/src2_result.png)       |
| Test2 | 4024*3036  | **Execution Time**:<br/>109ms<br/><br />**Inference Parameters**:<br/>f_angle_start=-180<br/>f_angle_stop=180<br/>f_min_score=0.7<br/>f_matche_numbers=200<br />f_max_overlap=0.5 | ![src1_result](./assert/.md/src1_result.png)       |
| Test3 | 2448*2048  | **Execution Time**:<br/>64ms<br/><br />**Inference Parameters**:<br/>f_angle_start=-180<br/>f_angle_stop=180<br/>f_min_score=0.7<br/>f_matche_numbers=200<br />f_max_overlap=0.5 | ![src_result](./assert/.md/src_result.png)         |
| Test4 | 2592*1944  | **Execution Time**:<br/>41ms<br/><br />**Inference Parameters**:<br/>f_angle_start=-180<br/>f_angle_stop=180<br/>f_min_score=0.7<br/>f_matche_numbers=200<br />f_max_overlap=0.5 | ![src10_2_result](./assert/.md/src10_2_result.png) |
| Test5 | 4096*3000  | **Execution Time**:<br/>160ms<br/><br />**Inference Parameters**:<br/>f_angle_start=-180<br/>f_angle_stop=180<br/>f_min_score=0.85<br/>f_matche_numbers=200<br />f_max_overlap=0.5 | ![src1_result](./assert/.md/src3_result.png)       |

> [!NOTE]  
>
> Some pictures are sourced from [NCC](https://github.com/DennisLiu1993/Fastest_Image_Pattern_Matching.git)



## Project Structure

The project directory structure is as follows:

```bash
.
├── 3rdparty
│   └── opencv # OpenCV dependencies
│       ├── bin # Executable binaries
│       ├── include # OpenCV headers
│       ├── lib # OpenCV libraries
│       └── share # OpenCV share files
├── assert # Test images for matching
├── build_opencv_with_contrib.sh # Script to build OpenCV with contrib modules
├── CMakeLists.txt # CMake build configuration file
├── include # Project header files
│   ├── FindTemplateV1.h
│   ├── MakeTemplateV1.h
│   ├── ROI.h
│   ├── Timer.h
│   └── Type.h
├── inference.cpp # Inference executable source code
├── LICENSE
├── README.md
├── src # Source code for the libraries
│   ├── FindTemplateV1.cpp
│   └── MakeTemplateV1.cpp
└── train.cpp # Training executable source code
```

## Prerequisites

Ensure that you have the following installed in the `Linux`:

- `CMake` (version 3.10 or higher)
- `g++` compiler (with support for C++11)
- `OpenCV 4.7.0 `(with contrib modules, optionally)

## Building OpenCV

To build OpenCV with contrib modules, use the provided `build_opencv_with_contrib.sh` script.

if you **cmake** `OpenCV 4.7.0` with contrib modules failed, you can download `.cache.zip` file from the following link:
https://wwyn.lanzout.com/iB7Mb34k8kwd and extract it to the `your_opencv_source_dir/.cache` directory.

### Steps to build OpenCV:

1. Open a terminal and navigate to the project root directory.
2. Run the script:

   ```bash
   ./build_opencv_with_contrib.sh

This will download and build `OpenCV4.7.0` along with the contrib modules, placing the necessary files under the `3rdparty/opencv` directory.

## Using CMake to Build the Project

1. Create a build directory:

   ```bash
   mkdir build
   cd build
   ```

2. Run `CMake` to configure the project:

   ```bash
   cmake ..
   ```

3. Build the project using `make`:

   ```bash
   make -j$(nproc)
   ```

### Running the Executables

After building the project, you can run the executables:

- To run the `training` executable:

  ```bash
  ./train
  ```

  or

  ```bash
  ./train ../assert/m1.png
  ```

- To run the `inference` executable:

  ```bash
  ./inference
  ```

  or

  ```bash
  ./inference ../assert/src.bmp
  ```

## Compiling with g++

If you prefer to compile using `g++` directly instead of using CMake, you can compile the source code manually.

1. Compile the libraries:

   - For compiling the shared library `libMakeTemplateV1.so`

   ```bash
   g++ -std=c++11 -O3 -fopenmp -fPIC -march=native -msse -msse2 -msse3 -msse4 -mavx -o libMakeTemplateV1.so -shared src/MakeTemplateV1.cpp \
   -I./include \
   -I./3rdparty/opencv/include \
   -L./3rdparty/opencv/lib \
   -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs -lopencv_calib3d \
   -pthread
   ```

   - For compiling the shared library `libFindTemplateV1.so`

   ```bash
   g++ -std=c++11 -O3 -fopenmp -fPIC -march=native -msse -msse2 -msse3 -msse4 -mavx -o libFindTemplateV1.so -shared src/FindTemplateV1.cpp \
   -I./include \
   -I./3rdparty/opencv/include \
   -L./3rdparty/opencv/lib \
   -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs -lopencv_calib3d \
   -pthread
   ```

2. Compile the executables:

   - For the `train` executable

   ```bash
   g++ train.cpp src/MakeTemplateV1.cpp -std=c++11 -O3 -fopenmp -Iinclude -I3rdparty/opencv/include -I3rdparty/opencv/include/opencv4 -L3rdparty/opencv/lib -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs -lopencv_features2d -lopencv_flann -lopencv_calib3d -lopencv_videoio -lopencv_ximgproc -lopencv_xfeatures2d -o train
   ```

   - For the `inference` executable

   ```bash
   g++ inference.cpp src/FindTemplateV1.cpp -std=c++11 -O3 -fopenmp -Iinclude -I3rdparty/opencv/include -I3rdparty/opencv/include/opencv4 -L3rdparty/opencv/lib -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs -lopencv_features2d -lopencv_flann -lopencv_calib3d -lopencv_videoio -lopencv_ximgproc -lopencv_xfeatures2d -o inference
   ```

### Running the Executables

After compilation, you can run the executables as follows:

- To run the `training` executable:

  ```bash
  ./train assert/m1.png
  ```

- To run the `inference` executable:

  ```bash
  ./inference assert/src.bmp
  ```

### Install the Libraries and Header

To install the libraries and header files to the system directories:

1. Run the following command to install:

   ```bash
   make install
   ```

This will copy the shared libraries and headers to the appropriate system directories.

- The shared libraries will be installed in the `publish/lib` directory.
- The executables will be installed in the `publish/bin` directory.
- The header files will be installed in the `publish/include/shapeMatch` directory.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
