# edgeMatching
> Pyramid search matching based on edge gradient cosine similarity

This project provides functionality for template matching using OpenCV. It includes shared libraries for template creation and finding, as well as executables for training and inference.

## Features

- **Compact canonical model**: training extracts only the canonical `0°` edge features at each pyramid level. Search angles are generated and cached when the model is loaded, rather than being redundantly calculated and stored during training.
- **Partial visibility**: targets cut by an image boundary can be matched. Only in-image feature points contribute to the score, subject to a configurable minimum visible ratio; drawing is clipped to the image.
- **Multi-scale matching**: a discrete scale range such as `0.8` to `1.2` can be searched without retraining the template.
- **Multiple templates**: several model files can be searched in one call. Every result includes its template ID, scale, score, pose, and visible ratio.
- **Readable visualization**: the result index, template ID, score, and scale are drawn parallel to a bounding-box edge. Contours, boxes, and labels are clipped safely at image boundaries.

The similarity is defined as:

![formula1.svg](assert/.md/formula1.svg)

Split into normalized vectors:

![formula1.svg](assert/.md/formula2.svg)

Then, the cosine similarity becomes:

![formula1.svg](assert/.md/formula3.svg)

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

  The complete command-line form is:

  ```bash
  ./inference [search_image] [model1.json model2.json ...] \
    [--scale-min N] [--scale-max N] [--scale-step N] \
    [--min-visible-ratio N] [--output FILE]
  ```

  For example, search two templates from `0.8x` through `1.2x`, allow targets with at least half of their template features visible, and choose the result path:

  ```bash
  ./inference ../assert/src.bmp model_a.json model_b.json \
    --scale-min 0.8 --scale-max 1.2 --scale-step 0.05 \
    --min-visible-ratio 0.5 --output result.png
  ```

  With no arguments, the legacy defaults remain in effect: `../assert/src.bmp`, `./model.json`, scale `1.0`, and full visibility. Paths are resolved from the process working directory, so explicit paths are recommended for automated use.

  Each console result contains:

  ```text
  [index] template_id=... x=... y=... angle=... score=... scale=... visible=...
  ```

## Algorithm Overview

### Template creation

The input template and mask must be non-empty, equally sized 8-bit images. Color inputs are converted to grayscale. The implementation creates a Gaussian pyramid, extracts edge points and normalized gradient directions inside the mask, converts coordinates to a template-center origin, and stores one canonical `0°` feature set per level.

JSON models therefore contain only the canonical features. During JSON or binary loading, the matcher uses `angle_start`, `angle_end`, and `angle_step` to rotate point coordinates and gradient vectors into the required angle cache. This reduces training work and model storage from approximately `O(levels * angles * features)` to `O(levels * features)`. Existing model loading remains supported; a historical binary model that already contains angle data must not be expanded twice.

### Coarse-to-fine search

The search image is converted into a gradient pyramid. Candidates are found at the coarsest usable level and refined through finer levels in position and angle. The score is based on the cosine similarity between template and image gradient directions. Nearby or strongly overlapping candidates are removed before returning at most the requested number of matches.

For partial targets, transformed points outside the image are skipped safely and the score is normalized by the number of visible points. `min_visible_ratio` rejects candidates supported by too little of the template. Result drawing similarly clips the transformed contour and rotated bounding box to the image.

Multi-scale matching evaluates discrete scales from `scale_min` through `scale_max` using `scale_step`, then performs cross-scale suppression. A result's `scale` is the target size relative to the trained template. Multi-template matching reuses the public search interface for each model, records `template_id`, merges candidates, and performs final overlap suppression.

## Parameter Guidance

| Parameter | Valid range / suggested start | Effect |
| --- | --- | --- |
| `angle_start`, `angle_end` | start `<=` end; restrict to the physically possible range | Wider ranges increase angle candidates and runtime. |
| `angle_step` | finite and `> 0`; commonly `1°` | Smaller steps improve angular resolution but increase load-time cache and search cost. |
| `min_score` | `[0, 1]`; start around `0.7` | Raise it to reduce false positives; lower it for noisy or partially visible edges. |
| `greediness` | `(0, 1]`; start around `0.9` | Controls early rejection of weak candidates. |
| `max_overlap` | `[0, 1]`; start around `0.5` | Lower values suppress nearby duplicate matches more aggressively. |
| `scale_min`, `scale_max` | `0 < min <= max`; use `1,1` for legacy behavior | Defines the supported target-size range. |
| `scale_step` | finite and `> 0`; start around `0.05` | Smaller values improve scale resolution at roughly proportional runtime cost. |
| `min_visible_ratio` | `(0, 1]`; use `1.0` for complete targets, `0.5` as a partial-target starting point | Lower values accept more boundary truncation but increase false-positive risk. |

The approximate brute-force cost is `O(positions * angles * scales * features)`. Keep angle and scale ranges as narrow as the application permits, then tune `min_score` and `min_visible_ratio` on representative normal and boundary samples.

## Automated Tests

CTest builds a deterministic synthetic regression that covers `0.8x`, `1.0x`, and `1.2x` matching, a target partially outside the right image boundary, two-template matching with template IDs, canonical-only training data, and visualization without an out-of-bounds failure.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

For a real-image smoke test without overwriting the repository model, run training and inference from a separate directory and pass explicit paths.

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
