# edgeMatching

> 基于边缘梯度余弦相似度的金字塔模板匹配

本项目使用 OpenCV 实现工业场景中的边缘形状模板制作与匹配，提供模板制作库、模板匹配库，以及 `train` 和 `inference` 可执行程序。

## 功能特性

- **紧凑的 canonical 模型**：训练时每层金字塔只提取 `0°` canonical 边缘特征；模型加载时根据角度配置生成并缓存搜索角度，避免训练阶段重复旋转和存储。
- **部分可见目标**：目标被图像边界截断时仍可匹配。评分只使用位于图像内的特征点，并由 `min_visible_ratio` 控制最低可见比例；绘制结果会自动裁剪到图像范围。
- **多尺度匹配**：无需重新训练模板即可搜索离散尺度范围，例如 `0.8` 至 `1.2`。
- **多模板匹配**：一次调用可加载多个模型文件；结果包含 `template_id`、位置、角度、尺度、得分和可见比例。
- **清晰的结果可视化**：沿外包围矩形边缘平行绘制索引号、模板 ID、得分和尺度；轮廓、矩形和文字均安全裁剪。

## 相似度原理

相似度计算公式如下：

![formula1.svg](assert/.md/formula1.svg)

将梯度拆分为归一化向量：

![formula1.svg](assert/.md/formula2.svg)

因此余弦相似度为：

![formula1.svg](assert/.md/formula3.svg)

## 目录结构

```bash
.
├── 3rdparty
│   └── opencv # OpenCV 依赖
├── assert # 匹配测试图像
├── build_opencv_with_contrib.sh # 构建带 contrib 模块的 OpenCV 脚本
├── CMakeLists.txt # CMake 构建配置
├── docs
│   └── template_matching_algorithm.md # 模板算法原理说明
├── include # 项目头文件
│   ├── FindTemplateV1.h
│   ├── MakeTemplateV1.h
│   ├── ROI.h
│   ├── Timer.h
│   └── Type.h
├── inference.cpp # 匹配程序源码
├── LICENSE
├── README.md
├── src # 库源码
│   ├── FindTemplateV1.cpp
│   └── MakeTemplateV1.cpp
├── tests
│   └── test_matching.cpp # 回归测试
└── train.cpp # 训练程序源码
```

## 环境要求

请在 `Linux` 环境安装：

- `CMake` 3.10 或更高版本
- 支持 C++11 的 `g++`
- `OpenCV 4.7.0`（可选 `contrib` 模块）

如需构建 OpenCV，请运行：

```bash
./build_opencv_with_contrib.sh
```

如果使用 `cmake` 构建 `OpenCV 4.7.0` 失败，可下载 `.cache.zip` 并解压到 `your_opencv_source_dir/.cache`：

https://wwyn.lanzout.com/iB7Mb34k8kwd

## CMake 构建

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
cmake --build build --parallel
```

## 程序使用

训练程序示例：

```bash
./train
./train ../assert/m1.png
```

匹配程序示例：

```bash
./inference
./inference ../assert/src.bmp
```

完整命令行格式如下：

```bash
./inference [search_image] [model1.json model2.json ...] \
  [--scale-min N] [--scale-max N] [--scale-step N] \
  [--min-visible-ratio N] [--output FILE]
```

以下命令加载两个模板，在 `0.8x` 至 `1.2x` 范围搜索，并允许至少一半特征可见：

```bash
./inference ../assert/src.bmp model_a.json model_b.json \
  --scale-min 0.8 --scale-max 1.2 --scale-step 0.05 \
  --min-visible-ratio 0.5 --output result.png
```

不传参数时，默认使用 `../assert/src.bmp`、`./model.json`、尺度 `1.0` 和完整可见目标。路径相对于进程工作目录解析，自动化运行时建议显式指定路径。

控制台结果格式为：

```text
[index] template_id=... x=... y=... angle=... score=... scale=... visible=...
```

## 算法说明

### 模板制作

模板图像和掩模必须为非空、尺寸相同的 8 位图像；彩色输入会先转换为灰度图。程序建立高斯金字塔，在掩模内提取边缘点和归一化梯度方向，将坐标转换为模板中心原点，并为每层保存一份 canonical `0°` 特征。

JSON 模型只保存 canonical 特征。加载 JSON 或二进制模型时，匹配器根据 `angle_start`、`angle_end` 和 `angle_step` 旋转点坐标与梯度向量，并生成所需角度缓存。这样模型存储和训练阶段内存复杂度由约 `O(levels * angles * features)` 降为 `O(levels * features)`。加载历史上已经包含多角度数据的二进制模型时，不会重复展开。

### 粗到精搜索

待测图像建立梯度金字塔，在可用的最高层进行位置和角度粗搜索，再逐层向原图精化。得分是模板梯度方向与图像梯度方向的余弦相似度。返回结果前会合并邻近候选并抑制重叠候选。

对于部分可见目标，变换后位于图像外的点会被安全跳过，得分按可见点数归一化；`min_visible_ratio` 可拒绝可见特征过少的候选。绘制时同样裁剪轮廓和旋转矩形。

多尺度匹配按 `scale_step` 遍历 `scale_min` 至 `scale_max`，并执行跨尺度重叠抑制。结果中的 `scale` 表示目标相对于训练模板的尺寸比例。多模板匹配逐个复用公开搜索接口，记录 `template_id`，合并候选后统一执行重叠抑制。

完整的数学推导、边界处理和工程注意事项见 [docs/template_matching_algorithm.md](docs/template_matching_algorithm.md)。

## 参数建议

| 参数 | 合法范围或建议初值 | 作用 |
| --- | --- | --- |
| `angle_start`, `angle_end` | `start <= end` | 限制搜索角度范围；范围越大耗时越高。 |
| `angle_step` | 有限正数，常用 `1°` | 越小角度分辨率越高，但缓存和搜索开销越大。 |
| `min_score` | `[0, 1]`，建议从 `0.7` 开始 | 提高可减少误检，降低可增强噪声或截断目标召回。 |
| `greediness` | `(0, 1]`，建议从 `0.9` 开始 | 控制弱候选的提前终止。 |
| `max_overlap` | `[0, 1]`，建议从 `0.5` 开始 | 越低越强地抑制重复目标。 |
| `scale_min`, `scale_max` | `0 < min <= max` | 定义目标尺度范围；传统单尺度行为使用 `1,1`。 |
| `scale_step` | 有限正数，建议从 `0.05` 开始 | 越小尺度分辨率越高，耗时近似成比例增加。 |
| `min_visible_ratio` | `(0, 1]` | 完整目标使用 `1.0`；边界截断目标可从 `0.5` 开始。 |

近似穷举复杂度为 `O(positions * angles * scales * features)`。应尽量收窄角度和尺度范围，并在正常样本及边界样本上联合调节 `min_score` 与 `min_visible_ratio`。

## 自动化测试

CTest 中的确定性合成回归测试覆盖 `0.8x`、`1.0x`、`1.2x` 匹配、右边界部分可见目标、带 `template_id` 的双模板匹配、canonical-only 训练数据，以及不越界的结果绘制。

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

## 直接使用 g++ 编译

也可以绕过 CMake，手动编译共享库和可执行文件：

```bash
g++ -std=c++11 -O3 -fopenmp -fPIC -march=native -msse -msse2 -msse3 -msse4 -mavx -o libMakeTemplateV1.so -shared src/MakeTemplateV1.cpp \
-I./include -I./3rdparty/opencv/include -L./3rdparty/opencv/lib \
-lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs -lopencv_calib3d -pthread

g++ -std=c++11 -O3 -fopenmp -fPIC -march=native -msse -msse2 -msse3 -msse4 -mavx -o libFindTemplateV1.so -shared src/FindTemplateV1.cpp \
-I./include -I./3rdparty/opencv/include -L./3rdparty/opencv/lib \
-lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs -lopencv_calib3d -pthread
```

## 性能参考

在 CPU `I7-10700` 上，已有样例的执行时间约为 `41ms` 至 `207ms`，实际耗时取决于图像分辨率、角度范围、尺度数量、模板特征数和候选数量。

部分图片来源于 [NCC](https://github.com/DennisLiu1993/Fastest_Image_Pattern_Matching.git)。

## 安装

```bash
make install
```

安装后共享库位于 `publish/lib`，可执行文件位于 `publish/bin`，头文件位于 `publish/include/shapeMatch`。

## 许可证

本项目采用 MIT License，详见 [LICENSE](LICENSE)。
