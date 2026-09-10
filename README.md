# edgeMatching

> 基于边缘梯度余弦相似度的金字塔模板匹配

本项目使用 OpenCV 实现工业场景中的边缘形状模板制作与匹配，提供模板制作库、模板匹配库，以及 `train` 和 `inference` 可执行程序。

## 功能特性

- **紧凑的 canonical 模型**：训练时每层金字塔只提取 `0°` canonical 边缘特征；模型加载时根据角度配置生成并缓存搜索角度，避免训练阶段重复旋转和存储。
- **部分可见目标**：目标被图像边界截断时仍可匹配。评分只使用位于图像内的特征点，并由 `min_visible_ratio` 控制最低可见比例；绘制结果会自动裁剪到图像范围。
- **多尺度匹配**：无需重新训练模板即可搜索离散尺度范围，例如 `0.8` 至 `1.2`。
- **多模板匹配**：一次调用可加载多个模型文件；结果包含 `template_id`、位置、角度、尺度、得分和可见比例。模型 ID 必须为唯一正数。
- **清晰的结果可视化**：沿外包围矩形边缘平行绘制索引号、模板 ID、得分和尺度；轮廓、矩形和文字均安全裁剪。
- **逐点匹配质量**：模板轮廓点按梯度方向余弦相似度着色，绿色为强匹配、黄色为中等匹配、红色大点为弱匹配或缺失边缘。
- **模板金字塔展开图**：训练时可将真实灰度金字塔及每层 canonical 特征保存为一张从左上到右下展开的图像，画布随层数和图像尺寸自动扩展。

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
./train ../assert/m1.png --id 7 --output model_7.json
./train ../assert/m1.png --id 7 --output model_7.json --pyramid-output pyramid_7.png
```

完整命令行格式如下：

```bash
./train [template_image] [--id N] [--output FILE] [--pyramid-output FILE]
```

不传参数时，默认读取 `../assert/m1.png`，生成 `./model.json`，模板 ID 为 `1`。
`--pyramid-output` 为可选项；指定后会额外保存模板特征金字塔图。图中黄色点是各层实际参与匹配的 canonical 特征，最粗层位于左上角，原始分辨率层位于右下角。

### 匹配过程可视化编译开关

粗匹配和逐层精匹配的交互式窗口默认关闭，避免影响并行计算和无人值守运行。按需重新配置构建目录：

```bash
# 观察最高金字塔层的粗匹配搜索及候选结果
cmake -S . -B build-debug -DSHAPE_MATCH_VISUALIZE_COARSE=ON

# 观察每个候选从高层向低层细化的过程
cmake -S . -B build-debug -DSHAPE_MATCH_VISUALIZE_FINE=ON

# 两者同时开启
cmake -S . -B build-debug \
  -DSHAPE_MATCH_VISUALIZE_COARSE=ON \
  -DSHAPE_MATCH_VISUALIZE_FINE=ON
cmake --build build-debug --parallel
```

这些开关使用 `cv::imshow`/`cv::waitKey` 逐帧暂停，并为对应阶段关闭 OpenMP 并行。轮廓绘制会跳过裁剪图之外的点，不影响部分可见目标的评分；如果当前 OpenCV 后端没有可用图形桌面，程序会输出一次提示、关闭后续窗口并继续完成推理。在服务器或 CI 中仍建议保持 `OFF`。

匹配程序示例：

```bash
./inference
./inference ../assert/src1_2_3.bmp
```

完整命令行格式如下：

```bash
./inference [search_image] [model1.json model2.json ...] \
  [--min-score N] [--max-overlap N] \
  [--angle-start DEG] [--angle-end DEG] \
  [--scale-min N] [--scale-max N] [--scale-step N] \
  [--min-visible-ratio N] [--min-contrast N] \
  [--metric use-polarity|ignore-global-polarity|ignore-local-polarity] \
  [--subpixel] [--output FILE]
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

## `assert` 样例快速复现

以下命令覆盖 `assert/` 目录中的全部 8 张模板图和 13 张待测图。请在仓库根目录执行；模型、结果图和日志都会写入 `build/repro/`，不会覆盖 `assert/` 内的原图。

先构建程序并创建输出目录：

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
cmake --build build --parallel
mkdir -p build/repro
```

训练全部模板（ID 与文件名中的数字一致）：

```bash
build/train assert/m1.png --id 1 --output build/repro/model_1.json | tee build/repro/train_1.log
build/train assert/m2.png --id 2 --output build/repro/model_2.json | tee build/repro/train_2.log
build/train assert/m3.png --id 3 --output build/repro/model_3.json | tee build/repro/train_3.log
build/train assert/m4.bmp --id 4 --output build/repro/model_4.json | tee build/repro/train_4.log
build/train assert/m5.jpg --id 5 --output build/repro/model_5.json | tee build/repro/train_5.log
build/train assert/m6.bmp --id 6 --output build/repro/model_6.json | tee build/repro/train_6.log
build/train assert/m7.bmp --id 7 --output build/repro/model_7.json | tee build/repro/train_7.log
build/train assert/m8.bmp --id 8 --output build/repro/model_8.json | tee build/repro/train_8.log
```

推理全部待测图：

```bash
# m1、m2、m3 三模板联合匹配，包含左右边界的半截目标
build/inference assert/src1_2_3.bmp build/repro/model_1.json build/repro/model_2.json build/repro/model_3.json --angle-start -5 --angle-end 5 --min-visible-ratio 0.5 --output build/repro/result_1_2_3.png | tee build/repro/infer_1_2_3.log

# m4、m5、m6 单模板匹配
build/inference assert/src4.bmp build/repro/model_4.json --output build/repro/result_4.png | tee build/repro/infer_4.log
build/inference assert/src5.bmp build/repro/model_5.json --output build/repro/result_5.png | tee build/repro/infer_5.log
build/inference assert/src6.jpg build/repro/model_6.json --output build/repro/result_6.png | tee build/repro/infer_6.log

# m7 在 8 张旋转/位移待测图上匹配
build/inference assert/src7_1.bmp build/repro/model_7.json --output build/repro/result_7_1.png | tee build/repro/infer_7_1.log
build/inference assert/src7_2.bmp build/repro/model_7.json --output build/repro/result_7_2.png | tee build/repro/infer_7_2.log
build/inference assert/src7_3.bmp build/repro/model_7.json --output build/repro/result_7_3.png | tee build/repro/infer_7_3.log
build/inference assert/src7_4.bmp build/repro/model_7.json --output build/repro/result_7_4.png | tee build/repro/infer_7_4.log
build/inference assert/src7_5.bmp build/repro/model_7.json --output build/repro/result_7_5.png | tee build/repro/infer_7_5.log
build/inference assert/src7_6.bmp build/repro/model_7.json --output build/repro/result_7_6.png | tee build/repro/infer_7_6.log
build/inference assert/src7_7.bmp build/repro/model_7.json --output build/repro/result_7_7.png | tee build/repro/infer_7_7.log
build/inference assert/src7_8.bmp build/repro/model_7.json --output build/repro/result_7_8.png | tee build/repro/infer_7_8.log

# m8 共 7 个真实目标：单尺度搜索，保留边界部分可见目标
build/inference assert/src8.bmp build/repro/model_8.json --min-score 0.95 --min-visible-ratio 0.5 --output build/repro/result_8.png | tee build/repro/infer_8.log
```

`src8.bmp` 中的真实目标尺度均为 `1.0x`。对该图强制遍历 `0.8x`–`1.2x`
不会增加有效召回，反而会近似按尺度数量成倍增加耗时，并且容易在条形模板的
局部重复结构上产生低分候选。此处使用 `--min-score 0.95` 过滤分数约为
`0.70`–`0.86` 的局部匹配，保留 7 个分数为 `0.95`–`0.996` 的真实结果。只有
待测数据确实存在尺寸变化时，才建议设置 `--scale-min`/`--scale-max`/`--scale-step`。

`src1_2_3.bmp` 的预期结果为 33 个：27 个完整目标、左边界 3 个部分可见的
`m3`，以及右边界 3 个中心位于图外的 `m1`。`--min-visible-ratio 0.5` 使部分目标
按实际可见特征评分；目标中心无需位于图内。
该样例的目标方向接近 `0°`，因此用 `--angle-start -5 --angle-end 5` 代替默认的
`-180°`–`180°` 全角度搜索，避免对三个模板计算数百个不可能的方向。

`src9_7.png` 中两个工件被其他金属工件大面积遮挡。`--min-visible-ratio`
只描述图像边界和用户掩模产生的几何可见率，不能识别场景内遮挡；默认
`use-polarity` 的有符号梯度会被遮挡物的反向边缘拉低。该场景应使用局部极性
鲁棒度量，并以较高阈值抑制杂乱背景候选：

```bash
build/inference assert/src9_7.png build/model_9.json build/model_10.json \
  build/model_11.json --min-visible-ratio 0.5 \
  --metric ignore-local-polarity --min-score 0.9 --output build/result.png
```

该配置返回 5 个目标；参考中心坐标约为 `(233,123)`、`(368,178)`、
`(204,240)`、`(323,327)` 和 `(201,334)`。局部极性模式约束更宽，不能简单
沿用默认低阈值，否则会增加假阳性。

最后运行确定性合成回归测试：

```bash
ctest --test-dir build --output-on-failure
```

## 算法说明

### 模板制作

模板图像和掩模必须为非空、尺寸相同的 8 位图像；彩色输入会先转换为灰度图。程序建立高斯金字塔，在掩模内提取边缘点和归一化梯度方向，将坐标转换为模板中心原点，并为每层保存一份 canonical `0°` 特征。

边缘特征会沿归一化梯度法线对前、中、后三个幅值样本做二次曲线拟合，将峰值位置限制在原像素的 `±0.5 px` 内，从而得到亚像素模板坐标。旋转模板缓存保留浮点坐标，不再在每个角度上量化为整数。

JSON 模型只保存 canonical 特征。加载 JSON 或二进制模型时，匹配器根据 `angle_start`、`angle_end` 和 `angle_step` 旋转点坐标与梯度向量，并生成所需角度缓存。这样模型存储和训练阶段内存复杂度由约 `O(levels * angles * features)` 降为 `O(levels * features)`。加载历史上已经包含多角度数据的二进制模型时，不会重复展开。

### 粗到精搜索

待测图像建立梯度金字塔，在可用的最高层进行位置和角度粗搜索，再逐层向原图精化。得分是模板梯度方向与图像梯度方向的余弦相似度。返回结果前会合并邻近候选并抑制重叠候选。

对于部分可见目标，变换后位于图像外的点会被安全跳过，得分按可见点数归一化；`min_visible_ratio` 可拒绝可见特征过少的候选。绘制时同样裁剪轮廓和旋转矩形。

多尺度匹配按 `scale_step` 遍历 `scale_min` 至 `scale_max`，并执行跨尺度重叠抑制。结果中的 `scale` 表示目标相对于训练模板的尺寸比例。多模板匹配只裁切一次 ROI、每个尺度只缩放一次输入，再由各模板复用只读尺度图像完成独立匹配；结果记录 `template_id`，合并候选后统一执行重叠抑制。

`--min-contrast` 接受 `[0,361]` 的整数，按搜索图中央差分梯度幅值过滤弱边缘；默认 `0` 保持旧行为。`visible` 只统计坐标在图内且掩模为白色的几何可见点，`matched` 是这些可见点中达到最小对比度且梯度非零的比例。弱边缘仍计入得分分母，因此不会因只剩少量强边缘而产生虚高分。

`--metric use-polarity` 使用有符号梯度方向；`ignore-global-polarity` 允许整个候选统一反色；`ignore-local-polarity` 则逐点忽略极性。后两者适合亮暗关系会变化的目标，但约束依次更宽松。

使用 `--subpixel` 后，NMS 仅对最终候选执行亚像素 `x/y/angle/scale` 精修：先在相邻角度得分上做抛物线拟合，再在精修角度下优化位置并联合复评；多尺度搜索还会对相邻三个尺度的同一空间峰做二次插值。内部目标下降时回退离散姿态。对外 `score` 保留离散匹配得分，不与内部双线性目标混用。精修每次最多均匀采样 512 个模板特征，输出比例是该固定采样集上的估计值。默认关闭。

例如，对 `src8.bmp` 输出亚像素位置：

```bash
build/inference assert/src8.bmp build/repro/model_8.json --min-score 0.95 \
  --min-visible-ratio 0.5 --subpixel --output build/repro/result_8_subpixel.png
```

多模板接口要求模型指针非空，且每个模型的 `template_id` 为唯一正数；不满足时接口返回 `false`。

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
| `min_contrast` | `[0, 361]`，默认 `0` | 屏蔽搜索图中的弱梯度；建议用代表性样本从 `10`逐步上调。 |
| `metric` | `use/global/local polarity` | 亮暗关系稳定时用 `use`；整体或局部反色时分别用 `global`/`local`。 |
| `subpixel` | 开/关，默认关 | 在 NMS 后联合精修 `x/y/angle/scale`，不让亚像素计算进入全图穷举。 |

近似穷举复杂度为 `O(positions * angles * scales * features)`。应尽量收窄角度和尺度范围，并在正常样本及边界样本上联合调节 `min_score` 与 `min_visible_ratio`。

## 自动化测试

CTest 中的确定性合成回归测试覆盖 `0.8x`、`1.0x`、`1.2x` 匹配、边界部分可见目标、多模板 ID、canonical-only 训练数据、全局/局部极性反转、搜索对比度、非整数角度精修，以及 CLI 非法参数拒绝。

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
