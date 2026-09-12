#include "MakeTemplateV1.h"
#include <omp.h>
#include <thread>
#include <fstream>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

using namespace SM_V1;

namespace
{
// 双线性采样仅在 Devernay 后端内部使用；匹配器继续使用稠密梯度场而不是稀疏
// NMS 响应，因此仍支持亚像素匹配。
bool sampleMagnitude(const std::vector<float>& magnitude, int width, int height,
                     double x, double y, float& value)
{
    if (x < 0.0 || y < 0.0 || x >= width - 1.0 || y >= height - 1.0) return false;
    const int x0 = static_cast<int>(std::floor(x));
    const int y0 = static_cast<int>(std::floor(y));
    const double ax = x - x0;
    const double ay = y - y0;
    const float v00 = magnitude[y0 * width + x0];
    const float v10 = magnitude[y0 * width + x0 + 1];
    const float v01 = magnitude[(y0 + 1) * width + x0];
    const float v11 = magnitude[(y0 + 1) * width + x0 + 1];
    value = static_cast<float>((1.0 - ay) * ((1.0 - ax) * v00 + ax * v10) +
                               ay * ((1.0 - ax) * v01 + ax * v11));
    return std::isfinite(value);
}

// 独立实现 Devernay 风格亚像素边缘检测：平滑图像 -> 中心差分梯度 -> 插值 NMS
// -> Canny 滞后连接 -> 沿梯度法线进行二次定位。
std::vector<T_T::TemplateFeatures> extractDevernayFeatures(
    const cv::Mat& image, const cv::Mat& mask, int minContrast, int maxContrast)
{
    std::vector<T_T::TemplateFeatures> features;
    if (image.empty() || image.type() != CV_8UC1 || mask.empty() ||
        mask.size() != image.size() || mask.type() != CV_8UC1 ||
        image.cols < 5 || image.rows < 5) return features;

    const int width = image.cols;
    const int height = image.rows;
    const size_t count = static_cast<size_t>(width) * height;
    std::vector<uint8_t> smooth(count, 0);
    const int kernel[25] = {1, 4, 7, 4, 1, 4, 16, 26, 16, 4,
                            7, 26, 41, 26, 7, 4, 16, 26, 16, 4,
                            1, 4, 7, 4, 1};
    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x)
            smooth[static_cast<size_t>(y) * width + x] = image.at<unsigned char>(y, x);
    for (int y = 2; y < height - 2; ++y)
        for (int x = 2; x < width - 2; ++x)
        {
            int sum = 0;
            int k = 0;
            for (int yy = y - 2; yy <= y + 2; ++yy)
                for (int xx = x - 2; xx <= x + 2; ++xx)
                    sum += image.at<unsigned char>(yy, xx) * kernel[k++];
            smooth[y * width + x] = static_cast<uint8_t>(sum / 273);
        }

    std::vector<float> gradX(count, 0.0f), gradY(count, 0.0f), magnitude(count, 0.0f);
    float maxMagnitude = 0.0f;
    for (int y = 1; y < height - 1; ++y)
        for (int x = 1; x < width - 1; ++x)
        {
            const size_t index = static_cast<size_t>(y) * width + x;
            if (mask.at<unsigned char>(y, x) != 255) continue;
            const float dx = static_cast<float>(smooth[index + 1]) - smooth[index - 1];
            const float dy = static_cast<float>(smooth[index + width]) - smooth[index - width];
            const float mag = std::sqrt(dx * dx + dy * dy);
            gradX[index] = dx;
            gradY[index] = dy;
            magnitude[index] = mag;
            maxMagnitude = std::max(maxMagnitude, mag);
        }
    if (!(maxMagnitude > 1e-6f) || !std::isfinite(maxMagnitude)) return features;

    // 沿连续梯度法线方向执行插值非极大值抑制，不将方向量化为 0/45/90/135 度。
    std::vector<float> nms(count, 0.0f);
    for (int y = 1; y < height - 1; ++y)
        for (int x = 1; x < width - 1; ++x)
        {
            const size_t index = static_cast<size_t>(y) * width + x;
            const float mag = magnitude[index];
            if (!(mag > 1e-6f)) continue;
            const float nx = gradX[index] / mag;
            const float ny = gradY[index] / mag;
            float minus = 0.0f, plus = 0.0f;
            if (!sampleMagnitude(magnitude, width, height, x - nx, y - ny, minus) ||
                !sampleMagnitude(magnitude, width, height, x + nx, y + ny, plus)) continue;
            if (mag >= minus && mag >= plus) nms[index] = mag;
        }

    // 保持现有公开约定：对比度取最强梯度的百分比，范围为 0 到 255。
    const float low = static_cast<float>(std::max(0, minContrast)) / 255.0f * maxMagnitude;
    const float high = static_cast<float>(std::max(0, maxContrast)) / 255.0f * maxMagnitude;
    std::vector<unsigned char> state(count, 0); // 1=弱边缘, 2=强边缘/已连接
    std::vector<size_t> pending;
    for (int y = 1; y < height - 1; ++y)
        for (int x = 1; x < width - 1; ++x)
        {
            const size_t index = static_cast<size_t>(y) * width + x;
            if (mask.at<unsigned char>(y, x) != 255 || !(nms[index] >= low)) continue;
            state[index] = nms[index] >= high ? 2 : 1;
            if (state[index] == 2) pending.push_back(index);
        }
    for (size_t head = 0; head < pending.size(); ++head)
    {
        const int x = static_cast<int>(pending[head] % width);
        const int y = static_cast<int>(pending[head] / width);
        for (int dy = -1; dy <= 1; ++dy)
            for (int dx = -1; dx <= 1; ++dx)
            {
                if (dx == 0 && dy == 0) continue;
                const int xx = x + dx, yy = y + dy;
                if (xx < 1 || xx >= width - 1 || yy < 1 || yy >= height - 1) continue;
                const size_t neighbor = static_cast<size_t>(yy) * width + xx;
                if (state[neighbor] == 1)
                {
                    state[neighbor] = 2;
                    pending.push_back(neighbor);
            }
        }
    }

    // 保留真实的分离小轮廓（例如金字塔缩小后的孔洞）。全局高阈值可能生成外边界
    // 种子，却因缺少强邻居而丢弃较弱闭合边；仅提升长度足够且局部峰值有意义的
    // 弱分量，孤立噪声仍然拒绝。
    std::vector<unsigned char> weakVisited(count, 0);
    const float componentFloor = std::max(low, high * 0.35f);
    for (int y = 1; y < height - 1; ++y)
        for (int x = 1; x < width - 1; ++x)
        {
            const size_t seed = static_cast<size_t>(y) * width + x;
            if (state[seed] != 1 || weakVisited[seed]) continue;
            std::vector<size_t> component;
            component.push_back(seed);
            weakVisited[seed] = 1;
            float componentMax = nms[seed];
            for (size_t head = 0; head < component.size(); ++head)
            {
                const int cx = static_cast<int>(component[head] % width);
                const int cy = static_cast<int>(component[head] / width);
                for (int dy = -1; dy <= 1; ++dy)
                    for (int dx = -1; dx <= 1; ++dx)
                    {
                        if (dx == 0 && dy == 0) continue;
                        const int xx = cx + dx, yy = cy + dy;
                        if (xx < 1 || xx >= width - 1 || yy < 1 || yy >= height - 1) continue;
                        const size_t neighbor = static_cast<size_t>(yy) * width + xx;
                        if (state[neighbor] == 1 && !weakVisited[neighbor])
                        {
                            weakVisited[neighbor] = 1;
                            component.push_back(neighbor);
                            componentMax = std::max(componentMax, nms[neighbor]);
                        }
                    }
            }
            if (component.size() >= 3 && componentMax >= componentFloor)
                for (const size_t index : component) state[index] = 2;
        }

    features.reserve(pending.size());
    for (int y = 1; y < height - 1; ++y)
        for (int x = 1; x < width - 1; ++x)
        {
            const size_t index = static_cast<size_t>(y) * width + x;
            if (state[index] != 2) continue;
            const float mag = magnitude[index];
            if (!(mag > 1e-6f)) continue;
            const float nx = gradX[index] / mag;
            const float ny = gradY[index] / mag;
            float minus = 0.0f, plus = 0.0f;
            if (!sampleMagnitude(magnitude, width, height, x - nx, y - ny, minus) ||
                !sampleMagnitude(magnitude, width, height, x + nx, y + ny, plus)) continue;
            const float denominator = minus - 2.0f * mag + plus;
            float offset = 0.0f;
            if (denominator < -1e-6f)
            {
                const float candidate = 0.5f * (minus - plus) / denominator;
                if (std::isfinite(candidate) && std::fabs(candidate) <= 0.5f)
                    offset = candidate;
            }
            features.push_back({static_cast<float>(x) + offset * nx,
                                static_cast<float>(y) + offset * ny,
                                nx, ny, 1.0f});
        }
    return features;
}
}

CreateTemplate::CreateTemplate() = default;
CreateTemplate::~CreateTemplate() = default;

// 将输入长度转换为最接近的2的幂
int CreateTemplate::_convertLength(int length_src)
{
    for (int i = 4;; i++)
    {
        int temp = (int)pow(2.0, i);
        if (temp >= length_src)
        {
            length_src = temp;
            break;
        }
    }
    return length_src;
}

// 特征点筛选：选取一些散的开的特征点
bool CreateTemplate::_selectScatteredFeatures(
    std::vector<T_T::TemplateFeatures> candidates,
    std::vector<T_T::TemplateFeatures>& templ,
    int featuresmaxnumber,
    float distance)
{
    templ.clear();
    float distance_square = distance * distance;
    int i = 0;
    std::set<int> index;
    while (templ.size() < featuresmaxnumber)
    {
        if (index.find(i) != index.end())
        {
            i++;
            continue;
        }
        if (i >= candidates.size()) { break; }
        T_T::TemplateFeatures c = candidates[i];
        //确保距离间隔大的点被选取
        bool keep = true;
        for (size_t j = 0; j < templ.size() && keep; ++j)
        {
            T_T::TemplateFeatures f = templ[j];
            keep = ((c.x - f.x) * (c.x - f.x) + (c.y - f.y) * (c.y - f.y) >= distance_square);
        }
        if (keep)
        {
            templ.push_back(c);
            index.insert(i);
        }
        if (++i == candidates.size())
        {
            i = 0;
            distance -= 1.0f;
            distance_square = distance * distance;
        }
    }
    return true;
}

// 5x5 高斯滤波
void CreateTemplate::_gaussianFilter(uint8_t* corrupted, uint8_t* smooth, int width, int height)
{
    int templates[25] = {1, 4, 7, 4, 1, 4, 16, 26, 16, 4, 7, 26, 41, 26, 7, 4, 16, 26, 16, 4, 1, 4, 7, 4, 1};

    memcpy(smooth, corrupted, width * height * sizeof(uint8_t));
    for (int j = 2; j < height - 2; j++)
    {
        for (int i = 2; i < width - 2; i++)
        {
            int sum = 0;
            int index = 0;
            for (int m = j - 2; m < j + 3; m++)
            {
                for (int n = i - 2; n < i + 3; n++)
                {
                    sum += corrupted[m * width + n] * templates[index++];
                }
            }
            sum /= 273;
            if (sum > 255) sum = 255;
            smooth[j * width + i] = (uint8_t)sum;
        }
    }
}

// 提取模板特征信息具体实现和算法
void CreateTemplate::_extractShapeInfo(
    cv::Mat image_data,
    uint8_t* mask_data,
    T_T::ShapeAngle::Ptr angle_info_data,
    int min_contrast,
    int max_contrast)
{
    // 原图像大小
    int width = image_data.cols;
    int height = image_data.rows;
    int32_t buffer_size = image_data.cols * image_data.rows;

    if (edge_method_ == T_T::EDGE_DEVERNAY)
    {
    // Devernay 特征已经定位到图像坐标；沿用 CURRENT 后端的中心原点模型表示和
    // 单位梯度约定，以共享序列化模型和旋转缓存。
    const cv::Mat mask_view(height, width, CV_8UC1, mask_data);
    const std::vector<T_T::TemplateFeatures> devernay_features =
        extractDevernayFeatures(image_data, mask_view, min_contrast, max_contrast);
    angle_info_data->shape_point.resize(devernay_features.size());
    for (size_t m = 0; m < devernay_features.size(); ++m)
    {
        angle_info_data->shape_point[m].x =
            devernay_features[m].x - origin_x_;
        angle_info_data->shape_point[m].y =
            devernay_features[m].y - origin_y_;
        angle_info_data->shape_point[m].edge_dx = devernay_features[m].edge_dx;
        angle_info_data->shape_point[m].edge_dy = devernay_features[m].edge_dy;
    }
        return;
    }

    std::vector<uint8_t> pBufOut(buffer_size);
    std::vector<int16_t> pBufGradX(buffer_size); //存取x方向偏导数dx
    std::vector<int16_t> pBufGradY(buffer_size); //存取y方向偏导数dy
    std::vector<int32_t> pBufOrien(buffer_size); //存取梯度方向
    std::vector<float> pBufMag(buffer_size); //存取梯度模

    std::vector<T_T::TemplateFeatures> TF0degree, TF0degree_temp;

    //===================================================================================
    // 步骤 0：获取高斯模糊后的梯度信息[grad_x_edge,grad_y_edge]及原始梯度信息[grad_x,grad_y]
    //===================================================================================
    ///************************** [高斯模糊->获取边缘点] ********************************///
#if 0
    cv::Mat GaussianImg;
    cv::GaussianBlur(image_data, GaussianImg, cv::Size(7, 7), 0, 0);
    cv::Mat grad_x_edge, grad_y_edge;
    cv::Sobel(GaussianImg, grad_x_edge, CV_16S, 1, 0, 3, 1.0);
    cv::Sobel(GaussianImg, grad_y_edge, CV_16S, 0, 1, 3, 1.0);

    ///************************** [在边缘点上 -> 获取真实的梯度信息] ********************************///
    cv::Mat grad_x, grad_y;
    cv::Sobel(image_data, grad_x, CV_16S, 1, 0, 3, 1.0);
    cv::Sobel(image_data, grad_y, CV_16S, 0, 1, 3, 1.0);
#endif
    uint8_t* pInput = (uint8_t*)malloc(buffer_size * sizeof(uint8_t));
    uint8_t* ImageData = (uint8_t*)image_data.data;
    _gaussianFilter(ImageData, pInput, width, height);

    //初始化
    float MaxGradient = -9999.99f;
    int count = 0, i, j;

    //===================================================================================
    // 步骤 1：获取图像的梯度方向
    //===================================================================================
    for (i = 1; i < width - 1; i++)
    {
        for (j = 1; j < height - 1; j++)
        {
            int index = j * width + i;
#if 0
            int16_t sdx   = grad_x_edge.at<short>(j, i);
            int16_t sdy   = grad_y_edge.at<short>(j, i);
#endif
            int16_t sdx = *(pInput + j * width + i + 1) - *(pInput + j * width + i - 1);
            int16_t sdy = *(pInput + (j + 1) * width + i) - *(pInput + (j - 1) * width + i);

            pBufGradX[index] = sdx;
            pBufGradY[index] = sdy;

            // 如果 usr_mask 图像画成了黑色，则梯度为0
            if (*(mask_data + index) != 0xff)
            {
                pBufGradX[index] = 0;
                pBufGradY[index] = 0;
            }

            float magnitude_edge = std::sqrt(static_cast<float>(sdx * sdx) + static_cast<float>(sdy * sdy));
            pBufMag[index] = magnitude_edge;

            // 找到图中最大的梯度强度值[MaxGradient]，后续用来归一化
            if (magnitude_edge > MaxGradient) MaxGradient = magnitude_edge;

            int16_t fdx = pBufGradX[index];
            int16_t fdy = pBufGradY[index];

            float direction = cv::fastAtan2(static_cast<float>(fdy), static_cast<float>(fdx));

            //  获取梯度方向接近 0, 45, 90, 135 的数据集
            if ((direction > 0 && direction <= 22.5f) || (direction > 157.5f && direction <= 202.5f) || (direction >
                337.5f && direction <= 360))
                direction = 0;
            else if ((direction > 22.5f && direction <= 67.5f) || (direction > 202.5f && direction <= 247.5f))
                direction = 45;
            else if ((direction > 67.5f && direction <= 112.5f) || (direction > 247.5f && direction <= 292.5f))
                direction = 90;
            else if ((direction > 112.5f && direction <= 157.5f) || (direction > 292.5f && direction <= 337.5f))
                direction = 135;
            else
                direction = 0;

            pBufOrien[count] = static_cast<int32_t>(direction);
            count++;
        }
    } // 结束 [S1：生成方向]

    //===================================================================================
    // 步骤 2：非最大值抑制[NMS]
    //===================================================================================
    // 初始化 count
    count = 0;
    float leftPixel, rightPixel;
    for (i = 1; i < width - 1; i++)
    {
        for (j = 1; j < height - 1; j++)
        {
            int index = j * width + i;
            switch (pBufOrien[count])
            {
            case 0:
                leftPixel = pBufMag[j * width + i - 1]; //左（i-1,j）
                rightPixel = pBufMag[j * width + i + 1]; //右（i+1,j）
                break;
            case 45:
                leftPixel = pBufMag[(j - 1) * width + i - 1]; //左上（i-1,j-1）
                rightPixel = pBufMag[(j + 1) * width + i + 1]; //右下（i+1,j+1）
                break;
            case 90:
                leftPixel = pBufMag[(j - 1) * width + i]; //上（i,j-1）
                rightPixel = pBufMag[(j + 1) * width + i]; //下（i,j+1）
                break;
            case 135:
                leftPixel = pBufMag[(j + 1) * width + i - 1]; //左下（i-1,j+1）
                rightPixel = pBufMag[(j - 1) * width + i + 1]; //右上（i+1,j-1）
                break;
            }

            // 如果当前像素幅值小于对应方向相邻的两个像素的幅值，则抑制这个像素：0
            // 用<=则不保留直线特征;<则保留直线特征
            if ((pBufMag[index] < leftPixel) || (pBufMag[index] < rightPixel) || (*(mask_data + index) == 0x00))
            {
                //模pBufMag比8邻域的像素强度要低则为0
                pBufOut[index] = 0;
            }
            else
            {
                pBufOut[index] = (uint8_t)(pBufMag[index] / MaxGradient * 255);
            }
            count++;
        }
    } // 结束 [S2：非极大值抑制]

    // 步骤 3：滞后阈值，双阈值
    //===================================================================================
    int flag = 1;
    for (i = 1; i < width - 1; i++)
    {
        for (j = 1; j < height - 1; j++)
        {
            int index = j * width + i;
#if 0
            // 获取真实梯度信息
            int16_t fdx              = grad_x.at<short>(j, i);
            int16_t fdy              = grad_y.at<short>(j, i);
            float   magnitude_origin = std::sqrt(static_cast<float>(fdx * fdx) + static_cast<float>(fdy * fdy));
#endif
            int16_t fdx = pBufGradX[index];
            int16_t fdy = pBufGradY[index];
            float magnitude_origin = pBufMag[index];

            /* 双阈值滞后过滤原理:
             * 若某一像素位置的梯度幅值超过高阈值[max_contrast]，则该位置被保留;[强边缘]
             * 若某一像素位置的梯度幅值低于低阈值[min_contrast]，则该位置被去除;[非边缘]
             * 若某一像素位置的梯度幅值处于高、低阈值之间，则该像素仅仅在连接到一个高于高阈值像素时被保留[虚边缘]；
             *
             */
            flag = 1;
            if (pBufOut[index] < max_contrast)
            {
                if (pBufOut[index] < min_contrast)
                {
                    pBufOut[index] = 0;
                    flag = 0; // 从边缘剔除标志
                }
                else
                {
                    // 如果任何8邻域都比max_contract小，抑制其边缘[0]
                    if ((pBufOut[(j - 1) * width + i - 1] < max_contrast) && (pBufOut[j * width + i - 1] < max_contrast)
                        &&
                        (pBufOut[(j + 1) * width + i - 1] < max_contrast) && (pBufOut[(j - 1) * width + i] <
                            max_contrast) &&
                        (pBufOut[(j + 1) * width + i] < max_contrast) && (pBufOut[(j - 1) * width + i + 1] <
                            max_contrast) &&
                        (pBufOut[j * width + i + 1] < max_contrast) && (pBufOut[(j + 1) * width + i + 1] <
                            max_contrast))
                    {
                        pBufOut[index] = 0;
                        flag = 0; // 从边缘剔除标志
                    }
                }
            }

            //===================================================================================
    // 步骤 4：保存数据集
            //===================================================================================
            if (flag != 0) //强边缘标志
            {
                if (fdx != 0 || fdy != 0)
                {
                    float magnitude = (!(std::fabs(magnitude_origin) < 1e-6)) ? (1 / magnitude_origin) : 0;
                    const float nx = fdx * magnitude;
                    const float ny = fdy * magnitude;
                    // 使用三个采样点沿梯度法线进行二次拟合定位边缘极值；有界偏移可
                    // 防止噪声或平坦剖面将特征移入其他像素。
                    const auto sampleMagnitude = [&](double x, double y) {
                        const int x0 = std::max(0, std::min(width - 2,
                            static_cast<int>(std::floor(x))));
                        const int y0 = std::max(0, std::min(height - 2,
                            static_cast<int>(std::floor(y))));
                        const double ax = x - x0;
                        const double ay = y - y0;
                        const float v00 = pBufMag[y0 * width + x0];
                        const float v10 = pBufMag[y0 * width + x0 + 1];
                        const float v01 = pBufMag[(y0 + 1) * width + x0];
                        const float v11 = pBufMag[(y0 + 1) * width + x0 + 1];
                        return static_cast<float>((1.0 - ay) * ((1.0 - ax) * v00 + ax * v10) +
                                                  ay * ((1.0 - ax) * v01 + ax * v11));
                    };
                    const float before = sampleMagnitude(i - nx, j - ny);
                    const float after = sampleMagnitude(i + nx, j + ny);
                    const float denominator = before - 2.0f * magnitude_origin + after;
                    float offset = 0.0f;
                    // CANNY_PIXEL 保留 NMS 像素位置；CURRENT 保留历史的法线方向
                    // 抛物线亚像素修正。
                    if (edge_method_ != T_T::EDGE_CANNY_PIXEL && denominator < -1e-6f)
                        offset = std::max(-0.5f, std::min(0.5f,
                            0.5f * (before - after) / denominator));
                    TF0degree.push_back({static_cast<double>(i) + offset * nx,
                                         static_cast<double>(j) + offset * ny,
                                         (float)fdx, (float)fdy, magnitude});
                }
            }
        }
    }
    /*
        //===================================================================================
        // 步骤 4：特征数过滤：按特征点比例计算保留数量
        // 原则：梯度强度由高到低排序，按特征采样比例剔除梯度较弱的梯度信息
        //===================================================================================
        if (features_rate_ >= 1.0f) features_rate_ = 1.0;
        int features_max_number = features_rate_ * TF0degree.size();

        if (TF0degree.size() > features_max_number && features_max_number > 0)
        {
            // 过滤特征点
            //梯度幅值从大 -> 小进行排序(1/magnitude)
            std::sort(TF0degree.begin(), TF0degree.end(), [](const T_T::TemplateFeatures& a, const T_T::TemplateFeatures& b)
            {
                return a.edge_mag < b.edge_mag;
            });
            float distance = static_cast<float>(TF0degree.size() / features_max_number + 1);

            //特征点筛选：当特征点数量＞设定的点数时，选取此点集合中的散的比较开的点集
            _selectScatteredFeatures(TF0degree, TF0degree_temp, features_max_number, distance);
        }
        else
        {
            TF0degree_temp = TF0degree;
        }
    */

    TF0degree_temp = TF0degree;

    //初始化：ShapePoint
    if (!TF0degree_temp.empty()) angle_info_data->shape_point.resize(TF0degree_temp.size());
    for (int m = 0; m < TF0degree_temp.size(); m++) //每层0角度下特征点数量
    {
        //坐标变化
        //此时特征点坐标按照坐标原点在图像的[左上角]  ---> 以图像[中心]为原点的坐标
        angle_info_data->shape_point[m].x = TF0degree_temp[m].x - origin_x_;
        angle_info_data->shape_point[m].y = TF0degree_temp[m].y - origin_y_;

        angle_info_data->shape_point[m].edge_dx = TF0degree_temp[m].edge_dx * TF0degree_temp[m].edge_mag;
        angle_info_data->shape_point[m].edge_dy = TF0degree_temp[m].edge_dy * TF0degree_temp[m].edge_mag;
    }

    TF0degree_temp.clear();
    TF0degree_temp.shrink_to_fit();
    TF0degree.clear();
    TF0degree.shrink_to_fit();
    free(pInput);
}

// 初始化各层金字塔的模板信息
void CreateTemplate::_initialShapeModelPyd(T_T::ShapeInfo::Ptr shape_info_vec, int angle_start, int angle_stop,
                                           double angle_step)
{
    // 训练文件只保存每层金字塔的 canonical 0° 特征。搜索角度序列在模型加载时
    // 由 angle_start/angle_stop/angle_step 确定性展开，避免训练阶段的无效旋转和内存峰值。
    (void)angle_start;
    (void)angle_stop;
    (void)angle_step;
    shape_info_vec->shape_angle.clear();
    auto canonical = std::make_shared<T_T::ShapeAngle>();
    canonical->angle = 0.0;
    shape_info_vec->shape_angle.push_back(canonical);
}

// 初始化模板资源
void CreateTemplate::_initialShapeModel(T_T::Template::Ptr model_id)
{
    int angleStart = model_id->template_cfg.angle_start;
    double angleStep = model_id->template_cfg.angle_step;
    int angleStop = model_id->template_cfg.angle_end;

    // 允许调用者复用同一 model_id，不保留上次创建的金字塔/角度数据。
    model_id->templates.clear();

    //初始化 Vector:templates，内含智能指针
    for (int i = 0; i < model_id->template_cfg.num_levels + 1; i++)
    {
        model_id->templates.push_back(std::make_shared<T_T::ShapeInfo>());
    }

    for (int initPyNum = 0; initPyNum < model_id->template_cfg.num_levels + 1; initPyNum++)
    {
        if (initPyNum == 0) { angleStep = model_id->template_cfg.angle_step; } //原始层角度步长为设置步长
        else
        {
            // angleStep *= 2; // 除掉原始层外，金字塔其他层每层模板角度细分策略1: eg:[0.1] -> 0.2 -> 0.4 -> 0.8 -> 1.6 -> 3.2 -> 6.4 -> 12.8
            angleStep = initPyNum * 2;
        }

        // 初始化各金字塔层的信息（模板:起始角度->角度步长->终止角度）
        switch (initPyNum)
        {
        case 0:
            //初始化
            _initialShapeModelPyd(model_id->templates[0], angleStart, angleStop, angleStep);
            break;
        case 1:
            //初始化
            _initialShapeModelPyd(model_id->templates[1], angleStart, angleStop, angleStep);
            break;
        case 2:
            //初始化
            _initialShapeModelPyd(model_id->templates[2], angleStart, angleStop, angleStep);
            break;
        case 3:
            //初始化
            _initialShapeModelPyd(model_id->templates[3], angleStart, angleStop, angleStep);
            break;
        case 4:
            //初始化
            _initialShapeModelPyd(model_id->templates[4], angleStart, angleStop, angleStep);
            break;
        case 5:
            //初始化
            _initialShapeModelPyd(model_id->templates[5], angleStart, angleStop, angleStep);
            break;
        case 6:
            //初始化
            _initialShapeModelPyd(model_id->templates[6], angleStart, angleStop, angleStep);
            break;
        case 7:
            //初始化
            _initialShapeModelPyd(model_id->templates[7], angleStart, angleStop, angleStep);
            break;
        default:
            break;
        }
    }
}

bool CreateTemplate::_rotatedShapeInfo(T_T::ShapeInfo::Ptr shape_info_vec, int xOffSet, int yOffSet)
{
    int thread_num = std::thread::hardware_concurrency();

    // S2-对金字塔第n层的1~361角度 ---> 模板特征进行旋转
    //以下遍历角度/点数
    int angleNum = shape_info_vec->shape_angle.size();
    int shapeSize = shape_info_vec->shape_angle[0]->shape_point.size();

    // 非0角度特征 - shape_point初始化
    for (int i = 1; i < angleNum; i++)
    {
        shape_info_vec->shape_angle[i]->shape_point.resize(shapeSize);
    }
#pragma omp parallel for num_threads(thread_num)
    for (int i = 1; i < angleNum; i++) //角度个数
    {
        double angle = -shape_info_vec->shape_angle[i]->angle;
        float rad = (double)((angle * CV_PI) / 180); // 180/π = 角度/弧度

        for (int j = 0; j < shapeSize; j++) //轮廓点数量
        {
            //坐标x,y变化
            double rOrigX, rOrigY;
            float X, Y, T;
            //通过坐标变化，将坐标原点0在左上角的图像坐标系转换为笛卡尔坐标系（原点在图像中心，x朝右，y朝上）
            X = shape_info_vec->shape_angle[0]->shape_point[j].x;
            Y = -shape_info_vec->shape_angle[0]->shape_point[j].y;
            T = X;
            X = X * std::cos(rad) - Y * std::sin(rad); // 逆时针旋转
            Y = T * std::sin(rad) + Y * std::cos(rad); // 逆时针旋转

            rOrigX = X + xOffSet;
            rOrigY = yOffSet - Y;

            // 更新旋转后的值x,y
            shape_info_vec->shape_angle[i]->shape_point[j].x = rOrigX - xOffSet;
            shape_info_vec->shape_angle[i]->shape_point[j].y = rOrigY - yOffSet;

            float DX, DY, DT;
            // dx,dy变换
            DX = shape_info_vec->shape_angle[0]->shape_point[j].edge_dx;
            DY = -shape_info_vec->shape_angle[0]->shape_point[j].edge_dy;
            DT = DX;
            DX = DX * std::cos(rad) - DY * std::sin(rad); // 逆时针旋转
            DY = DT * std::sin(rad) + DY * std::cos(rad); // 逆时针旋转

            // 更新旋转后的值dx,dy
            shape_info_vec->shape_angle[i]->shape_point[j].edge_dx = DX;
            shape_info_vec->shape_angle[i]->shape_point[j].edge_dy = -DY;
            // 更新旋转后的幅值 edge_mag 保持不变：无需在此处重新赋值
        }
    }
    return true;
}

// 创建角度模板序列
bool CreateTemplate::_buildModelList(
    T_T::ShapeInfo::Ptr shape_info_vec,
    cv::Mat image_data,
    cv::Mat mask_data,
    int min_contrast,
    int max_contrast)
{
    // S1-对金字塔第n层的0角度 ---> 模板特征进行提取
    _extractShapeInfo(image_data, (uint8_t*)mask_data.data, shape_info_vec->shape_angle[0], min_contrast, max_contrast);

    // S2-只计算 canonical 特征的外包围矩形。其他角度在加载阶段展开并计算。
    for (const auto item : shape_info_vec->shape_angle)
    {
        if (item->shape_point.size() != 0) //如果该层没有特征点则跳出
        {
            T_T::BboundingBox bbx;
            // 使用排序计算外接最大矩形框(左上,右下点)
            std::sort(item->shape_point.begin(), item->shape_point.end(),
                      [](const T_T::ShapePoint& pt1s, const T_T::ShapePoint& pt2s) { return pt1s.x < pt2s.x; });
            bbx.lt_x = static_cast<int>(std::floor(item->shape_point[0].x));
            std::sort(item->shape_point.begin(), item->shape_point.end(),
                      [](const T_T::ShapePoint& pt1s, const T_T::ShapePoint& pt2s) { return pt1s.y < pt2s.y; });
            bbx.lt_y = static_cast<int>(std::floor(item->shape_point[0].y));
            std::sort(item->shape_point.begin(), item->shape_point.end(),
                      [](const T_T::ShapePoint& pt1s, const T_T::ShapePoint& pt2s) { return pt1s.x > pt2s.x; });
            bbx.rb_x = static_cast<int>(std::ceil(item->shape_point[0].x));
            std::sort(item->shape_point.begin(), item->shape_point.end(),
                      [](const T_T::ShapePoint& pt1s, const T_T::ShapePoint& pt2s) { return pt1s.y > pt2s.y; });
            bbx.rb_y = static_cast<int>(std::ceil(item->shape_point[0].y));
            // 保存外接最大矩形框
            item->bbx = bbx;
        }
        else
        {
            break;
        }
    }

    return true;
}

// 创建匹配模板
bool CreateTemplate::_createModel(cv::Mat template_img, cv::Mat mask_img, T_T::Template::Ptr model_id)
{
    if (model_id->template_cfg.num_levels >= 0)
    {
        // 制作图像金字塔各层的模板特征信息
        bool isBuild = false;
        cv::Mat template_imgPy1, template_imgPy2, template_imgPy3, template_imgPy4, template_imgPy5, template_imgPy6,
                template_imgPy7;
        cv::Mat mask_imgPy1, mask_imgPy2, mask_imgPy3, mask_imgPy4, mask_imgPy5, mask_imgPy6, mask_imgPy7;

        for (int initPyNum = 0; initPyNum < model_id->template_cfg.num_levels + 1; initPyNum++)
        {
            switch (initPyNum)
            {
            case 0:
                {
                    origin_x_ = model_id->template_cfg.origin_mode == T_T::ORIGIN_DOMAIN_CENTROID
                        ? model_id->template_cfg.origin_x : template_img.cols * 0.5;
                    origin_y_ = model_id->template_cfg.origin_mode == T_T::ORIGIN_DOMAIN_CENTROID
                        ? model_id->template_cfg.origin_y : template_img.rows * 0.5;
                    isBuild = _buildModelList(
                        model_id->templates[0], template_img, mask_img, model_id->template_cfg.min_contrast,
                        model_id->template_cfg.max_contrast);
                    if (!isBuild) { return false; }
                }
                break;
            case 1:
                {
                    cv::pyrDown(template_img, template_imgPy1, cv::Size(template_img.cols / 2, template_img.rows / 2));
                    cv::pyrDown(mask_img, mask_imgPy1, cv::Size(mask_img.cols / 2, mask_img.rows / 2));

                    origin_x_ = model_id->template_cfg.origin_mode == T_T::ORIGIN_DOMAIN_CENTROID
                        ? model_id->template_cfg.origin_x / 2.0 : template_imgPy1.cols * 0.5;
                    origin_y_ = model_id->template_cfg.origin_mode == T_T::ORIGIN_DOMAIN_CENTROID
                        ? model_id->template_cfg.origin_y / 2.0 : template_imgPy1.rows * 0.5;

                    isBuild = _buildModelList(
                        model_id->templates[1],
                        template_imgPy1,
                        mask_imgPy1,
                        model_id->template_cfg.min_contrast,
                        model_id->template_cfg.max_contrast);
                    if (!isBuild) { return false; }
                }
                break;
            case 2:
                {
                    cv::pyrDown(template_imgPy1, template_imgPy2, cv::Size(template_imgPy1.cols / 2, template_imgPy1.rows / 2));
                    cv::pyrDown(mask_imgPy1, mask_imgPy2, cv::Size(mask_imgPy1.cols / 2, mask_imgPy1.rows / 2));

                    origin_x_ = model_id->template_cfg.origin_mode == T_T::ORIGIN_DOMAIN_CENTROID
                        ? model_id->template_cfg.origin_x / 4.0 : template_imgPy2.cols * 0.5;
                    origin_y_ = model_id->template_cfg.origin_mode == T_T::ORIGIN_DOMAIN_CENTROID
                        ? model_id->template_cfg.origin_y / 4.0 : template_imgPy2.rows * 0.5;

                    isBuild = _buildModelList(
                        model_id->templates[2],
                        template_imgPy2,
                        mask_imgPy2,
                        model_id->template_cfg.min_contrast,
                        model_id->template_cfg.max_contrast);
                    if (!isBuild) { return false; }
                }
                break;
            case 3:
                {
                    cv::pyrDown(template_imgPy2, template_imgPy3, cv::Size(template_imgPy2.cols / 2, template_imgPy2.rows / 2));
                    cv::pyrDown(mask_imgPy2, mask_imgPy3, cv::Size(mask_imgPy2.cols / 2, mask_imgPy2.rows / 2));

                    origin_x_ = model_id->template_cfg.origin_mode == T_T::ORIGIN_DOMAIN_CENTROID
                        ? model_id->template_cfg.origin_x / 8.0 : template_imgPy3.cols * 0.5;
                    origin_y_ = model_id->template_cfg.origin_mode == T_T::ORIGIN_DOMAIN_CENTROID
                        ? model_id->template_cfg.origin_y / 8.0 : template_imgPy3.rows * 0.5;

                    isBuild = _buildModelList(
                        model_id->templates[3],
                        template_imgPy3,
                        mask_imgPy3,
                        model_id->template_cfg.min_contrast,
                        model_id->template_cfg.max_contrast);
                    if (!isBuild) { return false; }
                }
                break;
            case 4:
                {
                    cv::pyrDown(template_imgPy3, template_imgPy4, cv::Size(template_imgPy3.cols / 2, template_imgPy3.rows / 2));
                    cv::pyrDown(mask_imgPy3, mask_imgPy4, cv::Size(mask_imgPy3.cols / 2, mask_imgPy3.rows / 2));

                    origin_x_ = model_id->template_cfg.origin_mode == T_T::ORIGIN_DOMAIN_CENTROID
                        ? model_id->template_cfg.origin_x / 16.0 : template_imgPy4.cols * 0.5;
                    origin_y_ = model_id->template_cfg.origin_mode == T_T::ORIGIN_DOMAIN_CENTROID
                        ? model_id->template_cfg.origin_y / 16.0 : template_imgPy4.rows * 0.5;

                    isBuild = _buildModelList(
                        model_id->templates[4],
                        template_imgPy4,
                        mask_imgPy4,
                        model_id->template_cfg.min_contrast,
                        model_id->template_cfg.max_contrast);
                    if (!isBuild) { return false; }
                }
                break;
            case 5:
                {
                    cv::pyrDown(template_imgPy4, template_imgPy5, cv::Size(template_imgPy4.cols / 2, template_imgPy4.rows / 2));
                    cv::pyrDown(mask_imgPy4, mask_imgPy5, cv::Size(mask_imgPy4.cols / 2, mask_imgPy4.rows / 2));

                    origin_x_ = model_id->template_cfg.origin_mode == T_T::ORIGIN_DOMAIN_CENTROID
                        ? model_id->template_cfg.origin_x / 32.0 : template_imgPy5.cols * 0.5;
                    origin_y_ = model_id->template_cfg.origin_mode == T_T::ORIGIN_DOMAIN_CENTROID
                        ? model_id->template_cfg.origin_y / 32.0 : template_imgPy5.rows * 0.5;

                    isBuild = _buildModelList(
                        model_id->templates[5],
                        template_imgPy5,
                        mask_imgPy5,
                        model_id->template_cfg.min_contrast,
                        model_id->template_cfg.max_contrast);
                    if (!isBuild) { return false; }
                }
                break;
            case 6:
                {
                    cv::pyrDown(template_imgPy5, template_imgPy6, cv::Size(template_imgPy5.cols / 2, template_imgPy5.rows / 2));
                    cv::pyrDown(mask_imgPy5, mask_imgPy6, cv::Size(mask_imgPy5.cols / 2, mask_imgPy5.rows / 2));

                    origin_x_ = model_id->template_cfg.origin_mode == T_T::ORIGIN_DOMAIN_CENTROID
                        ? model_id->template_cfg.origin_x / 64.0 : template_imgPy6.cols * 0.5;
                    origin_y_ = model_id->template_cfg.origin_mode == T_T::ORIGIN_DOMAIN_CENTROID
                        ? model_id->template_cfg.origin_y / 64.0 : template_imgPy6.rows * 0.5;

                    isBuild = _buildModelList(
                        model_id->templates[6],
                        template_imgPy6,
                        mask_imgPy6,
                        model_id->template_cfg.min_contrast,
                        model_id->template_cfg.max_contrast);
                    if (!isBuild) { return false; }
                }
                break;
            case 7:
                {
                    cv::pyrDown(template_imgPy6, template_imgPy7, cv::Size(template_imgPy6.cols / 2, template_imgPy6.rows / 2));
                    cv::pyrDown(mask_imgPy6, mask_imgPy7, cv::Size(mask_imgPy6.cols / 2, mask_imgPy6.rows / 2));

                    origin_x_ = model_id->template_cfg.origin_mode == T_T::ORIGIN_DOMAIN_CENTROID
                        ? model_id->template_cfg.origin_x / 128.0 : template_imgPy7.cols * 0.5;
                    origin_y_ = model_id->template_cfg.origin_mode == T_T::ORIGIN_DOMAIN_CENTROID
                        ? model_id->template_cfg.origin_y / 128.0 : template_imgPy7.rows * 0.5;

                    isBuild = _buildModelList(
                        model_id->templates[7],
                        template_imgPy7,
                        mask_imgPy7,
                        model_id->template_cfg.min_contrast,
                        model_id->template_cfg.max_contrast);
                    if (!isBuild) { return false; }
                }
                break;
            default:
                break;
    } // 分支结束
        } // 金字塔层数循环结束

        //金字塔层数优化（图像缩放变形后失真的问题）
        float coefficient = 0.0f;
        do
        {
            //=============================================
            //图像金字塔顶层灰度值方差(variance)与原始模板图像
            //方差(variance)的比值作为评估，来确定合适的金字塔层数
            //=============================================

            // 1)计算原始模板图像灰度值方差(template_Bordered)
            unsigned long long sum = 0; //灰度值求和
            unsigned long long sq_sum = 0; //灰度值求平方和
            float area = template_img.rows * template_img.cols;
            //局部区域：原始模板图像灰度值求和(sum)、灰度值平方求和(sq_sum)
            {
                unsigned char intension_each;
                sum = 0;
                sq_sum = 0;
                for (int y = 0; y < template_img.rows; ++y)
                {
                    for (int x = 0; x < template_img.cols; ++x)
                    {
                        intension_each = template_img.ptr<uchar>(y)[x];
                        sum += intension_each;
                        sq_sum += intension_each * intension_each;
                    }
                }
            }
            //求取原始模板图像灰度值方差(variance)
            float variance = std::sqrt(sq_sum - sum * sum / float(area));

            //读取金字塔每层的图像并存至pyLevelMat
            cv::Mat pyLevelMat;
            switch (model_id->template_cfg.num_levels)
            {
            case 0:
                pyLevelMat = template_img.clone();
                break;
            case 1:
                pyLevelMat = template_imgPy1.clone();
                break;
            case 2:
                pyLevelMat = template_imgPy2.clone();
                break;
            case 3:
                pyLevelMat = template_imgPy3.clone();
                break;
            case 4:
                pyLevelMat = template_imgPy4.clone();
                break;
            case 5:
                pyLevelMat = template_imgPy5.clone();
                break;
            case 6:
                pyLevelMat = template_imgPy6.clone();
                break;
            case 7:
                pyLevelMat = template_imgPy7.clone();
                break;
            default:
                break;
            }

            // 2)计算原始模板图像灰度值方差(pyLevelMat)
            area = pyLevelMat.rows * pyLevelMat.cols;
            {
                //局部区域：灰度值求和(sum1)、灰度值平方求和(sq_sum1)____同上1)
                unsigned char intension_each1;
                sum = 0;
                sq_sum = 0;
                for (int y = 0; y < pyLevelMat.rows; ++y)
                {
                    for (int x = 0; x < pyLevelMat.cols; ++x)
                    {
                        intension_each1 = pyLevelMat.ptr<uchar>(y)[x];
                        sum += intension_each1;
                        sq_sum += intension_each1 * intension_each1;
                    }
                }
            }
            // 3)计算原始模板图像方差(variance1) / 金字塔最高层图像方差(variance)的系数
            float variance1 = std::sqrt(sq_sum - sum * sum / float(area));
            coefficient = variance1 * model_id->template_cfg.num_levels * 2.0 / variance;

            //系数太小，金字塔层数过多，失真太大
            if (coefficient < 0.2) { --model_id->template_cfg.num_levels; }
        }
        while (coefficient < 0.2);
    }
    // 模板创建完毕标志位
    model_id->template_cfg.is_inited = true;

    return true;
}

// 创建模板函数入口（传入实参）
bool CreateTemplate::createTemplate(
    cv::Mat temp,
    cv::Mat mask,
    int num_levels,
    int angle_start,
    int angle_end,
    double angle_step,
    bool create_otsu,
    int min_contrast,
    int max_contrast,
    T_T::Template::Ptr model_id,
    T_T::EdgeMethod edge_method,
    T_T::TemplateOriginMode origin_mode)
{
    if (!model_id)
    {
        std::cerr << "Template creation failed: model_id is empty." << std::endl;
        return false;
    }
    if (edge_method != T_T::EDGE_CURRENT && edge_method != T_T::EDGE_DEVERNAY &&
        edge_method != T_T::EDGE_CANNY_PIXEL)
        return false;
    edge_method_ = edge_method;
    model_id->template_cfg.edge_method = edge_method;
    if (origin_mode != T_T::ORIGIN_IMAGE_CENTER &&
        origin_mode != T_T::ORIGIN_DOMAIN_CENTROID)
        return false;
    if (temp.empty() || mask.empty())
    {
        std::cerr << "Template creation failed: template image and mask must not be empty." << std::endl;
        return false;
    }
    if (temp.size() != mask.size())
    {
        std::cerr << "Template creation failed: template image and mask sizes differ." << std::endl;
        return false;
    }
    if (num_levels < -1 || num_levels > 7)
    {
        std::cerr << "Template creation failed: num_levels must be -1 or in [0, 7]." << std::endl;
        return false;
    }
    if (!std::isfinite(angle_step) || angle_step <= 0.0 || angle_start > angle_end)
    {
        std::cerr << "Template creation failed: invalid angle range or non-positive finite angle_step." << std::endl;
        return false;
    }
    if ((!create_otsu && (min_contrast < 0 || max_contrast > 255 || min_contrast > max_contrast)))
    {
        std::cerr << "Template creation failed: contrast thresholds must satisfy 0 <= min <= max <= 255." << std::endl;
        return false;
    }

    cv::Mat tempMat, maskMat;
    tempMat = temp.clone();
    maskMat = mask.clone();

    // 如果彩色图像，转灰度图像
    if (tempMat.channels() == 3) { cv::cvtColor(tempMat, tempMat, cv::COLOR_BGR2GRAY); }
    else if (tempMat.channels() == 4) { cv::cvtColor(tempMat, tempMat, cv::COLOR_BGRA2GRAY); }
    else if (tempMat.channels() != 1)
    {
        std::cerr << "Template creation failed: template image must have 1, 3, or 4 channels." << std::endl;
        return false;
    }
    if (maskMat.channels() == 3) { cv::cvtColor(maskMat, maskMat, cv::COLOR_BGR2GRAY); }
    else if (maskMat.channels() == 4) { cv::cvtColor(maskMat, maskMat, cv::COLOR_BGRA2GRAY); }
    else if (maskMat.channels() != 1)
    {
        std::cerr << "Template creation failed: mask must have 1, 3, or 4 channels." << std::endl;
        return false;
    }
    if (tempMat.depth() != CV_8U || maskMat.depth() != CV_8U)
    {
        std::cerr << "Template creation failed: template image and mask must be 8-bit images." << std::endl;
        return false;
    }
    // 自动阈值分割
    if (create_otsu == true)
    {
        double otsuthresh = 120;
        double thotsu = 100;
        cv::Mat threshsrc = tempMat.clone();
        cv::Mat threshdst;
        thotsu = cv::threshold(threshsrc, threshdst, otsuthresh, 255, cv::THRESH_OTSU + cv::THRESH_BINARY);
        if (thotsu == 0) { max_contrast = otsuthresh; }
        else
        {
            max_contrast = thotsu;
        }
        min_contrast = max_contrast / 2.5;
    }

    // 设置形状匹配参数
    model_id->template_cfg.num_levels = num_levels;
    model_id->template_cfg.angle_start = angle_start;
    model_id->template_cfg.angle_end = angle_end;
    model_id->template_cfg.angle_step = angle_step;
    model_id->template_cfg.create_otsu = create_otsu;
    model_id->template_cfg.min_contrast = min_contrast;
    model_id->template_cfg.max_contrast = max_contrast;
    // 保留调用者预先指定的模板 ID；未指定时使用兼容性的默认 ID 1。
    if (model_id->template_cfg.id <= 0) model_id->template_cfg.id = 1;
    model_id->template_cfg.image_width = tempMat.cols;
    model_id->template_cfg.image_height = tempMat.rows;
    model_id->template_cfg.origin_mode = origin_mode;
    if (origin_mode == T_T::ORIGIN_DOMAIN_CENTROID) {
        const cv::Moments moments = cv::moments(maskMat, true);
        if (moments.m00 > 1e-9) {
            model_id->template_cfg.origin_x = moments.m10 / moments.m00;
            model_id->template_cfg.origin_y = moments.m01 / moments.m00;
        } else {
            model_id->template_cfg.origin_mode = T_T::ORIGIN_IMAGE_CENTER;
            model_id->template_cfg.origin_x = tempMat.cols * 0.5;
            model_id->template_cfg.origin_y = tempMat.rows * 0.5;
        }
    } else {
        model_id->template_cfg.origin_x = tempMat.cols * 0.5;
        model_id->template_cfg.origin_y = tempMat.rows * 0.5;
    }
    // 由模板图像确定金字塔层数：-1则为自动设置层数
    if (num_levels == -1)
    {
        int pry_length = _convertLength(MAX(tempMat.rows, tempMat.cols));
        if (pry_length < 32) { model_id->template_cfg.num_levels = 0; }
        if (pry_length > 1024) { model_id->template_cfg.num_levels = 7; }
        switch (pry_length)
        {
        case 32:
            model_id->template_cfg.num_levels = 1;
            break;
        case 64:
            model_id->template_cfg.num_levels = 2;
            break;
        case 128:
            model_id->template_cfg.num_levels = 3;
            break;
        case 256:
            model_id->template_cfg.num_levels = 4;
            break;
        case 512:
            model_id->template_cfg.num_levels = 5;
            break;
        case 1024:
            model_id->template_cfg.num_levels = 6;
            break;
        default:
            break;
        }

    }
    else // 金字塔层数 = 0、1、2、3、4、5、6、7
    {
        model_id->template_cfg.num_levels = num_levels;
    }
    // 初始化model_id
    CreateTemplate::_initialShapeModel(model_id);

    // model_id存储的特征点坐标（模板中心点为原点坐标）
    if (!CreateTemplate::_createModel(tempMat, maskMat, model_id))
    {
        model_id->templates.clear();
        model_id->template_cfg.is_inited = false;
        model_id->is_empty = true;
        return false;
    }

    //金字塔层数优化：根据金字塔每层的特征点（轮廓点）的数量（>20）来定义金字塔层数
    if (num_levels == -1)
    {
        switch (model_id->template_cfg.num_levels)
        {
        case 0:
            break;
        case 1:
            //第1层轮廓点数量<20个，则金字塔层数为0
            if (model_id->templates[1]->shape_angle[0]->shape_point.size() < 20)
            {
                model_id->template_cfg.num_levels = 0;
            }
            break;
        case 2:
            //第2层轮廓点数量<20个：
            //****第1层轮廓点数量>20个,则金字塔层数为1；
            //****第1层轮廓点数量<20个,则金字塔层数为0
            if (model_id->templates[2]->shape_angle[0]->shape_point.size() < 20)
            {
                if (model_id->templates[1]->shape_angle[0]->shape_point.size() > 20)
                {
                    model_id->template_cfg.num_levels = 1;
                }
                else
                {
                    model_id->template_cfg.num_levels = 0;
                }
            }
            break;
        case 3:
            //第3层轮廓点数量<20个：
            //****第2层轮廓点数量>20个,则金字塔层数为2:
            //********第1层轮廓点数量>20个,则金字塔层数为1；
            //********第1层轮廓点数量<20个,则金字塔层数为0
            if (model_id->templates[3]->shape_angle[0]->shape_point.size() < 20)
            {
                if (model_id->templates[2]->shape_angle[0]->shape_point.size() > 20)
                {
                    model_id->template_cfg.num_levels = 2;
                }
                else
                {
                    if (model_id->templates[1]->shape_angle[0]->shape_point.size() > 20)
                    {
                        model_id->template_cfg.num_levels = 1;
                    }
                    else
                    {
                        model_id->template_cfg.num_levels = 0;
                    }
                }
            }
            break;
        case 4:
            //第4层轮廓点数量<20个：
            //****第3层轮廓点数量>20个,则金字塔层数为3；
            //********第2层轮廓点数量>20个,则金字塔层数为2:
            //************第1层轮廓点数量>20个,则金字塔层数为1;
            //************第1层轮廓点数量<20个,则金字塔层数为0;
            if (model_id->templates[4]->shape_angle[0]->shape_point.size() < 20)
            {
                if (model_id->templates[3]->shape_angle[0]->shape_point.size() > 20)
                {
                    model_id->template_cfg.num_levels = 3;
                }
                else
                {
                    if (model_id->templates[2]->shape_angle[0]->shape_point.size() > 20)
                    {
                        model_id->template_cfg.num_levels = 2;
                    }
                    else
                    {
                        if (model_id->templates[1]->shape_angle[0]->shape_point.size() > 20)
                        {
                            model_id->template_cfg.num_levels = 1;
                        }
                        else
                        {
                            model_id->template_cfg.num_levels = 0;
                        }
                    }
                }
            }
            break;
        case 5:
            //同上
            if (model_id->templates[5]->shape_angle[0]->shape_point.size() < 20)
            {
                if (model_id->templates[4]->shape_angle[0]->shape_point.size() > 20)
                {
                    model_id->template_cfg.num_levels = 4;
                }
                else
                {
                    if (model_id->templates[3]->shape_angle[0]->shape_point.size() > 20)
                    {
                        model_id->template_cfg.num_levels = 3;
                    }
                    else
                    {
                        if (model_id->templates[2]->shape_angle[0]->shape_point.size() > 20)
                        {
                            model_id->template_cfg.num_levels = 2;
                        }
                        else
                        {
                            if (model_id->templates[1]->shape_angle[0]->shape_point.size() > 20)
                            {
                                model_id->template_cfg.num_levels = 1;
                            }
                            else
                            {
                                model_id->template_cfg.num_levels = 0;
                            }
                        }
                    }
                }
            }
            break;
        case 6:
            //同上
            if (model_id->templates[6]->shape_angle[0]->shape_point.size() < 20)
            {
                if (model_id->templates[5]->shape_angle[0]->shape_point.size() > 20)
                {
                    model_id->template_cfg.num_levels = 5;
                }
                else
                {
                    if (model_id->templates[4]->shape_angle[0]->shape_point.size() > 20)
                    {
                        model_id->template_cfg.num_levels = 4;
                    }
                    else
                    {
                        if (model_id->templates[3]->shape_angle[0]->shape_point.size() > 20)
                        {
                            model_id->template_cfg.num_levels = 3;
                        }
                        else
                        {
                            if (model_id->templates[2]->shape_angle[0]->shape_point.size() > 20)
                            {
                                model_id->template_cfg.num_levels = 2;
                            }
                            else
                            {
                                if (model_id->templates[1]->shape_angle[0]->shape_point.size() > 20)
                                {
                                    model_id->template_cfg.num_levels = 1;
                                }
                                else
                                {
                                    model_id->template_cfg.num_levels = 0;
                                }
                            }
                        }
                    }
                }
            }
            break;
        case 7:
            //同上
            if (model_id->templates[7]->shape_angle[0]->shape_point.size() < 20)
            {
                if (model_id->templates[6]->shape_angle[0]->shape_point.size() > 20)
                {
                    model_id->template_cfg.num_levels = 6;
                }
                else
                {
                    if (model_id->templates[5]->shape_angle[0]->shape_point.size() > 20)
                    {
                        model_id->template_cfg.num_levels = 5;
                    }
                    else
                    {
                        if (model_id->templates[4]->shape_angle[0]->shape_point.size() > 20)
                        {
                            model_id->template_cfg.num_levels = 4;
                        }
                        else
                        {
                            if (model_id->templates[3]->shape_angle[0]->shape_point.size() > 20)
                            {
                                model_id->template_cfg.num_levels = 3;
                            }
                            else
                            {
                                if (model_id->templates[2]->shape_angle[0]->shape_point.size() > 20)
                                {
                                    model_id->template_cfg.num_levels = 2;
                                }
                                else
                                {
                                    if (model_id->templates[1]->shape_angle[0]->shape_point.size() > 20)
                                    {
                                        model_id->template_cfg.num_levels = 1;
                                    }
                                    else
                                    {
                                        model_id->template_cfg.num_levels = 0;
                                    }
                                }
                            }
                        }
                    }
                }
            }
            break;
        default:
            break;
        }
    }
    else
    {
        model_id->template_cfg.num_levels = num_levels;
    }

    if (num_levels == -1)
    {
        // 细长或低纹理模板不应在顶层被压缩到过少像素/特征。
        // 保证粗匹配层仍有足够的几何辨识度，避免正确候选在顶层丢失。
        while (model_id->template_cfg.num_levels > 0)
        {
            const int level = model_id->template_cfg.num_levels;
            const int scale = 1 << level;
            const int level_width = (tempMat.cols + scale - 1) / scale;
            const int level_height = (tempMat.rows + scale - 1) / scale;
            const size_t feature_count = model_id->templates[level]->shape_angle[0]->shape_point.size();
            // 自动模型保留既有质量下限 40；兼容 HALCON 的安全要求是至少四个点，
            // 因此稀疏层会回退而不会暴露给匹配器。
            if (std::min(level_width, level_height) >= 8 && feature_count >= 40) { break; }
            --model_id->template_cfg.num_levels;
        }
    }
    else if (model_id->template_cfg.num_levels > 0) {
        const int level = model_id->template_cfg.num_levels;
        const size_t feature_count = model_id->templates[level] &&
            !model_id->templates[level]->shape_angle.empty() &&
            model_id->templates[level]->shape_angle[0]
            ? model_id->templates[level]->shape_angle[0]->shape_point.size() : 0;
        if (feature_count < 4) {
            std::cerr << "Template creation failed: requested top pyramid level has fewer than 4 features."
                      << std::endl;
            model_id->templates.clear();
            model_id->template_cfg.is_inited = false;
            model_id->is_empty = true;
            return false;
        }
    }
    model_id->is_empty = false;
    model_id->is_inited = true;
    return true;
}

// 保存模板 JSON 文件
bool CreateTemplate::saveModelFile2Json(T_T::Template::Ptr model_id, std::string path)
{
    int num_pyramid = model_id->templates.size();
    std::string model_name = path;
    cv::FileStorage fs(model_name, cv::FileStorage::WRITE);

    // ShapeMatch 模型
    fs << "shapeMatch";
    fs << "{";

    fs << "angle_start" << model_id->template_cfg.angle_start; //模板制作-起始角度
    fs << "angle_end" << model_id->template_cfg.angle_end; //模板制作-终止角度
    fs << "angle_step" << model_id->template_cfg.angle_step; //模板制作-角度步长
    fs << "auto_threshold" << model_id->template_cfg.create_otsu; //模板制作-自动阈值设置（false->0;true->1）
    fs << "min_constract" << model_id->template_cfg.min_contrast; //模板制作-低阈值
    fs << "max_constract" << model_id->template_cfg.max_contrast; //模板制作-高阈值
    fs << "num_levels" << model_id->template_cfg.num_levels;
    fs << "id" << model_id->template_cfg.id;
    fs << "image_width" << model_id->template_cfg.image_width;
    fs << "image_height" << model_id->template_cfg.image_height;
    fs << "is_inited" << model_id->template_cfg.is_inited;
    fs << "edge_method" << static_cast<int>(model_id->template_cfg.edge_method);
    fs << "origin_mode" << static_cast<int>(model_id->template_cfg.origin_mode);
    fs << "origin_x" << model_id->template_cfg.origin_x;
    fs << "origin_y" << model_id->template_cfg.origin_y;

    //保存模板制作产生的特征[金字塔每层的0°角度的特征点]
    fs << "templates"
        << "[";
    {
        for (int i = 0; i < num_pyramid; i++) //每层
        {
            auto templ_templates = model_id->templates[i];
            fs << "{";
            fs << "template_py_number" << int(i);
            fs << "template_pyramid"
                << "[";
            {
                // 遍历每个角度
                auto templ_angle = templ_templates->shape_angle[0];
                fs << "{";
                fs << "angle" << templ_angle->angle;
                fs << "features"
                    << "[";
                {
                    for (int k = 0; k < templ_angle->shape_point.size(); k++) //每个点
                    {
                        auto templ_feature = templ_angle->shape_point[k];
                        fs << "[:" << templ_feature.x << templ_feature.y << templ_feature.edge_dx << templ_feature.
                            edge_dy
                            << /* 不写入梯度幅值 */ "]";
                    }
                }
                fs << "]";
                fs << "}";
            }
            fs << "]";
            fs << "}";
        }
    }
    fs << "]";
    fs << "}";

    fs.release();
    std::cout << "Template file saved successfully [JSON]." << std::endl;
    return true;
}

// 保存模板为二进制文件
bool CreateTemplate::saveModelFile2Binary(T_T::Template::Ptr model_id, std::string path)
{
    int num_pyramid = model_id->templates.size();

    std::ofstream ofs(path, std::ios::binary);
    if (!ofs.is_open())
    {
        std::cerr << "Failed to open file for saving: " << path << std::endl;
        return false;
    }

    // 保存模板的配置信息
    ofs.write(reinterpret_cast<char*>(&model_id->template_cfg.angle_start), sizeof(model_id->template_cfg.angle_start));
    ofs.write(reinterpret_cast<char*>(&model_id->template_cfg.angle_end), sizeof(model_id->template_cfg.angle_end));
    ofs.write(reinterpret_cast<char*>(&model_id->template_cfg.angle_step), sizeof(model_id->template_cfg.angle_step));
    ofs.write(reinterpret_cast<char*>(&model_id->template_cfg.create_otsu), sizeof(model_id->template_cfg.create_otsu));
    ofs.write(reinterpret_cast<char*>(&model_id->template_cfg.min_contrast),
              sizeof(model_id->template_cfg.min_contrast));
    ofs.write(reinterpret_cast<char*>(&model_id->template_cfg.max_contrast),
              sizeof(model_id->template_cfg.max_contrast));
    ofs.write(reinterpret_cast<char*>(&model_id->template_cfg.num_levels), sizeof(model_id->template_cfg.num_levels));
    ofs.write(reinterpret_cast<char*>(&model_id->template_cfg.id), sizeof(model_id->template_cfg.id));
    ofs.write(reinterpret_cast<char*>(&model_id->template_cfg.image_width), sizeof(model_id->template_cfg.image_width));
    ofs.write(reinterpret_cast<char*>(&model_id->template_cfg.image_height),
              sizeof(model_id->template_cfg.image_height));
    ofs.write(reinterpret_cast<char*>(&model_id->template_cfg.is_inited), sizeof(model_id->template_cfg.is_inited));

    // 保存金字塔层数
    ofs.write(reinterpret_cast<char*>(&num_pyramid), sizeof(num_pyramid));

    // 保存每一层的模板数据
    for (int i = 0; i < num_pyramid; ++i)
    {
        auto templ_templates = model_id->templates[i];
        int num_angles = templ_templates->shape_angle.size();

        // 保存每层的角度数
        ofs.write(reinterpret_cast<char*>(&num_angles), sizeof(num_angles));

        for (int j = 0; j < num_angles; ++j)
        {
            auto templ_angle = templ_templates->shape_angle[j];
            ofs.write(reinterpret_cast<char*>(&templ_angle->angle), sizeof(templ_angle->angle));

            int num_points = templ_angle->shape_point.size();
            ofs.write(reinterpret_cast<char*>(&num_points), sizeof(num_points));

            // 保存每个角度下的特征点数据
            for (int k = 0; k < num_points; ++k)
            {
                auto templ_feature = templ_angle->shape_point[k];
                ofs.write(reinterpret_cast<char*>(&templ_feature.x), sizeof(templ_feature.x));
                ofs.write(reinterpret_cast<char*>(&templ_feature.y), sizeof(templ_feature.y));
                ofs.write(reinterpret_cast<char*>(&templ_feature.edge_dx), sizeof(templ_feature.edge_dx));
                ofs.write(reinterpret_cast<char*>(&templ_feature.edge_dy), sizeof(templ_feature.edge_dy));
            }
        }
    }

    // 只追加元数据，保证旧版二进制负载与旧读取器保持逐字节兼容。
    const uint32_t metadataMagic = 0x534D4554u; // 元数据魔数 "SMET"
    const uint32_t metadataVersion = 2u;
    const uint8_t edgeMethod = static_cast<uint8_t>(model_id->template_cfg.edge_method);
    const uint8_t originMode = static_cast<uint8_t>(model_id->template_cfg.origin_mode);
    const uint8_t reserved[2] = {0, 0};
    ofs.write(reinterpret_cast<const char*>(&metadataMagic), sizeof(metadataMagic));
    ofs.write(reinterpret_cast<const char*>(&metadataVersion), sizeof(metadataVersion));
    ofs.write(reinterpret_cast<const char*>(&edgeMethod), sizeof(edgeMethod));
    ofs.write(reinterpret_cast<const char*>(&originMode), sizeof(originMode));
    ofs.write(reinterpret_cast<const char*>(reserved), sizeof(reserved));
    ofs.write(reinterpret_cast<const char*>(&model_id->template_cfg.origin_x), sizeof(model_id->template_cfg.origin_x));
    ofs.write(reinterpret_cast<const char*>(&model_id->template_cfg.origin_y), sizeof(model_id->template_cfg.origin_y));

    ofs.close(); // 关闭文件流
    std::cout << "Template file saved successfully [binary]." << std::endl;
    return true;
}


// 获取模板轮廓特征点信息(0,1,2,3...)
std::vector<cv::Point2d> CreateTemplate::getTemplatePointPyramid(T_T::Template::Ptr model_id, int num_level)
{
    if (model_id == nullptr)
    {
        std::cout << "ShapeMatchTemplate is nullptr!" << std::endl;
        return std::vector<cv::Point2d>();
    }
    std::vector<cv::Point2d> result_points;
    // 打印金字塔层级信息
    if (num_level <= model_id->template_cfg.num_levels)
    {
        const double scale = static_cast<double>(1 << std::max(0, num_level));
        const double origin_x = model_id->template_cfg.origin_mode == T_T::ORIGIN_DOMAIN_CENTROID
            ? model_id->template_cfg.origin_x / scale
            : (model_id->template_cfg.image_width / scale) * 0.5;
        const double origin_y = model_id->template_cfg.origin_mode == T_T::ORIGIN_DOMAIN_CENTROID
            ? model_id->template_cfg.origin_y / scale
            : (model_id->template_cfg.image_height / scale) * 0.5;
        for (int i = 0; i < model_id->templates[num_level]->shape_angle[0]->shape_point.size(); i++)
        {
            cv::Point2d point_one(
                origin_x + model_id->templates[num_level]->shape_angle[0]->shape_point[i].x,
                origin_y + model_id->templates[num_level]->shape_angle[0]->shape_point[i].y);
            result_points.push_back(point_one);
        }
    }
    else
    {
        result_points.clear();
    }

    return result_points;
}

bool CreateTemplate::drawPyramidFeatures(const cv::Mat& template_image,
                                         const T_T::Template::Ptr& model_id,
                                         cv::Mat& output) const
{
    output.release();
    if (template_image.empty() || !model_id || !model_id->is_inited ||
        model_id->template_cfg.num_levels < 0 || model_id->templates.empty()) return false;

    cv::Mat gray;
    if (template_image.channels() == 1) gray = template_image.clone();
    else if (template_image.channels() == 3)
        cv::cvtColor(template_image, gray, cv::COLOR_BGR2GRAY);
    else if (template_image.channels() == 4)
        cv::cvtColor(template_image, gray, cv::COLOR_BGRA2GRAY);
    else return false;
    if (gray.depth() != CV_8U) return false;

    const int last_level = std::min(model_id->template_cfg.num_levels,
                                    static_cast<int>(model_id->templates.size()) - 1);
    std::vector<cv::Mat> pyramid(1, gray);
    for (int level = 1; level <= last_level; ++level)
    {
        if (pyramid.back().cols < 2 || pyramid.back().rows < 2) return false;
        cv::Mat next;
        cv::pyrDown(pyramid.back(), next,
                    cv::Size(pyramid.back().cols / 2, pyramid.back().rows / 2));
        pyramid.push_back(next);
    }

    const int margin = 18;
    const int gap = 24;
    int canvas_width = margin * 2;
    int canvas_height = margin * 2;
    for (int level = last_level; level >= 0; --level)
    {
        canvas_width += pyramid[level].cols;
        canvas_height += pyramid[level].rows;
        if (level != 0) { canvas_width += gap; canvas_height += gap; }
    }
    output = cv::Mat(canvas_height, canvas_width, CV_8UC3, cv::Scalar(18, 18, 18));

    int x = margin;
    int y = margin;
    for (int level = last_level; level >= 0; --level)
    {
        cv::Mat tile;
        cv::cvtColor(pyramid[level], tile, cv::COLOR_GRAY2BGR);
        const auto& shape_info = model_id->templates[level];
        if (shape_info && !shape_info->shape_angle.empty() && shape_info->shape_angle[0])
        {
            const double scale = static_cast<double>(1 << level);
            const double origin_x = model_id->template_cfg.origin_mode == T_T::ORIGIN_DOMAIN_CENTROID
                ? model_id->template_cfg.origin_x / scale : tile.cols * 0.5;
            const double origin_y = model_id->template_cfg.origin_mode == T_T::ORIGIN_DOMAIN_CENTROID
                ? model_id->template_cfg.origin_y / scale : tile.rows * 0.5;
            for (const auto& feature : shape_info->shape_angle[0]->shape_point)
            {
                const cv::Point point(cvRound(origin_x + feature.x),
                                      cvRound(origin_y + feature.y));
                if (static_cast<unsigned>(point.x) < static_cast<unsigned>(tile.cols) &&
                    static_cast<unsigned>(point.y) < static_cast<unsigned>(tile.rows))
                {
                    tile.at<cv::Vec3b>(point.y, point.x) = cv::Vec3b(0, 255, 0);
                }
            }
        }
        cv::rectangle(output, cv::Rect(x - 1, y - 1, tile.cols + 2, tile.rows + 2),
                      cv::Scalar(105, 105, 105), 1);
        tile.copyTo(output(cv::Rect(x, y, tile.cols, tile.rows)));
        cv::putText(output, "L" + std::to_string(level) + "  " +
                    std::to_string(tile.cols) + "x" + std::to_string(tile.rows),
                    cv::Point(x, std::max(13, y - 5)), cv::FONT_HERSHEY_SIMPLEX,
                    0.38, cv::Scalar(230, 230, 230), 1, cv::LINE_AA);
        x += tile.cols + gap;
        y += tile.rows + gap;
    }
    return true;
}
