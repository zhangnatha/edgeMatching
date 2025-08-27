#include "FindTemplateV1.h"
#include <algorithm>
#include <cmath>
#include <omp.h>
#include <thread>
#include <fstream>
#include <immintrin.h>

using namespace SM_V1;

#define DEBUG_SHOW 0 //用于观测算法过程运行结果显示
#define COSTTIME_SHOW 1 //耗时统计-用于算法优化观测时间
#define DEBUG_COARSE_SHOW 0 //最高层金字塔粗匹配整个可视化过程

SearchTemplate::SearchTemplate() = default;
SearchTemplate::~SearchTemplate() = default;

// 将输入长度转换为最接近的2的幂
int SearchTemplate::_convertLength(int length_src)
{
    for (int i = 4;; i++)
    {
        int temp = static_cast<int>(pow(2.0, i));
        if (temp >= length_src)
        {
            length_src = temp;
            break;
        }
    }
    return length_src;
}

void SearchTemplate::_gaussianFilter(uint8_t* corrupted, uint8_t* smooth, int width, int height, bool useSIMD)
{
    // 高斯模板 (5x5) 共25个元素
    int templates[25] = {1, 4, 7, 4, 1, 4, 16, 26, 16, 4, 7, 26, 41, 26, 7, 4, 16, 26, 16, 4, 1, 4, 7, 4, 1};

    // 复制原始图像到平滑图像
    memcpy(smooth, corrupted, width * height * sizeof(uint8_t));

    if (useSIMD)
    {
        // 使用 AVX2 对图像进行高斯滤波
        for (int j = 2; j < height - 2; j++)
        {
            for (int i = 2; i < width - 2; i++)
            {
                // 加权和初始为0
                __m256i sum = _mm256_setzero_si256();
                int index = 0;

                for (int m = j - 2; m < j + 3; m++)
                {
                    for (int n = i - 2; n < i + 3; n++)
                    {
                        // 将数据加载到AVX寄存器中
                        __m256i pixel = _mm256_set1_epi32(corrupted[m * width + n]); // 将像素值复制到寄存器
                        int weight = templates[index++];

                        // 乘以高斯模板值并累加
                        sum = _mm256_adds_epu8(sum, _mm256_mullo_epi32(pixel, _mm256_set1_epi32(weight)));
                    }
                }

                // 将结果除以273（高斯核的总和）
                sum = _mm256_srli_epi32(sum, 8); // 除以273（通过右移8位）

                // 处理结果值，防止超过255
                sum = _mm256_min_epu8(sum, _mm256_set1_epi8(255));

                // 存储结果
                smooth[j * width + i] = (uint8_t)_mm256_extract_epi8(sum, 0);
            }
        }
    }
    else
    {
        // 常规逐像素高斯滤波
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
}

bool SearchTemplate::_maxOverlap(const cv::RotatedRect rect1, const cv::RotatedRect& rect2, float overlap)
{
    std::vector<cv::Point2f> inter_section;
    // 计算两个旋转矩形的相交的情况
    int ret = cv::rotatedRectangleIntersection(rect1, rect2, inter_section); //两旋转矩形的相交多边形的点(max=8)
    bool rb = false;
    if (ret == 0) //没有
    {
        rb = false;
    }
    else if (ret == 2) //包含
    {
        rb = true;
    }
    else if (ret == 1) //有
    {
        float inter_area = 0;
        // 计算相交的多边形面积
        inter_area = cv::contourArea(inter_section);
        float lap = inter_area / (rect1.size.width * rect1.size.height);
        rb = lap > overlap;
    }
    return rb;
}

std::vector<T_T::MatchResult> SearchTemplate::_filterNearCandidates(const std::vector<T_T::MatchResult>& input)
{
    std::vector<T_T::MatchResult> result;
    bool nearFlag = false;
    for (const auto& c : input)
    {
        //遍历所有已记录的结果
        nearFlag = false;
        for (auto& r : result)
        {
            //遍历所有已选择的结果
            if (std::abs(r.pose.x - c.pose.x) < 5 && std::abs(r.pose.y - c.pose.y) < 5)
            {
                //当结果位置相近时，竞选出一个结果保存
                nearFlag = true;
                if (c.score > r.score)
                {
                    //当未保存的结果更优时，进行替换
                    r = c;
                    break;
                }
            }
        }
        if (!nearFlag) { result.push_back(c); }
    }
    //按照得分从高到低进行排序
    std::sort(result.begin(), result.end(), [](const T_T::MatchResult& c1, const T_T::MatchResult& c2)
    {
        return c1.score > c2.score;
    });
    return result;
}

std::vector<T_T::MatchResult> SearchTemplate::_filterMaxOverLapCandidates(
    const std::vector<T_T::MatchResult>& input,
    float max_ovelap,
    int model_height,
    int model_width)
{
    std::vector<T_T::MatchResult> result;
    bool overlapFlag = false;
    for (const auto& c : input)
    {
        //遍历所有已记录的结果
        overlapFlag = false;
        for (auto& r : result)
        {
            //遍历所有已选择的结果
            cv::RotatedRect rect1(cv::Point2f(r.pose.x, r.pose.y), cv::Size2f(model_width, model_height),
                                  r.pose.angle + 180);
            cv::RotatedRect rect2(cv::Point2f(c.pose.x, c.pose.y), cv::Size2f(model_width, model_height),
                                  c.pose.angle + 180);
            if (_maxOverlap(rect1, rect2, max_ovelap))
            {
                //当结果位置相近时，竞选出一个结果保存
                overlapFlag = true;
                if (c.score > r.score)
                {
                    //当未保存的结果更优时，进行替换
                    r = c;
                    break;
                }
            }
        }
        if (!overlapFlag) { result.push_back(c); }
    }
    //按照得分从高到低进行排序
    std::sort(result.begin(), result.end(), [](const T_T::MatchResult& c1, const T_T::MatchResult& c2)
    {
        return c1.score > c2.score;
    });
    return result;
}

// 获取特征信息
void SearchTemplate::_getFeature(
    cv::Mat search_image,
    cv::Mat mask_image,
    int width,
    int height,
    std::vector<float>& p_buf_gradX,
    std::vector<float>& p_buf_gradY,
    bool useSIMD)
{
    // 分配内存
    uint32_t bufferSize = search_image.cols * search_image.rows;
    uint8_t* pInput = (uint8_t*)malloc(bufferSize * sizeof(uint8_t));

    uint8_t* SearchImage = static_cast<uint8_t*>(search_image.data);
    _gaussianFilter(SearchImage, pInput, width, height, true);

    // 待测图像的掩模图
    uint8_t* maskdata = static_cast<uint8_t*>(mask_image.data);

    // 提取待测图像的梯度信息
    if (useSIMD)
    {
        const __m256  vZero   = _mm256_setzero_ps();
        const __m256  vEps    = _mm256_set1_ps(1e-6f);
        const __m256i v255_i  = _mm256_set1_epi32(0xFF);

        for (int j = 1; j < height - 1; ++j)
        {
            // 向量化范围：一次 16 像素
            int i = 1;
            for (; i + 15 < width - 1; i += 16)
            {
                const int idx = j * width + i;

                // 载入 16 字节的左右/上下/掩膜
                __m128i left8  = _mm_loadu_si128((const __m128i*)(pInput   + idx - 1));
                __m128i right8 = _mm_loadu_si128((const __m128i*)(pInput   + idx + 1));
                __m128i up8    = _mm_loadu_si128((const __m128i*)(pInput   + idx - width));
                __m128i down8  = _mm_loadu_si128((const __m128i*)(pInput   + idx + width));
                __m128i msk8   = _mm_loadu_si128((const __m128i*)(maskdata + idx));

                // 扩展到 16 位（无符号->有符号容器）
                __m256i L16 = _mm256_cvtepu8_epi16(left8);
                __m256i R16 = _mm256_cvtepu8_epi16(right8);
                __m256i U16 = _mm256_cvtepu8_epi16(up8);
                __m256i D16 = _mm256_cvtepu8_epi16(down8);
                __m256i M16 = _mm256_cvtepu8_epi16(msk8);

                // 带符号差分（右-左、下-上），范围约 [-255, 255]
                __m256i DX16 = _mm256_sub_epi16(R16, L16);
                __m256i DY16 = _mm256_sub_epi16(D16, U16);

                // 拆成低/高 128 位，再从 i16 扩到 i32、再转 float
                __m128i DX_lo128 = _mm256_castsi256_si128(DX16);
                __m128i DX_hi128 = _mm256_extracti128_si256(DX16, 1);
                __m128i DY_lo128 = _mm256_castsi256_si128(DY16);
                __m128i DY_hi128 = _mm256_extracti128_si256(DY16, 1);

                __m256i DX32_lo = _mm256_cvtepi16_epi32(DX_lo128);
                __m256i DX32_hi = _mm256_cvtepi16_epi32(DX_hi128);
                __m256i DY32_lo = _mm256_cvtepi16_epi32(DY_lo128);
                __m256i DY32_hi = _mm256_cvtepi16_epi32(DY_hi128);

                __m256 DXf_lo = _mm256_cvtepi32_ps(DX32_lo);
                __m256 DXf_hi = _mm256_cvtepi32_ps(DX32_hi);
                __m256 DYf_lo = _mm256_cvtepi32_ps(DY32_lo);
                __m256 DYf_hi = _mm256_cvtepi32_ps(DY32_hi);

                // |g| = sqrt(dx^2 + dy^2)，并用 eps 夹住避免除零
                __m256 mag2_lo = _mm256_add_ps(_mm256_mul_ps(DXf_lo, DXf_lo),
                                               _mm256_mul_ps(DYf_lo, DYf_lo));
                __m256 mag2_hi = _mm256_add_ps(_mm256_mul_ps(DXf_hi, DXf_hi),
                                               _mm256_mul_ps(DYf_hi, DYf_hi));

                __m256 mag_lo = _mm256_sqrt_ps(mag2_lo);
                __m256 mag_hi = _mm256_sqrt_ps(mag2_hi);
                mag_lo = _mm256_max_ps(mag_lo, vEps);
                mag_hi = _mm256_max_ps(mag_hi, vEps);

                // 单位梯度
                __m256 NX_lo = _mm256_div_ps(DXf_lo, mag_lo);
                __m256 NY_lo = _mm256_div_ps(DYf_lo, mag_lo);
                __m256 NX_hi = _mm256_div_ps(DXf_hi, mag_hi);
                __m256 NY_hi = _mm256_div_ps(DYf_hi, mag_hi);

                // 掩膜：等于 255 的保留，其余置 0
                __m128i M_lo128 = _mm256_castsi256_si128(M16);
                __m128i M_hi128 = _mm256_extracti128_si256(M16, 1);
                __m256i M32_lo  = _mm256_cvtepi16_epi32(M_lo128);
                __m256i M32_hi  = _mm256_cvtepi16_epi32(M_hi128);
                __m256i Meq_lo  = _mm256_cmpeq_epi32(M32_lo, v255_i);
                __m256i Meq_hi  = _mm256_cmpeq_epi32(M32_hi, v255_i);
                __m256  Mmask_lo = _mm256_castsi256_ps(Meq_lo);
                __m256  Mmask_hi = _mm256_castsi256_ps(Meq_hi);

                __m256 outX_lo = _mm256_blendv_ps(vZero, NX_lo, Mmask_lo);
                __m256 outY_lo = _mm256_blendv_ps(vZero, NY_lo, Mmask_lo);
                __m256 outX_hi = _mm256_blendv_ps(vZero, NX_hi, Mmask_hi);
                __m256 outY_hi = _mm256_blendv_ps(vZero, NY_hi, Mmask_hi);

                // 写回
                _mm256_storeu_ps(&p_buf_gradX[idx + 0],  outX_lo);
                _mm256_storeu_ps(&p_buf_gradY[idx + 0],  outY_lo);
                _mm256_storeu_ps(&p_buf_gradX[idx + 8],  outX_hi);
                _mm256_storeu_ps(&p_buf_gradY[idx + 8],  outY_hi);
            }

            // 残量（不足 16 个）走标量
            for (; i < width - 1; ++i)
            {
                const int index = j * width + i;
                int16_t sdx = (int16_t)pInput[index + 1]        - (int16_t)pInput[index - 1];
                int16_t sdy = (int16_t)pInput[index + width]    - (int16_t)pInput[index - width];
                float mag = std::sqrt(float(sdx) * float(sdx) + float(sdy) * float(sdy));
                if (mag < 1e-6f) { p_buf_gradX[index] = 0.f; p_buf_gradY[index] = 0.f; }
                else { p_buf_gradX[index] = float(sdx) / mag; p_buf_gradY[index] = float(sdy) / mag; }
                if (maskdata[index] != 0xFF) { p_buf_gradX[index] = 0.f; p_buf_gradY[index] = 0.f; }
            }
        }

        free(pInput);
        return;
    }
    else
    {
        // 常规逐像素梯度计算
        for (int i = 1; i < width - 1; i++)
        {
            for (int j = 1; j < height - 1; j++)
            {
                int index = j * width + i;

                // X方向的梯度：dx = 右 - 左
                // Y方向的梯度：dy = 下 - 上
                int16_t sdx = *(pInput + j * width + i + 1) - *(pInput + j * width + i - 1);
                int16_t sdy = *(pInput + (j + 1) * width + i) - *(pInput + (j - 1) * width + i);

                float magnitude = std::sqrt(static_cast<float>(sdx * sdx) + static_cast<float>(sdy * sdy));

                // 梯度模不等于0（防止除法越界）
                if (!(std::fabs(magnitude) < 1e-6))
                {
                    p_buf_gradX[index] = static_cast<float>(sdx) / magnitude;
                    p_buf_gradY[index] = static_cast<float>(sdy) / magnitude;
                }
                else // 梯度模等于0
                {
                    p_buf_gradX[index] = 0;
                    p_buf_gradY[index] = 0;
                }

                // 待搜索图像的掩膜，如果不是白色部分，即用户涂黑的部分，该部分特征值为0,不进行计算
                if (*(maskdata + index) != 0xff)
                {
                    p_buf_gradX[index] = 0;
                    p_buf_gradY[index] = 0;
                }
            }
        }
    }

    // 释放内存
    free(pInput);
}

// 水平求和函数 __m256
static inline float hsum_ps_avx(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_hadd_ps(lo, lo);
    lo = _mm_hadd_ps(lo, lo);
    return _mm_cvtss_f32(lo);
}

// 待测图像精匹配
void SearchTemplate::_fineMatching(
    cv::Mat search_image,
    cv::Mat mask_image,
    T_T::ShapeInfo::Ptr shape_info_vec,
    int py_levels,
    int width,
    int height,
    float min_score,
    float greediness,
    T_T::SearchCfg search_region,
    T_T::MatchResult* result_list, bool useSIMD)
{
    // 计算图像的像素大小
    uint32_t bufferSize = width * height;

    // 定义存储用中间变量dx/dy
    std::vector<float> pBufGradX_new; //存取x方向偏导数Gx
    std::vector<float> pBufGradY_new; //存取y方向偏导数Gy

    // 初始化
    pBufGradX_new.resize(bufferSize);
    pBufGradY_new.resize(bufferSize);

    // 处理移动滑窗步长,皆为1
    int ijstep = (py_levels == 0) ? 1 : 1;

    // 获取每个像素的梯度信息：dx/dy
    _getFeature(search_image, mask_image, width, height, pBufGradX_new, pBufGradY_new, true);

    // 相似度计算
    float TempScore = 0; //精匹配，取最大分数值

    int limit_angle; // 0-左限位越界 1-左右限均不越界 2-右限位越界
    // start_angle 左限位越界
    if (start_angle_ <= search_region.stop_angle && search_region.stop_angle <= stop_angle_ && search_region.start_angle
        < start_angle_)
    {
        limit_angle = 0;
    }
    // start_angle 与 stop_angle 左右限均不越界
    if (start_angle_ <= search_region.start_angle && search_region.start_angle <= stop_angle_ && start_angle_ <=
        search_region.stop_angle &&
        search_region.stop_angle <= stop_angle_)
    {
        limit_angle = 1;
    }
    // stop_angle 右限位越界
    if (start_angle_ <= search_region.start_angle && search_region.start_angle <= stop_angle_ && search_region.
        stop_angle > stop_angle_)
    {
        limit_angle = 2;
    }
    int angle_range = std::abs(stop_angle_ - start_angle_);
#pragma omp parallel for schedule(dynamic) // 并行化角度循环
    for (int k = 0; k < shape_info_vec->shape_angle.size(); k++) //[0]角度的个数
    {
        float resultscore = 0;

        // 过滤角度不在待搜索范围的匹配运算
        // 左限位越界[fix]
        if (limit_angle == 0)
        {
            if (shape_info_vec->shape_angle[k]->angle > search_region.stop_angle &&
                shape_info_vec->shape_angle[k]->angle < (search_region.start_angle + angle_range))
            {
                continue;
            }
        }
        // 左右限均不越界
        if (limit_angle == 1)
        {
            if (shape_info_vec->shape_angle[k]->angle < search_region.start_angle || shape_info_vec->shape_angle[k]->
                angle > search_region.stop_angle)
            {
                continue;
            }
        }
        // 右限位越界
        if (limit_angle == 2)
        {
            if (shape_info_vec->shape_angle[k]->angle < search_region.start_angle &&
                shape_info_vec->shape_angle[k]->angle > (search_region.stop_angle - angle_range))
            {
                continue;
            }
        }

        auto shape_angle = shape_info_vec->shape_angle[k];
        int point_size = shape_angle->shape_point.size();

        // 计算模板点的边界框以调整搜索区域，避免边界检查
        int min_dx = INT_MAX, max_dx = INT_MIN, min_dy = INT_MAX, max_dy = INT_MIN;
        for (int mm = 0; mm < point_size; ++mm) {
            int dx = shape_angle->shape_point[mm].x;
            int dy = shape_angle->shape_point[mm].y;
            min_dx = std::min(min_dx, dx);
            max_dx = std::max(max_dx, dx);
            min_dy = std::min(min_dy, dy);
            max_dy = std::max(max_dy, dy);
        }

        int adj_start_X = std::max(search_region.start_X, -min_dx);
        int adj_end_X = std::min(search_region.end_X, width - 1 - max_dx);
        int adj_start_Y = std::max(search_region.start_Y, -min_dy);
        int adj_end_Y = std::min(search_region.end_Y, height - 1 - max_dy);

        // 准备模板点的连续数组，便于SIMD
        std::vector<int> rel_offsets(point_size);
        std::vector<float> tmpl_dx(point_size), tmpl_dy(point_size);
        for (int mm = 0; mm < point_size; ++mm) {
            rel_offsets[mm] = shape_angle->shape_point[mm].y * width + shape_angle->shape_point[mm].x;
            tmpl_dx[mm] = shape_angle->shape_point[mm].edge_dx;
            tmpl_dy[mm] = shape_angle->shape_point[mm].edge_dy;
        }

        int TempPiontX = 0;
        int TempPiontY = 0;

        float anMinScore = min_score - 1;
        float NormMinScore = min_score / point_size;
        float NormGreediness = ((1 - greediness * min_score) / (1 - greediness)) / point_size; //计算贪婪数
#pragma omp parallel for collapse(2) schedule(dynamic) // 并行化搜索区域循环
        for (int i = adj_start_X; i < adj_end_X; i += ijstep)
        {
            for (int j = adj_start_Y; j < adj_end_Y; j += ijstep)
            {
                float PartialSum = 0; //初始化相似性度量分数
                int SumOfCoords = 0;
                float PartialScore = 0;

                int base = j * width + i;

                if (!useSIMD) {
                    // 普通模式：逐点计算，无需边界检查（因调整了搜索区域）
                    for (int m = 0; m < point_size; m++)
                    {
                        /*
                        curX = i + shape_angle->shape_point[m].x; //模板X坐标
                        curY = j + shape_angle->shape_point[m].y; //模板Y坐标
                        int offSet = curY * width + curX;
                        =>
                            offSet = (j + shape_angle->shape_point[m].y) * width + (i + shape_angle->shape_point[m].x)
                        =>
                            offset = [j * width + i] + [shape_angle->shape_point[m].y * width + shape_angle->shape_point[m].x]
                            offset = base + rel_offsets[m]
                        */
                        int offSet = base + rel_offsets[m];
                        float iTx = tmpl_dx[m]; //模板X方向的梯度
                        float iTy = tmpl_dy[m]; //模板Y方向的梯度
                        float iSx = pBufGradX_new[offSet]; //从搜索图像中获取对应的X梯度
                        float iSy = pBufGradY_new[offSet]; //从搜索图像中获取对应的Y梯度

                        //排除梯度为0的点
                        if ((iSx != 0.0f || iSy != 0.0f) && (iTx != 0.0f || iTy != 0.0f))
                        {
                            PartialSum += ((iSx * iTx) + (iSy * iTy))/** (iTm * iSm)*/; // 计算相似度
                        }
                        SumOfCoords = m + 1;
                        PartialScore = PartialSum / SumOfCoords; // 归一化
                        //===================================================================================
                        // 终止策略
                        // Sm<MIN((Smin-1+(1-g*Smin)/(1-g)*(m/n)),(Smin*m/n))
                        //===================================================================================
                        if (PartialScore < (MIN(anMinScore + NormGreediness * SumOfCoords, NormMinScore * SumOfCoords)))
                            break;
                    } //遍历完毕<特征点数>
                } else {
                    // SIMD模式：使用AVX2向量化计算，每8个点一组处理
                    __m256 zero = _mm256_setzero_ps();
                    for (int m = 0; m < point_size; m += 8) {
                        int step = std::min(8, point_size - m);

                        if (step != 8) {
                            // 对于尾部，使用标量处理
                            for (int mm = m; mm < m + step; ++mm) {
                                int offSet = base + rel_offsets[mm];
                                float iTx = tmpl_dx[mm];
                                float iTy = tmpl_dy[mm];
                                float iSx = pBufGradX_new[offSet];
                                float iSy = pBufGradY_new[offSet];

                                if ((iSx != 0.0f || iSy != 0.0f) && (iTx != 0.0f || iTy != 0.0f)) {
                                    PartialSum += (iSx * iTx) + (iSy * iTy);
                                }
                                ++SumOfCoords;
                                PartialScore = PartialSum / SumOfCoords;
                                if (PartialScore < (std::min(anMinScore + NormGreediness * SumOfCoords, NormMinScore * SumOfCoords)))
                                    goto early_exit; // 跳出整个循环
                            }
                            continue;
                        }

                        // 加载模板梯度
                        __m256 tx = _mm256_loadu_ps(&tmpl_dx[m]);
                        __m256 ty = _mm256_loadu_ps(&tmpl_dy[m]);

                        // 加载相对偏移
                        __m256i rel_off = _mm256_loadu_si256((const __m256i*)&rel_offsets[m]);

                        // 计算绝对偏移
                        __m256i abs_off = _mm256_add_epi32(_mm256_set1_epi32(base), rel_off);

                        // Gather图像梯度
                        __m256 sx = _mm256_i32gather_ps(pBufGradX_new.data(), abs_off, sizeof(float));
                        __m256 sy = _mm256_i32gather_ps(pBufGradY_new.data(), abs_off, sizeof(float));

                        // 计算非零掩码
                        __m256 mask_s = _mm256_or_ps(_mm256_cmp_ps(sx, zero, _CMP_NEQ_OQ), _mm256_cmp_ps(sy, zero, _CMP_NEQ_OQ));
                        __m256 mask_t = _mm256_or_ps(_mm256_cmp_ps(tx, zero, _CMP_NEQ_OQ), _mm256_cmp_ps(ty, zero, _CMP_NEQ_OQ));
                        __m256 mask = _mm256_and_ps(mask_s, mask_t);

                        // 计算点积
                        __m256 dot = _mm256_add_ps(_mm256_mul_ps(sx, tx), _mm256_mul_ps(sy, ty));

                        // 应用掩码
                        dot = _mm256_and_ps(mask, dot);

                        // 水平求和并累加到PartialSum
                        PartialSum += hsum_ps_avx(dot);

                        // 更新计数
                        SumOfCoords += 8;

                        // 计算分数并检查终止
                        PartialScore = PartialSum / SumOfCoords;
                        if (PartialScore < (std::min(anMinScore + NormGreediness * SumOfCoords, NormMinScore * SumOfCoords)))
                            goto early_exit;
                    }// 遍历完毕<特征点数>
                }// <SIMD/普通模式>选择结束
early_exit:
#pragma omp critical // 保护共享资源 resultscore 的更新
                {
                    //每个角度下取分数最大的那个匹配结果
                    if (PartialScore > resultscore)
                    {
                        resultscore = PartialScore; // 匹配分数
                        TempPiontX = i; // 坐标X结果值
                        TempPiontY = j; //  坐标Y结果值
                    }
                }
            } // 遍历完毕<搜索区域y>
        } // 遍历完毕<搜索区域x>

#pragma omp critical // 保护共享资源 result_list 的更新
        {
            //所有角度下取分数最大的那个匹配结果
            if (resultscore > TempScore) //精匹配结果更新与记录
            {
                TempScore = resultscore;
                result_list->score = TempScore;
                result_list->pose.x = TempPiontX;
                result_list->pose.y = TempPiontY;
                result_list->pose.angle = shape_angle->angle;
            }
        }
    } // 遍历完毕<角度数量>
}

// 待测图像粗匹配：特征提取和相似性度量
void SearchTemplate::_coarseMatching(
    cv::Mat search_image,
    cv::Mat mask_image,
    T_T::ShapeInfo::Ptr shape_info_vec,
    int width,
    int height,
    int model_width,
    int model_height,
    int left,
    int top,
    float min_score,
    float greediness,
    float max_overlap,
    T_T::SearchCfg search_region,
    std::vector<T_T::MatchResult>& resultList)
{
    // 计算梯度信息存储大小（像素大小:每个像素点的dx/dy/mag）
    uint32_t bufferSize = width * height;

    // 定义存储用中间变量dx/dy
    std::vector<float> pBufGradX(bufferSize); //存取x方向偏导数Gx
    std::vector<float> pBufGradY(bufferSize); //存取y方向偏导数Gy

    std::vector<T_T::MatchResult> totalResultsTemp, resultsfilter;
    std::mutex locker;

    // 提取sobel梯度信息
    _getFeature(search_image, mask_image, width, height, pBufGradX, pBufGradY, true);

    //相似度计算
    int k_size = shape_info_vec->shape_angle.size();
#if !DEBUG_SHOW
#pragma omp parallel for num_threads(thread_num_)
#endif
    for (int k = 0; k < k_size; k++) //角度数量
    {
        auto shape_angle = shape_info_vec->shape_angle[k];
        auto point_size = shape_angle->shape_point.size();
        //过滤角度不在待搜索范围的匹配运算
        if (shape_angle->angle < search_region.start_angle || shape_angle->angle > search_region.stop_angle)
        {
            continue;
        }

        //每个角度下的粗匹配结果: resultsPerDeg
        //每个角度下粗匹配过近，则选得分较大者{筛选1}：resultsPerDegCandidates
        std::vector<T_T::MatchResult> resultsPerDeg, resultsPerDegCandidates;

        // for循环内终止策略使用的变量
        float anMinScore = min_score - 1;
        float NormMinScore = min_score / point_size;
        float NormGreediness = ((1 - greediness * min_score) / (1 - greediness)) / point_size; //计算贪婪数

        // 更新搜索区域(根据不同角度模板进行搜索)
        // 不同角度下模板特征的外包络框大小不一致
        // 在待测图像上，遍历搜索时，防止目标贴边压不上的可能
        int model_cx = -shape_info_vec->shape_angle[k]->bbx.lt_x;
        int model_cy = -shape_info_vec->shape_angle[k]->bbx.lt_y;
        search_region.start_X = model_cx + left - 1;
        search_region.start_Y = model_cy + top - 1;
        search_region.end_X = width - search_region.start_X + 1;
        search_region.end_Y = height - search_region.start_Y + 1;

        for (int i = search_region.start_X; i < search_region.end_X; i++) //搜索范围x
        {
            for (int j = search_region.start_Y; j < search_region.end_Y; j++) //搜索范围y
            {
                float PartialScore = 0;
                float PartialSum = 0; //初始化相似性度量分数
                int SumOfCoords = 0;

                for (int m = 0; m < point_size; m++) //某角度下的特征点数量
                {
                    int curX = 0;
                    int curY = 0;

                    float iTx = 0;
                    float iTy = 0;
                    float iSx = 0;
                    float iSy = 0;

                    curX = i + shape_angle->shape_point[m].x; //模板X坐标
                    curY = j + shape_angle->shape_point[m].y; //模板Y坐标

                    if (curX < 0 || curY < 0 || curX > width - 1 || curY > height - 1)
                    {
                        continue; //如果模板超出搜索图像边界范围，跳出继续，加速
                    }
                    iTx = shape_angle->shape_point[m].edge_dx; //模板X方向的梯度
                    iTy = shape_angle->shape_point[m].edge_dy; //模板Y方向的梯度

                    int offSet = curY * width + curX;
                    iSx = pBufGradX[offSet]; //从搜索图像中获取对应的X梯度
                    iSy = pBufGradY[offSet]; //从搜索图像中获取对应的Y梯度

                    //排除梯度为0的点
                    if ((iSx != 0.0f || iSy != 0.0f) && (iTx != 0.0f || iTy != 0.0f))
                    {
                        //===================================================================================
                        // 相似性度量公式
                        //===================================================================================
                        PartialSum += ((iSx * iTx) + (iSy * iTy)) /* * (iSm * iTm)*/; // 计算相似度
                    }
                    SumOfCoords = m + 1;
                    PartialScore = PartialSum / SumOfCoords; // 归一化

                    //===================================================================================
                    // 终止策略
                    //===================================================================================
                    if (PartialScore < (MIN(anMinScore + NormGreediness * SumOfCoords, NormMinScore * SumOfCoords)))
                    {
                        break;
                    }
                }

                if (PartialScore > min_score)
                {
                    locker.lock();
                    resultsPerDeg.push_back({T_T::Pose2d(i, j, shape_angle->angle), PartialScore});
                    locker.unlock();
                } // if 语句:大于最小得分值
#if DEBUG_COARSE_SHOW
                //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~绘制匹配过程~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                cv::Mat search_image_back;
                cv::cvtColor(search_image,search_image_back,cv::COLOR_GRAY2BGR);
                cv::drawMarker(search_image_back,cv::Point2f(i,j),cv::Scalar(0, 0, 255));
                cv::rectangle(search_image_back,cv::Point2f(search_region.start_X,search_region.start_Y),cv::Point2f(search_region.end_X,search_region.end_Y),cv::Scalar(255, 0, 0));
                for (const auto it:shape_angle->shape_point) {
                    search_image_back.at<cv::Vec3b>(it.y+j,it.x+i) = cv::Vec3b(0,255,0);;
                }
                cv::putText(search_image_back,std::to_string(PartialScore),cv::Point2f(i,j),cv::FONT_HERSHEY_DUPLEX,0.5,cv::Scalar(0, 0, 255));
                cv::imshow("COARSE",search_image_back);
                cv::waitKey(5);
                //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#endif
            } // 搜索区域j的for
        } // 搜索区域i的for

        //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        //~~~~~~~~~~~~~~< 对每一个角度下的粗匹配结果进行竞选 >~~~~~~~~~~~
        //~~~~~~~~~~~~~~< 原则:5*5区域内，粗匹配结果中取分数最大的 >~~~~~
        //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        resultsPerDegCandidates = _filterNearCandidates(resultsPerDeg);

        //对于每一个角度的模板，匹配结束，将结果保存至totalResultsTemp中
        for (const auto& ri : resultsPerDegCandidates)
        {
            locker.lock();
            totalResultsTemp.push_back(ri);
            locker.unlock();
        }
    } // 角度k的for循环 [OMP]

    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    //~~~~~~~~~~~~~~< 对所有角度下的粗匹配结果进行竞选 >~~~~~~~~~~~~
    //~~~~~~~~~~~~~~< 原则:5*5区域内，粗匹配结果中取分数最大的 >~~~~~
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    std::vector<T_T::MatchResult> totalResultsTemp1 = _filterNearCandidates(totalResultsTemp);

    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    //~~~~~~~~~~~~~~< 对所有角度下的粗匹配结果进行竞选 >~~~~~~~~~~~~
    //~~~~~~~~~~~~~~< 原则:重叠者，粗匹配结果中取分数最大的 >~~~~~
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    resultsfilter = _filterMaxOverLapCandidates(totalResultsTemp1, max_overlap, model_height, model_width);

    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    //~~~~~~~~~~~~~< 对粗匹配结果-分数从高到低排序竞选 >~~~~~~~~~~~~
    //~~~~~~~~~~~~~~< 原则:保留前1/3的分数较高的粗匹配结果 >~~~~~~~~
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#if 1
    double maxscore = 0;
    std::sort(
        resultsfilter.begin(),
        resultsfilter.end(),
        [](const T_T::MatchResult& result1, const T_T::MatchResult& result2)
        {
            return result1.score > result2.score;
        });
    if (!resultsfilter.empty()) { maxscore = resultsfilter[0].score; }
    // 根据分数比值，将小分数的情况排除
    // 挑选分数较大的一些匹配结果
    for (auto& rn : resultsfilter)
    {
        double proportion = 0;
        proportion = maxscore / rn.score;
        // |---------------|---->maxscore
        //          L1        L2
        // L2/(L1+L2) = 1/1.25 = 4/5
        if (proportion < 1.5) { resultList.push_back(rn); }
    }
#else
    std::sort(
        resultsfilter.begin(),
        resultsfilter.end(),
        [](const s::vision::SearchShapeMatchUtil::MatchResult &result1, const s::vision::SearchShapeMatchUtil::MatchResult &result2) {
            return result1.score > result2.score;
        });
    resultList = resultsfilter;
#endif
}


// 1.2 function:待测图像粗匹配到精匹配策略
bool SearchTemplate::_coarse2FineMatching(
    cv::Mat p_image_py,
    cv::Mat mask_image,
    T_T::Template::Ptr model_id,
    int py_levels,
    int width,
    int height,
    float min_score,
    float greediness,
    T_T::MatchResult* result_list_high,
    T_T::MatchResult* result_list_low,
    int Numpyl)
{
    T_T::SearchCfg SearchRegion{};

    double MatchPiontX = 0;
    double MatchPiontY = 0;
    double MatchAngle = 0;
    // 获取金字塔上一层匹配的目标的中心点
    MatchPiontX = result_list_high->pose.x;
    MatchPiontY = result_list_high->pose.y;
    MatchAngle = result_list_high->pose.angle;

    int cropImgW = 0;
    int cropImgH = 0;

    int Row1, Col1, Row2, Col2, ResultPiontX, ResultPiontY, ReferPointX, ReferPointY;

    T_T::ShapeInfo::Ptr pInfoPy = nullptr;

    //角度步长相关
    int OffSet = 1 << py_levels;
    // 取出相应金字塔层数的模板特征信息
    switch (py_levels)
    {
    case 0:
        pInfoPy = model_id->templates[0];
        break;
    case 1:
        pInfoPy = model_id->templates[1];
        break;
    case 2:
        pInfoPy = model_id->templates[2];
        break;
    case 3:
        pInfoPy = model_id->templates[3];
        break;
    case 4:
        pInfoPy = model_id->templates[4];
        break;
    case 5:
        pInfoPy = model_id->templates[5];
        break;
    case 6:
        pInfoPy = model_id->templates[6];
        break;
    case 7:
        pInfoPy = model_id->templates[7];
        break;
    default:
        break;
    } // END:获取模板制作的每层金字塔图像的模板特征信息model_id/templates

    // 在金字塔层图像中搜索模板
    int WidthPy = width >> py_levels;
    int HeightPy = height >> py_levels;

    // 金字塔高层图往低层搜索策略
    // 因此，此层的参考点应该是上一层匹配的中心点的2倍（金字塔采样比率为1/2）
    ResultPiontX = ((MatchPiontX * 2) < 0) ? 0 : (MatchPiontX * 2);
    ResultPiontY = ((MatchPiontY * 2) < 0) ? 0 : (MatchPiontY * 2);

    // 计算每层模板中心点的位置
    ReferPointX = (model_id->template_cfg.image_width >> py_levels);
    ReferPointY = (model_id->template_cfg.image_height >> py_levels);


    // 根据以上求解裁切框的左上、右下点坐标
    Row1 = ((ResultPiontX - ReferPointX - 2) < 0) ? 0 : (ResultPiontX - ReferPointX - 2);
    Col1 = ((ResultPiontY - ReferPointY - 2) < 0) ? 0 : (ResultPiontY - ReferPointY - 2);
    Row2 = ((ResultPiontX + ReferPointX + 2) > WidthPy) ? WidthPy : (ResultPiontX + ReferPointX + 2);
    Col2 = ((ResultPiontY + ReferPointY + 2) > HeightPy) ? HeightPy : (ResultPiontY + ReferPointY + 2);

    // 裁切框的大小
    cropImgW = abs(Row2 - Row1);
    cropImgH = abs(Col2 - Col1);
    if (Row1 > Row2) Row1 = p_image_py.cols / 2 - cropImgW / 2;
    if (Col1 > Col2) Col1 = p_image_py.rows / 2 - cropImgH / 2;

    if (Row1 < 0) Row1 = 0;
    if (Col1 < 0) Col1 = 0;

    cv::Mat cropImage;
    cv::Mat cropMask;

    // 对非金字塔高层的待测图像进行局部裁切
    // 裁切的目的：减少全图匹配的长耗时
    if ((Row1 >= p_image_py.cols) || (Col1 >= p_image_py.rows))
    {
        cropImage = p_image_py.clone();
        cropMask = mask_image.clone();
    }
    else
    {
        cv::Mat cropImageTemp(p_image_py, cv::Rect(Row1, Col1, cropImgW, cropImgH));
        cropImage = cropImageTemp.clone();
        cv::Mat cropMaskTemp(mask_image, cv::Rect(Row1, Col1, cropImgW, cropImgH));
        cropMask = cropMaskTemp.clone();
    }

    //===================================================================================
    //------------------------<  上层至下层的搜索范围以及搜索角度更新  >-----------------------
    //===================================================================================
    // 左上、右下搜索区域各偏置5个像素
    SearchRegion.start_X = ((ResultPiontX - Row1 - 5) < 0) ? 0 : (ResultPiontX - Row1 - 5);
    SearchRegion.start_Y = ((ResultPiontY - Col1 - 5) < 0) ? 0 : (ResultPiontY - Col1 - 5);
    SearchRegion.end_X = SearchRegion.start_X + 10;
    SearchRegion.end_Y = SearchRegion.start_Y + 10;
    // 搜索角度根据上层匹配角度逆时针、顺时针各偏移4度
    SearchRegion.start_angle = (MatchAngle - 4);
    SearchRegion.stop_angle = (MatchAngle + 4);

    if (pInfoPy == nullptr) return false;

    // 待测图像精匹配
    _fineMatching(cropImage, cropMask, pInfoPy, py_levels, cropImgW, cropImgH, min_score, greediness, SearchRegion,
                  result_list_low,false);
#if DEBUG_SHOW
    cv::Mat cropImageBGR;
    cv::cvtColor(cropImage, cropImageBGR, cv::COLOR_GRAY2BGR);
    //绘制匹配上的轮廓点
    cv::Vec3b contours_color(0, 255, 0);
    std::vector<T_T::ShapePoint> contours;
    for (const auto& it1 : pInfoPy->shape_angle)
    {
        if (std::abs(result_list_low->pose.angle - it1->angle) < 0.001)
        {
            contours = it1->shape_point;
            break;
        }
    }
    for (const auto& it2 : contours)
    {
        cropImageBGR.at<cv::Vec3b>(it2.y + result_list_low->pose.y, it2.x + result_list_low->pose.x) = contours_color;
    }

    cv::drawMarker(cropImageBGR, cv::Point2f(result_list_low->pose.x, result_list_low->pose.y), cv::Scalar(0, 0, 255),
                   cv::MARKER_CROSS);
    cv::putText(
        cropImageBGR, std::to_string(result_list_low->score),
        cv::Point2f(result_list_low->pose.x, result_list_low->pose.y), 1, 1,
        cv::Scalar(0, 0, 255));
    cv::putText(
        cropImageBGR, "Py" + std::to_string(Numpyl), cv::Point2f(cropImageBGR.cols / 2, 10), cv::FONT_HERSHEY_DUPLEX,
        0.5, cv::Scalar(0, 255, 0));
    cv::putText(
        cropImageBGR,
        "Deg" + std::to_string(result_list_low->pose.angle),
        cv::Point2f(result_list_low->pose.x, result_list_low->pose.y + 20),
        1,
        1,
        cv::Scalar(0, 0, 255));
    cv::imshow("精匹配", cropImageBGR);
    cv::waitKey(0);
#endif
    // 坐标变换：
    // 精匹配得到结果转换至原图上
    // [裁切图] --->  [原图]
    result_list_low->pose.x = result_list_low->pose.x + Row1;
    result_list_low->pose.y = result_list_low->pose.y + Col1;
    return true;
}

bool SearchTemplate::searchTemplate(cv::Mat image, cv::Mat s_mask_image,
                    T_T::Template::Ptr model_id,
                    int angle_start, int angle_extent,
                    float min_score, int num_matches,
                    float max_overlap, int num_levels,
                    float greediness, bool sort_by_y,
                    std::vector<T_T::MatchResult>& result_list)
{
    ROI roi; // 默认空
    return searchTemplate(image, s_mask_image, roi, model_id, angle_start, angle_extent,
                          min_score, num_matches, max_overlap, num_levels, greediness,
                          sort_by_y, result_list);
}

// 模板匹配程序入口程序
bool SearchTemplate::searchTemplate(
    cv::Mat image,
    cv::Mat s_mask_image,
    ROI roi,
    T_T::Template::Ptr model_id,
    int angle_start,
    int angle_extent,
    float min_score,
    int num_matches,
    float max_overlap,
    int num_levels,
    float greediness,
    bool sort_by_y,
    std::vector<T_T::MatchResult>& result_list)
{
#if COSTTIME_SHOW
    auto start_prepare = std::chrono::high_resolution_clock::now();
#endif
    if(!roi.empty())
        // 根据ROI裁出子图
        roi.crop(image, image);

    // 补充掩模图像，防止掩模图像为空
    if(s_mask_image.empty())
        s_mask_image = cv::Mat(image.rows, image.cols, CV_8UC1, cv::Scalar(255));

    ///开始模板匹配工作
    std::mutex locker;
    //多核并行多线程加速的核数
    thread_num_ = std::thread::hardware_concurrency();

    cv::Mat Image_c, smaskimage_c;
    cv::Mat Image = image.clone();
    cv::Mat smaskimage = s_mask_image.clone();
    if (Image.channels() == 3) { cv::cvtColor(Image, Image, cv::COLOR_BGR2GRAY); }
    if (smaskimage.channels() == 3) { cv::cvtColor(smaskimage, smaskimage, cv::COLOR_BGR2GRAY); }

    //解决find中金字塔设置问题
    if (num_levels == -1) { num_levels = model_id->template_cfg.num_levels; }
    //自定义金字塔层数必≤模板侧层数
    else if (num_levels > model_id->template_cfg.num_levels)
    {
        num_levels = model_id->template_cfg.num_levels;
    }
    //旋转角度处理
    if (angle_start < model_id->template_cfg.angle_start) { angle_start = model_id->template_cfg.angle_start; }
    if (angle_extent > model_id->template_cfg.angle_end) { angle_extent = model_id->template_cfg.angle_end; }

    if (num_levels >= 0)
    {
        // 待测图像图像扩展，为图像金字塔处理做准备
        cv::Mat ImgBordered = Image, MaskBordered = smaskimage;
        int top = 0, bottom = 0, left = 0, right = 0;

        // 待测图长、宽不为16的倍数：图像的长宽边进行扩展
        if ((Image.cols % 16 != 0) && (Image.rows % 16 != 0))
        {
            int BorderedWidth  = _convertLength(Image.cols);
            int BorderedHeight = _convertLength(Image.rows);
            int y2Offset = BorderedHeight - Image.rows;
            int x2Offset = BorderedWidth  - Image.cols;

            top    = (y2Offset + 1) / 2;
            bottom = y2Offset / 2;
            left   = (x2Offset + 1) / 2;
            right  = x2Offset / 2;

            cv::copyMakeBorder(Image, ImgBordered, top, bottom, left, right, cv::BORDER_REPLICATE);
            cv::copyMakeBorder(smaskimage, MaskBordered, top, bottom, left, right, cv::BORDER_REPLICATE);
        } // END:对输入的待测图像、掩模图像的padding操作，按长短边的2^n(n>=4)来扩
        else
        {
            ImgBordered = Image;
            MaskBordered = smaskimage;
        }
        // 待测图像金字塔处理: 金字塔每层的待测图像、掩模图像进行下采样（1/2）
        std::vector<cv::Mat> imagePyr(num_levels + 1), maskPyr(num_levels + 1);
        imagePyr[0] = ImgBordered;
        maskPyr[0]  = MaskBordered;

        for (int i = 1; i <= num_levels; i++) {
            cv::pyrDown(imagePyr[i-1], imagePyr[i]);
            cv::pyrDown(maskPyr[i-1],  maskPyr[i]);
        }

        //++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        //+++++++++++++++++++++++++待测图像金字塔最高层图像的粗匹配作用域+++++++++++++++++++++++++
        //++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        std::vector<T_T::MatchResult> ResultListPyRude;

        T_T::ShapeInfo::Ptr pInfoPy = model_id->templates[num_levels];
        cv::Mat pImage = imagePyr[num_levels];
        cv::Mat pMask  = maskPyr[num_levels];

        int WidthPy = ImgBordered.cols >> num_levels;
        int HeightPy = ImgBordered.rows >> num_levels;
        int Left = left >> num_levels;
        int Top = top >> num_levels;

        //搜索区域(SearchRegion)赋值:模板原点在模板中心，将模板贴着待搜索图像滑窗，并忽略掉padding的边界
        T_T::SearchCfg SearchRegion{};
        SearchRegion.start_angle = angle_start;
        SearchRegion.stop_angle = angle_extent;
        start_angle_ = model_id->template_cfg.angle_start;
        stop_angle_ = model_id->template_cfg.angle_end;

        if (pInfoPy == nullptr) { return false; }

        int modelwidth = model_id->template_cfg.image_width >> num_levels;
        int modelheight = model_id->template_cfg.image_height >> num_levels;

        // 待测图像粗匹配：特征提取和相似性度量
        max_contrast_ = model_id->template_cfg.max_contrast;
#if COSTTIME_SHOW
        auto                                      end_prepare      = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration_prepare = end_prepare - start_prepare;
        std::cout << "前处理耗时: " << duration_prepare.count() << " ms." << std::endl;
#endif

#if COSTTIME_SHOW
        auto start_coarse = std::chrono::high_resolution_clock::now();
#endif
        // 金字塔最高层粗匹配（全图搜索）
        _coarseMatching(
            pImage,
            pMask,
            pInfoPy,
            WidthPy,
            HeightPy,
            modelwidth,
            modelheight,
            Left,
            Top,
            min_score,
            greediness,
            max_overlap,
            SearchRegion,
            ResultListPyRude);
#if COSTTIME_SHOW
        auto                                      end_coarse      = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration_coarse = end_coarse - start_coarse;
        std::cout << "粗匹配耗时: " << duration_coarse.count() << " ms." << std::endl;
#endif

#if DEBUG_SHOW
        printf("粗匹配结果数量: %ld\n", ResultListPyRude.size());
        cv::Mat pImageBGR;
        cv::cvtColor(pImage, pImageBGR, cv::COLOR_GRAY2BGR);
        cv::putText(
            pImageBGR,
            "Pixel[" + std::to_string(pImage.rows) + "*" + std::to_string(pImage.cols) + "]",
            cv::Point2f(10, 10),
            cv::FONT_HERSHEY_DUPLEX,
            0.5,
            cv::Scalar(0, 255, 0));
        cv::Vec3b contours_color(0, 255, 0);
        for (const auto& it : ResultListPyRude)
        {
            //绘制中心点位置以及分数
            cv::drawMarker(pImageBGR, cv::Point2f(it.pose.x, it.pose.y), cv::Scalar(255, 0, 0), cv::MARKER_CROSS);
            cv::putText(
                pImageBGR, std::to_string(it.score), cv::Point2f(it.pose.x, it.pose.y), cv::FONT_HERSHEY_DUPLEX, 0.25,
                cv::Scalar(0, 0, 255));
            //绘制匹配上的轮廓点
            std::vector<T_T::ShapePoint> contours;
            for (const auto& it1 : pInfoPy->shape_angle)
            {
                if (std::abs(it.pose.angle - it1->angle) < 0.001)
                {
                    contours = it1->shape_point;
                    break;
                }
            }
            for (const auto& it2 : contours)
            {
                pImageBGR.at<cv::Vec3b>(it2.y + it.pose.y, it2.x + it.pose.x) = contours_color;
            }
        }
        cv::putText(
            pImageBGR, "Py" + std::to_string(Numpyl), cv::Point2f(pImageBGR.cols / 2, 10), cv::FONT_HERSHEY_DUPLEX, 0.5,
            cv::Scalar(0, 255, 0));
        cv::putText(
            pImageBGR,
            std::to_string(ResultListPyRude.size()) + "pcs",
            cv::Point2f(pImageBGR.cols / 2, pImageBGR.rows - 5),
            cv::FONT_HERSHEY_DUPLEX,
            0.5,
            cv::Scalar(0, 0, 255));
        cv::imshow("粗匹配结果", pImageBGR);
        cv::waitKey(0);
#endif
        //++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        //+++++++++++++++++++++++++待测图像非金字塔最高层精匹配作用域+++++++++++++++++++++++++++++
        //++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        std::vector<T_T::MatchResult> findResult; // vector:用来存储待测图像目标点排序筛选剔除后的结果
        std::vector<T_T::MatchResult> TempResult; // vector:中间变量
#if COSTTIME_SHOW
        auto start_fine = std::chrono::high_resolution_clock::now();
#endif
#if !DEBUG_SHOW
#pragma omp parallel for num_threads(thread_num_)
#endif
        for (int ri = 0; ri < ResultListPyRude.size(); ri++) //获取金字塔最高层所有的粗匹配结果:ResultListPyRude
        {
            T_T::MatchResult ResultListHigh = ResultListPyRude[ri]; //最高层粗匹配下的逐个结果
            T_T::MatchResult ResultListLow = ResultListHigh;

            // 从 num_levels-1 层往下逐层精匹配
            for (int N = num_levels - 1; N >= 0; N--) {
                cv::Mat pImage = imagePyr[N];
                cv::Mat pMask  = maskPyr[N];

                // 待测图像粗匹配到精匹配策略（每次传入ResultListHigh，并获取ResultListLow结果）
                _coarse2FineMatching(
                    pImage,
                    pMask,
                    model_id,
                    N,
                    ImgBordered.cols,
                    ImgBordered.rows,
                    min_score,
                    greediness,
                    &ResultListHigh,
                    &ResultListLow,
                    N);

                ResultListHigh = ResultListLow;

                if (ResultListLow.score < static_cast<double>(min_score))
                {
                    break; //高层至底层匹配的过程中，分数低于设定值则停止对这个粗匹配的向下寻找真理（直到第0层）
                }
            } // END:非金字塔最高层的每一层金字塔精匹配

            if (ResultListLow.score > static_cast<double>(min_score))
            {
                //由padding图上的结果还原至原图上的结果(左扩边界left,上扩边界top)
                T_T::MatchResult Po{
                    T_T::Pose2d(ResultListLow.pose.x - left, ResultListLow.pose.y - top, ResultListLow.pose.angle),
                    ResultListLow.score
                };
                if (Po.pose.x >= 0 && Po.pose.y >= 0 && Po.pose.x <= Image.cols && Po.pose.y <= Image.rows)
                    if (Po.pose.x >= 0 && Po.pose.y >= 0 && Po.pose.x <= Image.cols && Po.pose.y <= Image.rows)
                    {
                        locker.lock();
                        TempResult.push_back(Po);
                        locker.unlock();
                    }
            }
        } // END:遍历完最高层金字塔匹配出的所有可能结果 [OMP]
#if COSTTIME_SHOW
        auto                                      end_fine      = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration_fine = end_fine - start_fine;
        std::cout << "精匹配耗时: " << duration_fine.count() << " ms." << std::endl;
#endif

#if COSTTIME_SHOW
        auto start_final = std::chrono::high_resolution_clock::now();
#endif
        //++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        //+++++++++++++++++++++所有精匹配结果，按照重叠率筛选[重则选分数较大者]++++++++++++++++++++
        //++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        findResult = _filterMaxOverLapCandidates(TempResult, max_overlap, model_id->template_cfg.image_height,
                                                 model_id->template_cfg.image_width);

        //++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        //+++++++++++++++++++++++++待测图像目标点排序筛选剔除+++++++++++++++++++++++++++++
        //++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        // 待测图像目标点排序筛选剔除
        std::sort(
            findResult.begin(),
            findResult.end(),
            [](const T_T::MatchResult& result1, const T_T::MatchResult& result2)
            {
                return result1.score > result2.score;
            });
        //设置的匹配数量 大于 固有的匹配数量
        if (num_matches >= findResult.size() || (num_matches == -1))
        {
            for (auto& fn : findResult)
            {
                if (fn.score != 0.0)
                {
                    fn.pose.angle = -fn.pose.angle; //逆向为角度正

                    /// ROI裁剪子图到全图坐标转换
                    if(!roi.empty())
                    {
                        double xinimg,yinimg,angleinimg;
                        bool ret = roi.toImageCoord(fn.pose.x, fn.pose.y, fn.pose.angle, xinimg,yinimg,angleinimg);
                        if (ret){
                            fn.pose.x = xinimg;
                            fn.pose.y = yinimg;
                            fn.pose.angle = angleinimg;
                        }
                    }

                    result_list.push_back(fn);
                }
            }
        }
        else //设置的匹配数量 小于 固有的匹配数量
        {
            for (int n = 0; n < num_matches; n++)
            {
                if (findResult[n].score != 0.0)
                {
                    findResult[n].pose.angle = -findResult[n].pose.angle; //逆向为角度正

                    /// ROI裁剪子图到全图坐标转换
                    if(!roi.empty())
                    {
                        double xinimg,yinimg,angleinimg;
                        bool ret = roi.toImageCoord(findResult[n].pose.x, findResult[n].pose.y, findResult[n].pose.angle, xinimg,yinimg,angleinimg);
                        if (ret){
                            findResult[n].pose.x = xinimg;
                            findResult[n].pose.y = yinimg;
                            findResult[n].pose.angle = angleinimg;
                        }
                    }

                    result_list.push_back(findResult[n]);
                }
            }
        }

        // 按x、y坐标排序
        if (sort_by_y)
            std::sort(result_list.begin(), result_list.end(), [](const T_T::MatchResult& result1, const T_T::MatchResult& result2) {return result1.pose.y < result2.pose.y;});//y坐标升序
        else
            std::sort(result_list.begin(), result_list.end(), [](const T_T::MatchResult& result1, const T_T::MatchResult& result2) {return result1.pose.x < result2.pose.x;});//x坐标升序

#if COSTTIME_SHOW
        auto                                      end_final      = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration_final = end_final - start_final;
        std::cout << "后处理耗时: " << duration_final.count() << " ms." << std::endl;
#endif

        return true;
    } // scope
    else
    {
        return false;
    }
}

//初始化各层金字塔的模板信息
void initialShapeModelPyd(T_T::ShapeInfo::Ptr shape_info_vec, int angle_start, int angle_stop, double angle_step)
{
    //初始化 Vector:shape_angle，内含智能指针
    int angle_num = 0;
    for (double iAngle = angle_start; iAngle < angle_stop; iAngle += angle_step)
    {
        angle_num++;
    }
    for (int i = 0; i < angle_num + 2; i++)
    {
        shape_info_vec->shape_angle.push_back(std::make_shared<T_T::ShapeAngle>());
    }

    int angleNum = 0;
    //如果起始角度与终止角度相同（-180~-180）
    if (angle_start == angle_stop)
    {
        angleNum = 2; //角度变化只有1个，加上模板为0角度，则是2个
        shape_info_vec->shape_angle[0]->angle = 0;
        shape_info_vec->shape_angle[1]->angle = angle_start;
    }
    //如果起始角度与终止角度不同（-180~180）
    else
    {
        shape_info_vec->shape_angle[0]->angle = 0;
        for (double iAngle = angle_start; iAngle < angle_stop; iAngle += angle_step)
        {
            shape_info_vec->shape_angle[angleNum + 1]->angle = iAngle; //[]内为1-360
            angleNum++;
        }
        shape_info_vec->shape_angle[angleNum + 1]->angle = angle_stop;
    }
}

// 初始化模板资源
void initialShapeModel(T_T::Template::Ptr model_id)
{
    int angleStart = model_id->template_cfg.angle_start;
    double angleStep = model_id->template_cfg.angle_step;
    int angleStop = model_id->template_cfg.angle_end;

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
            initialShapeModelPyd(model_id->templates[0], angleStart, angleStop, angleStep);
            break;
        case 1:
            //初始化
            initialShapeModelPyd(model_id->templates[1], angleStart, angleStop, angleStep);
            break;
        case 2:
            //初始化
            initialShapeModelPyd(model_id->templates[2], angleStart, angleStop, angleStep);
            break;
        case 3:
            //初始化
            initialShapeModelPyd(model_id->templates[3], angleStart, angleStop, angleStep);
            break;
        case 4:
            //初始化
            initialShapeModelPyd(model_id->templates[4], angleStart, angleStop, angleStep);
            break;
        case 5:
            //初始化
            initialShapeModelPyd(model_id->templates[5], angleStart, angleStop, angleStep);
            break;
        case 6:
            //初始化
            initialShapeModelPyd(model_id->templates[6], angleStart, angleStop, angleStep);
            break;
        case 7:
            //初始化
            initialShapeModelPyd(model_id->templates[7], angleStart, angleStop, angleStep);
            break;
        default:
            break;
        }
    }
}

T_T::Template::Ptr SearchTemplate::loadModelFileFromJson(std::string path)
{
    T_T::Template::Ptr temp = std::make_shared<T_T::Template>();

    // 读取模板文件
    cv::FileStorage fs(path, cv::FileStorage::READ);
    if (!fs.isOpened())
    {
        std::cout << "load model failed!" << std::endl;
        return nullptr;
    }

    // 读取模板配置
    cv::FileNode fn = fs.root();
    cv::FileNode fn_shapeMatchPre = fn["shapeMatch"];

    temp->template_cfg.angle_start = fn_shapeMatchPre["angle_start"];
    temp->template_cfg.angle_end = fn_shapeMatchPre["angle_end"];
    temp->template_cfg.angle_step = fn_shapeMatchPre["angle_step"];
    int temp_otsu = fn_shapeMatchPre["auto_threshold"];
    if (temp_otsu == 0)
        temp->template_cfg.create_otsu = false;
    else if (temp_otsu == 1)
        temp->template_cfg.create_otsu = true;
    temp->template_cfg.min_contrast = fn_shapeMatchPre["min_constract"];
    temp->template_cfg.max_contrast = fn_shapeMatchPre["max_constract"];
    temp->template_cfg.num_levels = fn_shapeMatchPre["num_levels"];
    temp->template_cfg.id = fn_shapeMatchPre["id"];
    // temp->template_cfg.is_inited = fn_shapeMatchPre["is_inited"];
    image_width_ = temp->template_cfg.image_width = fn_shapeMatchPre["image_width"];
    image_height_ = temp->template_cfg.image_height = fn_shapeMatchPre["image_height"];

    // 读取模板的特征数据
    cv::FileNode tps_fn = fn_shapeMatchPre["templates"];
    cv::FileNodeIterator tps_it = tps_fn.begin(), tps_it_end = tps_fn.end();

    for (; tps_it != tps_it_end; ++tps_it)
    {
        // 初始化模板信息
        T_T::ShapeInfo::Ptr temp_shapeinfo = std::make_shared<T_T::ShapeInfo>();

        cv::FileNode pyrds_fn = (*tps_it)["template_pyramid"];
        cv::FileNodeIterator pyrd_it = pyrds_fn.begin(), pyrd_it_end = pyrds_fn.end();

        for (; pyrd_it != pyrd_it_end; ++pyrd_it) //遍历模板金字塔层
        {
            T_T::ShapeAngle::Ptr temp_angle = std::make_shared<T_T::ShapeAngle>();
            temp_angle->angle = (*pyrd_it)["angle"];

            cv::FileNode ShapeAngle_fn = (*pyrd_it)["features"];
            cv::FileNodeIterator features_it = ShapeAngle_fn.begin();
            cv::FileNodeIterator features_it_end = ShapeAngle_fn.end();
            for (; features_it != features_it_end; ++features_it) //遍历features（-180~180）
            {
                cv::FileNodeIterator feature_info = (*features_it).begin();
                T_T::ShapePoint shape_point;
                double x;
                double y;
                float edge_dx;
                float edge_dy;
                //                float                                 edge_mag;
                feature_info >> x >> y >> edge_dx >> edge_dy /*>> edge_mag*/;

                shape_point.x = x;
                shape_point.y = y;
                shape_point.edge_dx = edge_dx;
                shape_point.edge_dy = edge_dy;
                //                shape_point.edge_mag = edge_mag;
                temp_angle->shape_point.push_back(shape_point);
            }
            temp_shapeinfo->shape_angle.push_back(temp_angle);
        }
        temp->templates.push_back(temp_shapeinfo);
    }

    fs.release(); // 关闭文件流
    std::cout << "加载模板数据成功[Json]!" << std::endl;

    // ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓ 补全模板数据（旋转特征点）↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
    if (temp != nullptr)
    {
        T_T::Template::Ptr model_id_cp = std::make_shared<T_T::Template>();
        model_id_cp->template_cfg = temp->template_cfg;

        // 初始化模型
        initialShapeModel(model_id_cp);
        for (int index = 0; index < model_id_cp->template_cfg.num_levels + 1; index++)
        {
            model_id_cp->templates[index]->shape_angle[0] = temp->templates[index]->shape_angle[0];
        }

        // 遍历金字塔层数
        for (int index = 0; index < temp->template_cfg.num_levels + 1; index++)
        {
            // 遍历用的角度数量
            int angle_num = model_id_cp->templates[index]->shape_angle.size();
            // 遍历用的点数量
            int shape_size = model_id_cp->templates[index]->shape_angle[0]->shape_point.size();

            // 非0角度特征 - shape_point初始化
            for (int i = 1; i < angle_num; i++)
            {
                model_id_cp->templates[index]->shape_angle[i]->shape_point.resize(shape_size);
            }
#pragma omp parallel for num_threads(thread_num_)
            //对0角度下的模板特征进行旋转、赋值
            for (int i = 1; i < angle_num; i++) //角度个数
            {
                int xOffSet = (model_id_cp->template_cfg.image_width >> index) / 2;
                int yOffSet = (model_id_cp->template_cfg.image_height >> index) / 2;
                double angle = -model_id_cp->templates[index]->shape_angle[i]->angle;
                float rad = (double)((angle * CV_PI) / 180); // 180/π =angle/rad

                for (int j = 0; j < shape_size; j++) //轮廓点数量
                {
                    //坐标x,y变化
                    int rOrigX, rOrigY;
                    float X, Y, T;
                    //通过坐标变化，将坐标原点0在左上角的图像坐标系转换为笛卡尔坐标系（原点在图像中心，x朝右，y朝上）
                    X = model_id_cp->templates[index]->shape_angle[0]->shape_point[j].x;
                    Y = -model_id_cp->templates[index]->shape_angle[0]->shape_point[j].y;
                    T = X;
                    X = X * std::cos(rad) - Y * std::sin(rad); // 逆时针旋转
                    Y = T * std::sin(rad) + Y * std::cos(rad); // 逆时针旋转

                    rOrigX = (X + xOffSet > 0.0) ? (X + xOffSet + 0.5) : (X + xOffSet - 0.5); //四舍五入取整数
                    rOrigY = (yOffSet - Y > 0.0) ? (yOffSet - Y + 0.5) : (yOffSet - Y - 0.5); //四舍五入取整数

                    float DX, DY, DT;
                    // dx,dy变换
                    DX = model_id_cp->templates[index]->shape_angle[0]->shape_point[j].edge_dx;
                    DY = -model_id_cp->templates[index]->shape_angle[0]->shape_point[j].edge_dy;
                    DT = DX;
                    DX = DX * std::cos(rad) - DY * std::sin(rad); // 逆时针旋转
                    DY = DT * std::sin(rad) + DY * std::cos(rad); // 逆时针旋转

                    // 更新旋转后的坐标x,y，以及梯度dx,dy
                    model_id_cp->templates[index]->shape_angle[i]->shape_point[j].x = rOrigX - xOffSet;
                    model_id_cp->templates[index]->shape_angle[i]->shape_point[j].y = rOrigY - yOffSet;

                    model_id_cp->templates[index]->shape_angle[i]->shape_point[j].edge_dx = DX;
                    model_id_cp->templates[index]->shape_angle[i]->shape_point[j].edge_dy = -DY;
                }
            }
        }

        // 将更新后的模型数据返回
        temp.reset();
        temp = model_id_cp;
        model_id_cp.reset();
        std::cout << "补全模板数据成功[Json]!" << std::endl;
    }
    // ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑
    return temp;
}

T_T::Template::Ptr SearchTemplate::loadModelFileFromBinary(std::string path)
{
    // 创建模板智能指针
    T_T::Template::Ptr temp = std::make_shared<T_T::Template>();
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs.is_open())
    {
        std::cerr << "无法打开文件 " << path << " 来加载数据!" << std::endl;
        return nullptr;
    }

    // 读取模板的配置信息
    ifs.read(reinterpret_cast<char*>(&temp->template_cfg.angle_start), sizeof(temp->template_cfg.angle_start));
    ifs.read(reinterpret_cast<char*>(&temp->template_cfg.angle_end), sizeof(temp->template_cfg.angle_end));
    ifs.read(reinterpret_cast<char*>(&temp->template_cfg.angle_step), sizeof(temp->template_cfg.angle_step));
    ifs.read(reinterpret_cast<char*>(&temp->template_cfg.create_otsu), sizeof(temp->template_cfg.create_otsu));
    ifs.read(reinterpret_cast<char*>(&temp->template_cfg.min_contrast), sizeof(temp->template_cfg.min_contrast));
    ifs.read(reinterpret_cast<char*>(&temp->template_cfg.max_contrast), sizeof(temp->template_cfg.max_contrast));
    ifs.read(reinterpret_cast<char*>(&temp->template_cfg.num_levels), sizeof(temp->template_cfg.num_levels));
    ifs.read(reinterpret_cast<char*>(&temp->template_cfg.id), sizeof(temp->template_cfg.id));
    ifs.read(reinterpret_cast<char*>(&temp->template_cfg.image_width), sizeof(temp->template_cfg.image_width));
    ifs.read(reinterpret_cast<char*>(&temp->template_cfg.image_height), sizeof(temp->template_cfg.image_height));
    ifs.read(reinterpret_cast<char*>(&temp->template_cfg.is_inited), sizeof(temp->template_cfg.is_inited));

    image_height_ = temp->template_cfg.image_height;
    image_width_ = temp->template_cfg.image_width;

    // 读取金字塔层数
    int num_pyramids;
    ifs.read(reinterpret_cast<char*>(&num_pyramids), sizeof(num_pyramids));

    // 读取每层模板数据
    for (int i = 0; i < num_pyramids; ++i)
    {
        T_T::ShapeInfo::Ptr temp_shapeinfo = std::make_shared<T_T::ShapeInfo>();

        int num_angles;
        ifs.read(reinterpret_cast<char*>(&num_angles), sizeof(num_angles));

        // 读取每个角度下的特征点数据
        for (int j = 0; j < num_angles; ++j)
        {
            T_T::ShapeAngle::Ptr temp_angle = std::make_shared<T_T::ShapeAngle>();

            ifs.read(reinterpret_cast<char*>(&temp_angle->angle), sizeof(temp_angle->angle));

            int num_points;
            ifs.read(reinterpret_cast<char*>(&num_points), sizeof(num_points));

            // 读取每个特征点的坐标和梯度
            for (int k = 0; k < num_points; ++k)
            {
                T_T::ShapePoint shape_point;

                ifs.read(reinterpret_cast<char*>(&shape_point.x), sizeof(shape_point.x));
                ifs.read(reinterpret_cast<char*>(&shape_point.y), sizeof(shape_point.y));
                ifs.read(reinterpret_cast<char*>(&shape_point.edge_dx), sizeof(shape_point.edge_dx));
                ifs.read(reinterpret_cast<char*>(&shape_point.edge_dy), sizeof(shape_point.edge_dy));

                temp_angle->shape_point.push_back(shape_point);
            }

            temp_shapeinfo->shape_angle.push_back(temp_angle);
        }

        temp->templates.push_back(temp_shapeinfo);
    }

    ifs.close(); // 关闭文件流
    std::cout << "加载模板数据成功[Binary]!" << std::endl;

    // ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓ 补全模板数据（旋转特征点）↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
    if (temp != nullptr)
    {
        T_T::Template::Ptr model_id_cp = std::make_shared<T_T::Template>();
        model_id_cp->template_cfg = temp->template_cfg;

        // 初始化模型
        initialShapeModel(model_id_cp);

        for (int index = 0; index < model_id_cp->template_cfg.num_levels + 1; index++)
        {
            model_id_cp->templates[index]->shape_angle[0] = temp->templates[index]->shape_angle[0];
        }

        // 遍历金字塔层数
        for (int index = 0; index < temp->template_cfg.num_levels + 1; index++)
        {
            // 遍历用的角度数量
            int angle_num = model_id_cp->templates[index]->shape_angle.size();
            // 遍历用的点数量
            int shape_size = model_id_cp->templates[index]->shape_angle[0]->shape_point.size();

            // 非0角度特征 - shape_point初始化
            for (int i = 1; i < angle_num; i++)
            {
                model_id_cp->templates[index]->shape_angle[i]->shape_point.resize(shape_size);
            }

#pragma omp parallel for num_threads(thread_num_)
            //对0角度下的模板特征进行旋转、赋值
            for (int i = 1; i < angle_num; i++) //角度个数
            {
                int xOffSet = (model_id_cp->template_cfg.image_width >> index) / 2;
                int yOffSet = (model_id_cp->template_cfg.image_height >> index) / 2;
                double angle = -model_id_cp->templates[index]->shape_angle[i]->angle;
                float rad = (double)((angle * CV_PI) / 180); // 180/π = angle/rad

                for (int j = 0; j < shape_size; j++) //轮廓点数量
                {
                    //坐标x,y变化
                    int rOrigX, rOrigY;
                    float X, Y, T;
                    //通过坐标变化，将坐标原点0在左上角的图像坐标系转换为笛卡尔坐标系（原点在图像中心，x朝右，y朝上）
                    X = model_id_cp->templates[index]->shape_angle[0]->shape_point[j].x;
                    Y = -model_id_cp->templates[index]->shape_angle[0]->shape_point[j].y;
                    T = X;
                    X = X * std::cos(rad) - Y * std::sin(rad); // 逆时针旋转
                    Y = T * std::sin(rad) + Y * std::cos(rad); // 逆时针旋转

                    rOrigX = (X + xOffSet > 0.0) ? (X + xOffSet + 0.5) : (X + xOffSet - 0.5); //四舍五入取整数
                    rOrigY = (yOffSet - Y > 0.0) ? (yOffSet - Y + 0.5) : (yOffSet - Y - 0.5); //四舍五入取整数

                    float DX, DY, DT;
                    // dx,dy变换
                    DX = model_id_cp->templates[index]->shape_angle[0]->shape_point[j].edge_dx;
                    DY = -model_id_cp->templates[index]->shape_angle[0]->shape_point[j].edge_dy;
                    DT = DX;
                    DX = DX * std::cos(rad) - DY * std::sin(rad); // 逆时针旋转
                    DY = DT * std::sin(rad) + DY * std::cos(rad); // 逆时针旋转

                    // 更新旋转后的坐标x,y，以及梯度dx,dy
                    model_id_cp->templates[index]->shape_angle[i]->shape_point[j].x = rOrigX - xOffSet;
                    model_id_cp->templates[index]->shape_angle[i]->shape_point[j].y = rOrigY - yOffSet;

                    model_id_cp->templates[index]->shape_angle[i]->shape_point[j].edge_dx = DX;
                    model_id_cp->templates[index]->shape_angle[i]->shape_point[j].edge_dy = -DY;
                }
            }
        }

        // 将更新后的模型数据返回
        temp.reset();
        temp = model_id_cp;
        model_id_cp.reset();
        std::cout << "补全模板数据成功[Binary]!" << std::endl;
    }

    // ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑
    return temp;
}


// Helper function to check if a point is within image bounds
bool isPointInBounds(const cv::Point2f& pt, int cols, int rows) {
    return pt.x >= 0 && pt.x < cols && pt.y >= 0 && pt.y < rows;
}
// Helper function to clip a point to image boundaries
cv::Point2f clipPoint(const cv::Point2f& pt, int cols, int rows) {
    return cv::Point2f(
        std::max(0.0f, std::min(static_cast<float>(cols - 1), pt.x)),
        std::max(0.0f, std::min(static_cast<float>(rows - 1), pt.y))
    );
}


void SearchTemplate::drawMatchResults(cv::Mat& image, const std::vector<T_T::MatchResult>& results,
                                      T_T::ShapeInfo::Ptr shapeInfo)
{
    std::vector<T_T::ShapePoint> shapePoints;
    for (int i = 0; i < results.size(); i++)
    {
        // 颜色生成
        int r, g, b;
        _hsvToRgb(&r, &g, &b, 360.0 / results.size() * i, 100, 100);
        if (results[i].score == 0) continue;

        std::vector<T_T::ShapePoint> contours;
        for (const auto& shapeAngle : shapeInfo->shape_angle)
        {
            double angleDiff = fabs(shapeAngle->angle - (-results[i].pose.angle));
            if (angleDiff < 0.001)
            {
                contours = shapeAngle->shape_point;
                break;
            }
        }

        // 绘制轮廓点
        cv::Point center(results[i].pose.x, results[i].pose.y);
        for (const auto& contour : contours)
        {
            T_T::ShapePoint pt;
            pt.x = contour.x + center.x;
            pt.y = contour.y + center.y;
            if (pt.x >= 0 && pt.x < image.cols && pt.y >= 0 && pt.y < image.rows)
            {
                image.at<cv::Vec3b>(pt.y, pt.x) = cv::Vec3b(0, 255, 0);
            }
        }

        // // 绘制中心点
        // cv::drawMarker(image, center, cv::Scalar(0, 0, 255), cv::MARKER_TILTED_CROSS, 20, 1, cv::LINE_AA);
        //
        // // 绘制模板框
        // cv::RotatedRect rotatedRect = cv::RotatedRect(
        //     cv::Point2f(results[i].pose.x, results[i].pose.y),
        //     cv::Size2f(image_width_, image_height_), // 使用模板实际尺寸
        //     -results[i].pose.angle // 角度取反以匹配坐标系
        // );
        // cv::Point2f vertices[4];
        // rotatedRect.points(vertices);
        // for (int j = 0; j < 4; j++)
        // {
        //     cv::line(image, vertices[j], vertices[(j + 1) % 4], cv::Scalar(r, g, b), 2, cv::LINE_AA);
        // }
        //
        // // 绘制模板中心指向右侧的矢量
        // cv::Point2f start_point(results[i].pose.x, results[i].pose.y);
        // cv::Point2f end_point;
        // double rad = -results[i].pose.angle * CV_PI / 180.0; // 角度取反并转换为弧度
        // float arrow_length = image_width_ / 2.0f;
        // end_point.x = start_point.x + arrow_length * cos(rad);
        // end_point.y = start_point.y + arrow_length * sin(rad);
        // cv::arrowedLine(image, start_point, end_point, cv::Scalar(r, g, b), 2, cv::LINE_AA, 0, 0.2);

        // 绘制中心点
        if (isPointInBounds(center, image.cols, image.rows)) {
            cv::drawMarker(image, clipPoint(center, image.cols, image.rows),
                           cv::Scalar(0, 0, 255), cv::MARKER_TILTED_CROSS, 20, 1, cv::LINE_AA);
        }

        // 绘制模板框
        cv::RotatedRect rotatedRect = cv::RotatedRect(
            cv::Point2f(results[i].pose.x, results[i].pose.y),
            cv::Size2f(image_width_, image_height_),
            -results[i].pose.angle
        );
        cv::Point2f vertices[4];
        rotatedRect.points(vertices);
        for (int j = 0; j < 4; j++) {
            cv::Point2f p1_f = vertices[j];
            cv::Point2f p2_f = vertices[(j + 1) % 4];
            // Convert to cv::Point for clipLine
            cv::Point p1 = cv::Point(static_cast<int>(p1_f.x), static_cast<int>(p1_f.y));
            cv::Point p2 = cv::Point(static_cast<int>(p2_f.x), static_cast<int>(p2_f.y));
            if (cv::clipLine(cv::Rect(0, 0, image.cols, image.rows), p1, p2)) {
                // Convert back to cv::Point2f for drawing
                p1_f = cv::Point2f(static_cast<float>(p1.x), static_cast<float>(p1.y));
                p2_f = cv::Point2f(static_cast<float>(p2.x), static_cast<float>(p2.y));
                cv::line(image, p1_f, p2_f, cv::Scalar(r, g, b), 2, cv::LINE_AA);
            }
        }

        // 绘制模板中心指向右侧的矢量
        cv::Point2f start_point(results[i].pose.x, results[i].pose.y);
        cv::Point2f end_point;
        double rad = -results[i].pose.angle * CV_PI / 180.0;
        float arrow_length = image_width_ / 2.0f;
        end_point.x = start_point.x + arrow_length * cos(rad);
        end_point.y = start_point.y + arrow_length * sin(rad);
        // Convert to cv::Point for clipLine
        cv::Point start = cv::Point(static_cast<int>(start_point.x), static_cast<int>(start_point.y));
        cv::Point end = cv::Point(static_cast<int>(end_point.x), static_cast<int>(end_point.y));
        if (cv::clipLine(cv::Rect(0, 0, image.cols, image.rows), start, end)) {
            // Convert back to cv::Point2f for drawing
            start_point = cv::Point2f(static_cast<float>(start.x), static_cast<float>(start.y));
            end_point = cv::Point2f(static_cast<float>(end.x), static_cast<float>(end.y));
            cv::arrowedLine(image, start_point, end_point, cv::Scalar(r, g, b), 2, cv::LINE_AA, 0, 0.2);
        }
    }
/*
    // 绘制点与点之间的距离与连线
    for (size_t i = 0; i < results.size() - 1; ++i)
    {
        // 颜色生成
        int r, g, b;
        _hsvToRgb(&r, &g, &b, 360.0 / results.size() * i, 100, 100);
        if (results[i].score == 0) continue;

        const T_T::MatchResult& p1 = results[i];
        const T_T::MatchResult& p2 = results[i + 1];
        // 计算欧氏距离
        double distance = std::sqrt((p2.pose.x - p1.pose.x)*(p2.pose.x-p1.pose.x) +(p2.pose.y - p1.pose.y)*(p2.pose.y-p1.pose.y) );
        // 计算中点
        cv::Point2f midpoint((p1.pose.x + p2.pose.x) / 2, (p1.pose.y + p2.pose.y) / 2);
        // 绘制连线
        cv::line(image, cv::Point2f(p1.pose.x, p1.pose.y), cv::Point2f(p2.pose.x, p2.pose.y), cv::Scalar(r, g, b), 1, cv::LINE_AA);
        // 绘制距离
        cv::putText(image, std::to_string(distance), midpoint, cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(r, g, b), 1, cv::LINE_AA);
    }
*/
}

void SearchTemplate::_hsvToRgb(int* r, int* g, int* b, int h, int s, int v)
{
    int i;

    float rgb_min, rgb_max;
    rgb_max = v * 2.55f;
    rgb_min = rgb_max * (100 - s) / 100.0f;

    i = h / 60;
    int difs = h % 60;

    float rgb_adj = (rgb_max - rgb_min) * difs / 60.0f;

    switch (i)
    {
    case 0:
        *r = rgb_max;
        *g = rgb_min + rgb_adj;
        *b = rgb_min;
        break;
    case 1:
        *r = rgb_max - rgb_adj;
        *g = rgb_max;
        *b = rgb_min;
        break;
    case 2:
        *r = rgb_min;
        *g = rgb_max;
        *b = rgb_min + rgb_adj;
        break;
    case 3:
        *r = rgb_min;
        *g = rgb_max - rgb_adj;
        *b = rgb_max;
        break;
    case 4:
        *r = rgb_min + rgb_adj;
        *g = rgb_min;
        *b = rgb_max;
        break;
    default: // case 5:
        *r = rgb_max;
        *g = rgb_min;
        *b = rgb_max - rgb_adj;
        break;
    }
}

