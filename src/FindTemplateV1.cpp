#include "FindTemplateV1.h"
#include <algorithm>
#include <cmath>
#include <omp.h>
#include <thread>
#include <fstream>
#include <immintrin.h>
#include <set>

using namespace SM_V1;

#ifndef SHAPE_MATCH_VISUALIZE_COARSE
#define SHAPE_MATCH_VISUALIZE_COARSE 0
#endif
#ifndef SHAPE_MATCH_VISUALIZE_FINE
#define SHAPE_MATCH_VISUALIZE_FINE 0
#endif
#define COSTTIME_SHOW 1 // 耗时统计，用于算法优化观测

SearchTemplate::SearchTemplate()
    : thread_num_(std::max(1u, std::thread::hardware_concurrency()))
{
}
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

    // 原 AVX 分支使用字节饱和累加，并以右移 8 位代替除以 273，
    // 与训练端的滤波结果不一致。先统一为精确标量实现，便于后续做等价 SIMD 优化。
    (void)useSIMD;
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
            smooth[j * width + i] = static_cast<uint8_t>(sum / 273);
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
    std::vector<T_T::MatchResult> candidates = input;
    std::sort(candidates.begin(), candidates.end(), [](const T_T::MatchResult& lhs, const T_T::MatchResult& rhs)
    {
        return lhs.score > rhs.score;
    });
    std::vector<T_T::MatchResult> result;
    bool nearFlag = false;
    for (const auto& c : candidates)
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
                break;
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
    std::vector<T_T::MatchResult> candidates = input;
    std::sort(candidates.begin(), candidates.end(), [](const T_T::MatchResult& lhs, const T_T::MatchResult& rhs)
    {
        return lhs.score > rhs.score;
    });
    std::vector<T_T::MatchResult> result;
    bool overlapFlag = false;
    for (const auto& c : candidates)
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
                break;
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
    std::vector<float>& p_buf_magnitude,
    bool useSIMD)
{
    // 分配内存
    uint32_t bufferSize = search_image.cols * search_image.rows;
    uint8_t* pInput = (uint8_t*)malloc(bufferSize * sizeof(uint8_t));

    uint8_t* SearchImage = static_cast<uint8_t*>(search_image.data);
    _gaussianFilter(SearchImage, pInput, width, height, true);

    // 待测图像的掩模图
    uint8_t* maskdata = static_cast<uint8_t*>(mask_image.data);

    p_buf_magnitude.assign(bufferSize, 0.0f);

    // 提取待测图像的梯度信息
    if (useSIMD)
    {
        const __m256  vZero   = _mm256_setzero_ps();
        const __m256  vEps    = _mm256_set1_ps(1e-6f);
        const __m256  vThreshold = _mm256_set1_ps(static_cast<float>(search_min_contrast_));
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
                _mm256_storeu_ps(&p_buf_magnitude[idx + 0], mag_lo);
                _mm256_storeu_ps(&p_buf_magnitude[idx + 8], mag_hi);
                const __m256 Gmask_lo = _mm256_and_ps(
                    _mm256_cmp_ps(mag_lo, vEps, _CMP_GT_OQ),
                    _mm256_cmp_ps(mag_lo, vThreshold, _CMP_GE_OQ));
                const __m256 Gmask_hi = _mm256_and_ps(
                    _mm256_cmp_ps(mag_hi, vEps, _CMP_GT_OQ),
                    _mm256_cmp_ps(mag_hi, vThreshold, _CMP_GE_OQ));
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
                Mmask_lo = _mm256_and_ps(Mmask_lo, Gmask_lo);
                Mmask_hi = _mm256_and_ps(Mmask_hi, Gmask_hi);

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
                p_buf_magnitude[index] = mag;
                if (!(mag > 1e-6f && mag >= static_cast<float>(search_min_contrast_)))
                    { p_buf_gradX[index] = 0.f; p_buf_gradY[index] = 0.f; }
                else { p_buf_gradX[index] = float(sdx) / mag; p_buf_gradY[index] = float(sdy) / mag; }
                if (maskdata[index] != 0xFF) { p_buf_gradX[index] = 0.f; p_buf_gradY[index] = 0.f; }
            }
        }

        free(pInput);
        return;
    }

    // Scalar fallback. All current call sites request the SIMD feature extractor,
    // but keep this path correct for debug builds and future non-AVX dispatch.
    for (int i = 1; i < width - 1; i++)
    {
        for (int j = 1; j < height - 1; j++)
        {
            const int index = j * width + i;
            const float dx = float(pInput[index + 1]) - float(pInput[index - 1]);
            const float dy = float(pInput[index + width]) - float(pInput[index - width]);
            const float magnitude = std::sqrt(dx * dx + dy * dy);
            p_buf_magnitude[index] = magnitude;
            if (maskdata[index] == 0xff && magnitude > 1e-6f &&
                magnitude >= static_cast<float>(search_min_contrast_))
            {
                p_buf_gradX[index] = dx / magnitude;
                p_buf_gradY[index] = dy / magnitude;
            }
            else p_buf_gradX[index] = p_buf_gradY[index] = 0.0f;
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
    std::vector<float> pBufMagnitude_new;

    // 初始化
    pBufGradX_new.resize(bufferSize);
    pBufGradY_new.resize(bufferSize);

    // 处理移动滑窗步长,皆为1
    int ijstep = (py_levels == 0) ? 1 : 1;

    // 获取每个像素的梯度信息：dx/dy
    _getFeature(search_image, mask_image, width, height, pBufGradX_new, pBufGradY_new,
                pBufMagnitude_new, true);

    cv::Mat validMask, validIntegral;
    if (variable_visibility_)
    {
        cv::compare(mask_image, cv::Scalar(255), validMask, cv::CMP_EQ);
        cv::integral(validMask, validIntegral, CV_64F);
    }
    const auto rectangleIsVisible = [&](int x0, int y0, int x1, int y1) {
        if (x0 < 0 || y0 < 0 || x1 >= width || y1 >= height || x0 > x1 || y0 > y1)
            return false;
        if (!variable_visibility_) return true;
        if (validIntegral.empty()) return false;
        const double sum = validIntegral.at<double>(y1 + 1, x1 + 1) -
                           validIntegral.at<double>(y0, x1 + 1) -
                           validIntegral.at<double>(y1 + 1, x0) +
                           validIntegral.at<double>(y0, x0);
        return sum == 255.0 * (x1 - x0 + 1) * (y1 - y0 + 1);
    };

    // 每个角度独立记录最优结果，最后一次归约，避免逐像素全局锁竞争。
    const int angle_count = static_cast<int>(shape_info_vec->shape_angle.size());
    std::vector<float> best_scores(angle_count, 0.0f);
    std::vector<int> best_x(angle_count, 0);
    std::vector<int> best_y(angle_count, 0);
    std::vector<double> best_visible(angle_count, 0.0);
    std::vector<double> best_matched(angle_count, 0.0);

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
    for (int k = 0; k < angle_count; k++) //[0]角度的个数
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
            int dx = cvRound(shape_angle->shape_point[mm].x);
            int dy = cvRound(shape_angle->shape_point[mm].y);
            min_dx = std::min(min_dx, dx);
            max_dx = std::max(max_dx, dx);
            min_dy = std::min(min_dy, dy);
            max_dy = std::max(max_dy, dy);
        }

        int adj_start_X = std::max(0, search_region.start_X);
        int adj_end_X = std::min(width, search_region.end_X);
        int adj_start_Y = std::max(0, search_region.start_Y);
        int adj_end_Y = std::min(height, search_region.end_Y);

        // 准备模板点的连续数组，便于SIMD
        std::vector<int> rel_offsets(point_size);
        std::vector<float> tmpl_dx(point_size), tmpl_dy(point_size);
        for (int mm = 0; mm < point_size; ++mm) {
            rel_offsets[mm] = cvRound(shape_angle->shape_point[mm].y) * width +
                              cvRound(shape_angle->shape_point[mm].x);
            tmpl_dx[mm] = shape_angle->shape_point[mm].edge_dx;
            tmpl_dy[mm] = shape_angle->shape_point[mm].edge_dy;
        }

        int TempPiontX = 0;
        int TempPiontY = 0;
        double bestVisibleRatio = 0.0;
        double bestMatchedRatio = 0.0;

        const float normalizedMinScore = min_score / point_size;
        const float normalizedGreediness = greediness < 1.0f
            ? ((1.0f - greediness * min_score) / (1.0f - greediness)) / point_size
            : 0.0f;

        for (int i = adj_start_X; i < adj_end_X; i += ijstep)
        {
            for (int j = adj_start_Y; j < adj_end_Y; j += ijstep)
            {
                float PartialSum = 0; //初始化相似性度量分数
                float PartialScore = 0;
                int visibleCount = 0;

                int matchedCount = 0;
                bool cannotReachBest = false;
                bool rejectedByGreediness = false;
                const bool fixedDenominator = rectangleIsVisible(
                    i + min_dx, j + min_dy, i + max_dx, j + max_dy);
                const bool fastSIMD = useSIMD && fixedDenominator &&
                                      search_min_contrast_ == 0 &&
                                      metric_ == I_I::USE_POLARITY;
                if (fastSIMD)
                {
                    const __m256 zero = _mm256_setzero_ps();
                    const __m256 lower = _mm256_set1_ps(-1.0f);
                    const __m256 upper = _mm256_set1_ps(1.0f);
                    const int base = j * width + i;
                    int m = 0;
                    for (; m + 7 < point_size; m += 8)
                    {
                        const __m256i relative = _mm256_loadu_si256(
                            reinterpret_cast<const __m256i*>(&rel_offsets[m]));
                        const __m256i offsets = _mm256_add_epi32(
                            _mm256_set1_epi32(base), relative);
                        const __m256 sx = _mm256_i32gather_ps(
                            pBufGradX_new.data(), offsets, sizeof(float));
                        const __m256 sy = _mm256_i32gather_ps(
                            pBufGradY_new.data(), offsets, sizeof(float));
                        const __m256 valid = _mm256_or_ps(
                            _mm256_cmp_ps(sx, zero, _CMP_NEQ_OQ),
                            _mm256_cmp_ps(sy, zero, _CMP_NEQ_OQ));
                        __m256 dot = _mm256_add_ps(
                            _mm256_mul_ps(sx, _mm256_loadu_ps(&tmpl_dx[m])),
                            _mm256_mul_ps(sy, _mm256_loadu_ps(&tmpl_dy[m])));
                        dot = _mm256_max_ps(lower, _mm256_min_ps(upper, dot));
                        PartialSum += hsum_ps_avx(_mm256_and_ps(valid, dot));
                        matchedCount += __builtin_popcount(
                            static_cast<unsigned>(_mm256_movemask_ps(valid)));
                        const int processed = m + 8;
                        const int remaining = point_size - processed;
                        if ((PartialSum + remaining) / point_size <= resultscore)
                        {
                            cannotReachBest = true;
                            break;
                        }
                    }
                    for (; !cannotReachBest && m < point_size; ++m)
                    {
                        const int offset = base + rel_offsets[m];
                        const float sx = pBufGradX_new[offset];
                        const float sy = pBufGradY_new[offset];
                        if (sx == 0.0f && sy == 0.0f) continue;
                        ++matchedCount;
                        float dot = sx * tmpl_dx[m] + sy * tmpl_dy[m];
                        PartialSum += std::max(-1.0f, std::min(1.0f, dot));
                    }
                    visibleCount = point_size;
                }
                else
                {
                    for (int m = 0; m < point_size; ++m)
                    {
                        const int curX = i + cvRound(shape_angle->shape_point[m].x);
                        const int curY = j + cvRound(shape_angle->shape_point[m].y);
                        if (curX < 0 || curX >= width || curY < 0 || curY >= height) continue;
                        if (mask_image.at<unsigned char>(curY, curX) != 255) continue;
                        ++visibleCount;
                        const int offset = curY * width + curX;
                        const bool hasGradient = pBufGradX_new[offset] != 0.0f ||
                                                 pBufGradY_new[offset] != 0.0f;
                        if (hasGradient)
                        {
                            ++matchedCount;
                            float dot = pBufGradX_new[offset] * tmpl_dx[m] +
                                        pBufGradY_new[offset] * tmpl_dy[m];
                            dot = std::max(-1.0f, std::min(1.0f, dot));
                            PartialSum += metric_ == I_I::IGNORE_LOCAL_POLARITY ? std::abs(dot) : dot;
                        }
                        const float partialScore = PartialSum / visibleCount;
                        if (!variable_visibility_ && metric_ == I_I::USE_POLARITY &&
                            greediness < 1.0f &&
                            partialScore < std::min(min_score - 1.0f +
                                                       normalizedGreediness * visibleCount,
                                                   normalizedMinScore * visibleCount))
                        {
                            rejectedByGreediness = true;
                            break;
                        }
                        if (fixedDenominator)
                        {
                            const int remaining = point_size - m - 1;
                            const double upperNumerator = metric_ == I_I::IGNORE_GLOBAL_POLARITY
                                ? std::abs(static_cast<double>(PartialSum)) + remaining
                                : static_cast<double>(PartialSum) + remaining;
                            if (upperNumerator / point_size <= resultscore)
                            {
                                cannotReachBest = true;
                                break;
                            }
                        }
                    }
                }
                if (cannotReachBest || rejectedByGreediness) continue;
                if (visibleCount > 0)
                {
                    const double raw = PartialSum / visibleCount;
                    PartialScore = metric_ == I_I::IGNORE_GLOBAL_POLARITY ? std::abs(raw) : raw;
                    PartialScore = std::max(0.0f, std::min(1.0f, PartialScore));
                }
                const double visibleRatio = point_size > 0
                                                ? static_cast<double>(visibleCount) / point_size : 0.0;
                const double matchedRatio = visibleCount > 0
                                                ? static_cast<double>(matchedCount) / visibleCount : 0.0;
                // 每个角度循环只由一个工作线程处理，局部更新无需加锁。
                if (visibleRatio >= min_visible_ratio_ && PartialScore > resultscore)
                {
                    resultscore = PartialScore; // 匹配分数
                    TempPiontX = i; // 坐标X结果值
                    TempPiontY = j; //  坐标Y结果值
                    bestVisibleRatio = visibleRatio;
                    bestMatchedRatio = matchedRatio;
                }
            } // 遍历完毕<搜索区域y>
        } // 遍历完毕<搜索区域x>

        best_scores[k] = resultscore;
        best_x[k] = TempPiontX;
        best_y[k] = TempPiontY;
        best_visible[k] = bestVisibleRatio;
        best_matched[k] = bestMatchedRatio;
    } // 遍历完毕<角度数量>

    float best_score = 0.0f;
    for (int k = 0; k < angle_count; ++k)
    {
        if (best_scores[k] > best_score)
        {
            best_score = best_scores[k];
            result_list->score = best_score;
            result_list->pose.x = best_x[k];
            result_list->pose.y = best_y[k];
            result_list->pose.angle = shape_info_vec->shape_angle[k]->angle;
            result_list->visible_ratio = best_visible[k];
            result_list->matched_ratio = best_matched[k];
        }
    }
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
    std::vector<float> pBufMagnitude;

    std::vector<T_T::MatchResult> totalResultsTemp, resultsfilter;
    std::mutex locker;

    // 提取sobel梯度信息
    _getFeature(search_image, mask_image, width, height, pBufGradX, pBufGradY,
                pBufMagnitude, true);

    cv::Mat validMask, validIntegral;
    if (variable_visibility_)
    {
        cv::compare(mask_image, cv::Scalar(255), validMask, cv::CMP_EQ);
        cv::integral(validMask, validIntegral, CV_64F);
    }
    const auto rectangleIsVisible = [&](int x0, int y0, int x1, int y1) {
        if (validIntegral.empty()) return false;
        if (x0 < 0 || y0 < 0 || x1 >= width || y1 >= height || x0 > x1 || y0 > y1)
            return false;
        const double sum = validIntegral.at<double>(y1 + 1, x1 + 1) -
                           validIntegral.at<double>(y0, x1 + 1) -
                           validIntegral.at<double>(y1 + 1, x0) +
                           validIntegral.at<double>(y0, x0);
        return sum == 255.0 * (x1 - x0 + 1) * (y1 - y0 + 1);
    };

    //相似度计算
    int k_size = shape_info_vec->shape_angle.size();
#if !SHAPE_MATCH_VISUALIZE_COARSE
#pragma omp parallel for num_threads(thread_num_)
#endif
    for (int k = 0; k < k_size; k++) //角度数量
    {
        auto shape_angle = shape_info_vec->shape_angle[k];
        const int point_size = static_cast<int>(shape_angle->shape_point.size());
        //过滤角度不在待搜索范围的匹配运算
        if (shape_angle->angle < search_region.start_angle || shape_angle->angle > search_region.stop_angle)
        {
            continue;
        }

        //每个角度下的粗匹配结果: resultsPerDeg
        //每个角度下粗匹配过近，则选得分较大者{筛选1}：resultsPerDegCandidates
        std::vector<T_T::MatchResult> resultsPerDeg, resultsPerDegCandidates;

        const float normalizedMinScore = min_score / point_size;
        const float normalizedGreediness = greediness < 1.0f
            ? ((1.0f - greediness * min_score) / (1.0f - greediness)) / point_size
            : 0.0f;

        // 更新搜索区域(根据不同角度模板进行搜索)
        // 不同角度下模板特征的外包络框大小不一致
        // 在待测图像上，遍历搜索时，防止目标贴边压不上的可能
        const int search_start_x = std::max(left, search_region.start_X);
        const int search_start_y = std::max(top, search_region.start_Y);
        const int search_end_x = std::min(width - 1, search_region.end_X > 0
                                                         ? search_region.end_X : width - 1);
        const int search_end_y = std::min(height - 1, search_region.end_Y > 0
                                                          ? search_region.end_Y : height - 1);

        for (int i = search_start_x; i <= search_end_x; i++) //搜索范围x
        {
            for (int j = search_start_y; j <= search_end_y; j++) //搜索范围y
            {
                float PartialScore = 0;
                float PartialSum = 0; //初始化相似性度量分数
                int SumOfCoords = 0;
                int visibleCount = 0;
                int matchedCount = 0;
                bool cannotReachThreshold = false;
                bool rejectedByGreediness = false;
                const bool fixedDenominator = rectangleIsVisible(
                    i + shape_angle->bbx.lt_x, j + shape_angle->bbx.lt_y,
                    i + shape_angle->bbx.rb_x, j + shape_angle->bbx.rb_y);

                for (int m = 0; m < point_size; m++) //某角度下的特征点数量
                {
                    int curX = 0;
                    int curY = 0;

                    float iTx = 0;
                    float iTy = 0;
                    float iSx = 0;
                    float iSy = 0;

                    curX = i + cvRound(shape_angle->shape_point[m].x); //模板X坐标
                    curY = j + cvRound(shape_angle->shape_point[m].y); //模板Y坐标

                    if (curX < 0 || curY < 0 || curX > width - 1 || curY > height - 1)
                    {
                        continue; //如果模板超出搜索图像边界范围，跳出继续，加速
                    }
                    if (mask_image.at<unsigned char>(curY, curX) != 255) continue;
                    ++visibleCount;
                    iTx = shape_angle->shape_point[m].edge_dx; //模板X方向的梯度
                    iTy = shape_angle->shape_point[m].edge_dy; //模板Y方向的梯度

                    int offSet = curY * width + curX;
                    iSx = pBufGradX[offSet]; //从搜索图像中获取对应的X梯度
                    iSy = pBufGradY[offSet]; //从搜索图像中获取对应的Y梯度

                    //排除梯度为0的点
                    const bool hasGradient = iSx != 0.0f || iSy != 0.0f;
                    if (hasGradient && (iTx != 0.0f || iTy != 0.0f))
                    {
                        //===================================================================================
                        // 相似性度量公式
                        //===================================================================================
                        ++matchedCount;
                        float dot = std::max(-1.0f, std::min(1.0f,
                            iSx * iTx + iSy * iTy));
                        PartialSum += metric_ == I_I::IGNORE_LOCAL_POLARITY
                                          ? std::abs(dot) : dot;
                    }
                    SumOfCoords = visibleCount;
                    PartialScore = PartialSum / SumOfCoords; // 归一化

                    // Preserve the documented greediness/speed tradeoff for the
                    // legacy polarity metric, but only where the denominator is
                    // known. Partial and masked candidates always use the strict
                    // bound below and can therefore not be discarded heuristically.
                    if (!variable_visibility_ && metric_ == I_I::USE_POLARITY && greediness < 1.0f &&
                        PartialScore < std::min(min_score - 1.0f +
                                                   normalizedGreediness * SumOfCoords,
                                               normalizedMinScore * SumOfCoords))
                    {
                        rejectedByGreediness = true;
                        break;
                    }

                    // Strict score upper bound. This is only valid when the complete
                    // model bounding box lies in the valid mask, hence V is known to
                    // equal N before all points have been visited.
                    if (fixedDenominator)
                    {
                        const int remaining = point_size - m - 1;
                        const double upperNumerator = metric_ == I_I::IGNORE_GLOBAL_POLARITY
                            ? std::abs(static_cast<double>(PartialSum)) + remaining
                            : static_cast<double>(PartialSum) + remaining;
                        if (upperNumerator / point_size <= min_score)
                        {
                            cannotReachThreshold = true;
                            break;
                        }
                    }
                }

                if (cannotReachThreshold || rejectedByGreediness) continue;

                const double visibleRatio = point_size > 0
                                                ? static_cast<double>(visibleCount) / point_size : 0.0;
                if (metric_ == I_I::IGNORE_GLOBAL_POLARITY) PartialScore = std::abs(PartialScore);
                PartialScore = std::max(0.0f, std::min(1.0f, PartialScore));
                const double matchedRatio = visibleCount > 0
                                                ? static_cast<double>(matchedCount) / visibleCount : 0.0;
                if (visibleRatio >= min_visible_ratio_ && PartialScore > min_score)
                {
                    resultsPerDeg.push_back(T_T::MatchResult(
                        T_T::Pose2d(i, j, shape_angle->angle), PartialScore, 1.0, -1,
                        visibleRatio, matchedRatio));
                } // if 语句:大于最小得分值
#if SHAPE_MATCH_VISUALIZE_COARSE
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
        {
            std::lock_guard<std::mutex> guard(locker);
            totalResultsTemp.insert(
                totalResultsTemp.end(), resultsPerDegCandidates.begin(), resultsPerDegCandidates.end());
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

    // 粗层的坐标和角度尚有量化误差，此时做旋转框 NMS 会将真实候选与局部假峰一并删除。
    // 只做近邻峰值合并，重叠率 NMS 延后到 L0 精匹配完成后执行。
    resultsfilter = std::move(totalResultsTemp1);

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


// 1.2 函数：待测图像从粗匹配到精匹配的策略
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

    if (py_levels < 0 || py_levels >= static_cast<int>(model_id->templates.size())) { return false; }
    T_T::ShapeInfo::Ptr pInfoPy = model_id->templates[py_levels];

    // 在金字塔层图像中搜索模板
    int WidthPy = p_image_py.cols;
    int HeightPy = p_image_py.rows;

    // 金字塔高层图往低层搜索策略
    // 因此，此层的参考点应该是上一层匹配的中心点的2倍（金字塔采样比率为1/2）
    ResultPiontX = ((MatchPiontX * 2) < 0) ? 0 : (MatchPiontX * 2);
    ResultPiontY = ((MatchPiontY * 2) < 0) ? 0 : (MatchPiontY * 2);

    // 计算每层模板中心点的位置
    const int level_scale = 1 << py_levels;
    const double level_width = static_cast<double>(model_id->template_cfg.image_width) / level_scale;
    const double level_height = static_cast<double>(model_id->template_cfg.image_height) / level_scale;
    const double angle_rad = MatchAngle * CV_PI / 180.0;
    ReferPointX = static_cast<int>(std::ceil(
        (std::abs(std::cos(angle_rad)) * level_width + std::abs(std::sin(angle_rad)) * level_height) * 0.5));
    ReferPointY = static_cast<int>(std::ceil(
        (std::abs(std::sin(angle_rad)) * level_width + std::abs(std::cos(angle_rad)) * level_height) * 0.5));


    constexpr int refinement_radius = 6;
    // 裁切范围覆盖完整模板及对称精匹配搜索半径。
    Row1 = std::max(0, ResultPiontX - ReferPointX - refinement_radius);
    Col1 = std::max(0, ResultPiontY - ReferPointY - refinement_radius);
    Row2 = std::min(WidthPy, ResultPiontX + ReferPointX + refinement_radius + 1);
    Col2 = std::min(HeightPy, ResultPiontY + ReferPointY + refinement_radius + 1);

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
    // 以上一层候选的2倍坐标为中心，使用对称且闭区间等价的局部搜索窗。
    const int predicted_x = ResultPiontX - Row1;
    const int predicted_y = ResultPiontY - Col1;
    SearchRegion.start_X = std::max(0, predicted_x - refinement_radius);
    SearchRegion.start_Y = std::max(0, predicted_y - refinement_radius);
    SearchRegion.end_X = std::min(cropImgW, predicted_x + refinement_radius + 1);
    SearchRegion.end_Y = std::min(cropImgH, predicted_y + refinement_radius + 1);
    // 搜索角度根据上层匹配角度逆时针、顺时针各偏移4度
    SearchRegion.start_angle = (MatchAngle - 4);
    SearchRegion.stop_angle = (MatchAngle + 4);

    if (pInfoPy == nullptr) return false;

    // 待测图像精匹配
    _fineMatching(cropImage, cropMask, pInfoPy, py_levels, cropImgW, cropImgH, min_score, greediness, SearchRegion,
                  result_list_low,false);
#if SHAPE_MATCH_VISUALIZE_FINE
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
    return searchTemplate(image, s_mask_image, model_id, angle_start, angle_extent,
                          min_score, num_matches, max_overlap, num_levels, greediness,
                          sort_by_y, T_T::ScaleSearchCfg(), result_list);
}

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
    return searchTemplate(image, s_mask_image, roi, model_id, angle_start, angle_extent,
                          min_score, num_matches, max_overlap, num_levels, greediness,
                          sort_by_y, T_T::ScaleSearchCfg(), result_list);
}

namespace
{
void refineSubpixelPosition(T_T::MatchResult& result, const T_T::ShapeInfo::Ptr& shapeInfo,
                            const cv::Mat& image, const cv::Mat& mask,
                            double minVisibleRatio, int minContrast, I_I::Metric metric,
                            double angleStep, double angleStart, double angleEnd);
}

// 单尺度匹配内核，由公开重载统一调用。
bool SearchTemplate::_searchTemplateSingleScale(
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
    {
        // 图像与掩模必须在同一 ROI 坐标系中裁切。
        roi.crop(image, image);
        if (!s_mask_image.empty()) roi.crop(s_mask_image, s_mask_image);
    }

    // 补充掩模图像，防止掩模图像为空
    if(s_mask_image.empty())
        s_mask_image = cv::Mat(image.rows, image.cols, CV_8UC1, cv::Scalar(255));

    ///开始模板匹配工作
    std::mutex locker;
    //多核并行多线程加速的核数
    thread_num_ = std::thread::hardware_concurrency();

    cv::Mat Image_c, smaskimage_c;
    cv::Mat Image = image;
    cv::Mat smaskimage = s_mask_image;
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

        // 对齐金字塔尺寸；部分可见搜索还需把模板中心的搜索域扩展到图外。
        const int pyramid_alignment = 1 << num_levels;
        const int BorderedWidth = ((Image.cols + pyramid_alignment - 1) /
                                   pyramid_alignment) * pyramid_alignment;
        const int BorderedHeight = ((Image.rows + pyramid_alignment - 1) /
                                    pyramid_alignment) * pyramid_alignment;
        const int x2Offset = BorderedWidth - Image.cols;
        const int y2Offset = BorderedHeight - Image.rows;
        const int requiredMargin = min_visible_ratio_ < 1.0
            ? static_cast<int>(std::ceil(std::hypot(model_id->template_cfg.image_width,
                                                    model_id->template_cfg.image_height) * 0.5)) + 2
            : 0;
        const int partialMargin = ((requiredMargin + pyramid_alignment - 1) /
                                   pyramid_alignment) * pyramid_alignment;
        top = (y2Offset + 1) / 2 + partialMargin;
        bottom = y2Offset / 2 + partialMargin;
        left = (x2Offset + 1) / 2 + partialMargin;
        right = x2Offset / 2 + partialMargin;

        if (top > 0 || bottom > 0 || left > 0 || right > 0)
        {
            // Replicate intensity to avoid an artificial gradient at the image
            // boundary. The zero-padded mask below still excludes all pixels
            // outside the original image from visibility and scoring.
            cv::copyMakeBorder(Image, ImgBordered, top, bottom, left, right,
                               cv::BORDER_REPLICATE);
            cv::copyMakeBorder(smaskimage, MaskBordered, top, bottom, left, right,
                               cv::BORDER_CONSTANT, cv::Scalar(0));
        }
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
        std::cout << "Preprocessing time: " << duration_prepare.count() << " ms." << std::endl;
#endif

#if COSTTIME_SHOW
        auto start_coarse = std::chrono::high_resolution_clock::now();
#endif
        // 顶层受下采样量化影响更大：使用略低的候选阈值保证召回，
        // 最终判定仍由 L0 精匹配使用用户指定的 min_score 完成。
        const float coarse_min_score = std::max(0.4f, min_score - 0.3f);
        // 金字塔最高层粗匹配（全图搜索）
        _coarseMatching(
            pImage,
            pMask,
            pInfoPy,
            WidthPy,
            HeightPy,
            modelwidth,
            modelheight,
            min_visible_ratio_ < 1.0 ? 0 : Left,
            min_visible_ratio_ < 1.0 ? 0 : Top,
            coarse_min_score,
            greediness,
            max_overlap,
            SearchRegion,
            ResultListPyRude);
#if COSTTIME_SHOW
        auto                                      end_coarse      = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration_coarse = end_coarse - start_coarse;
        std::cout << "Coarse matching time: " << duration_coarse.count() << " ms." << std::endl;
#endif

#if SHAPE_MATCH_VISUALIZE_COARSE
        printf("Coarse matching result count: %ld\n", ResultListPyRude.size());
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
            pImageBGR, "Py" + std::to_string(num_levels), cv::Point2f(pImageBGR.cols / 2, 10), cv::FONT_HERSHEY_DUPLEX, 0.5,
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
#if !SHAPE_MATCH_VISUALIZE_FINE
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
            } // 结束：非金字塔最高层的逐层精匹配

            if (ResultListLow.score > static_cast<double>(min_score))
            {
                //由padding图上的结果还原至原图上的结果(左扩边界left,上扩边界top)
                T_T::MatchResult Po{
                    T_T::Pose2d(ResultListLow.pose.x - left, ResultListLow.pose.y - top, ResultListLow.pose.angle),
                    ResultListLow.score, 1.0, -1, ResultListLow.visible_ratio,
                    ResultListLow.matched_ratio
                };
                // A valid partial target may have its geometric center outside the
                // image. Visibility and score already describe whether enough of
                // the template is present, so rejecting an out-of-image center
                // incorrectly drops legitimate border matches.
                std::lock_guard<std::mutex> resultGuard(locker);
                TempResult.push_back(Po);
            }
        } // 结束：遍历最高层金字塔的所有候选结果 [OMP]
#if COSTTIME_SHOW
        auto                                      end_fine      = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration_fine = end_fine - start_fine;
        std::cout << "Fine matching time: " << duration_fine.count() << " ms." << std::endl;
#endif

#if COSTTIME_SHOW
        auto start_final = std::chrono::high_resolution_clock::now();
#endif
        //++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        //+++++++++++++++++++++所有精匹配结果，按照重叠率筛选[重则选分数较大者]++++++++++++++++++++
        //++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        findResult = _filterMaxOverLapCandidates(TempResult, max_overlap, model_id->template_cfg.image_height,
                                                 model_id->template_cfg.image_width);

        // Refine only candidates that survived NMS. Performing six bilinear score
        // evaluations for every coarse candidate would erase the benefit of the
        // integer/SIMD search path.
        if (subpixel_refine_)
        for (auto& candidate : findResult)
        {
            candidate.pose.x += left;
            candidate.pose.y += top;
            refineSubpixelPosition(candidate, model_id->templates[0],
                                   ImgBordered, MaskBordered, min_visible_ratio_,
                                   search_min_contrast_, metric_,
                                   model_id->template_cfg.angle_step,
                                   start_angle_, stop_angle_);
            candidate.pose.x -= left;
            candidate.pose.y -= top;
        }

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
        std::cout << "Postprocessing time: " << duration_final.count() << " ms." << std::endl;
#endif

        return true;
} // 作用域结束
    else
    {
        return false;
    }
}

namespace
{
template <typename F>
class ScopeExit
{
public:
    explicit ScopeExit(F action) : action_(std::move(action)) {}
    ScopeExit(ScopeExit&& other) : action_(std::move(other.action_)), active_(other.active_)
    { other.active_ = false; }
    ~ScopeExit() { if (active_) action_(); }
    ScopeExit(const ScopeExit&) = delete;
    ScopeExit& operator=(const ScopeExit&) = delete;

private:
    F action_;
    bool active_ = true;
};

template <typename F>
ScopeExit<F> makeScopeExit(F action)
{
    return ScopeExit<F>(std::move(action));
}

bool validScaleCfg(const T_T::ScaleSearchCfg& cfg)
{
    return std::isfinite(cfg.scale_min) && std::isfinite(cfg.scale_max) &&
           std::isfinite(cfg.scale_step) && std::isfinite(cfg.min_visible_ratio) &&
           cfg.scale_min > 0.0 &&
           cfg.scale_max >= cfg.scale_min && cfg.scale_step > 0.0 &&
           cfg.min_visible_ratio > 0.0 && cfg.min_visible_ratio <= 1.0 &&
           cfg.min_contrast >= 0 && cfg.min_contrast <= 361 &&
           (cfg.metric == I_I::USE_POLARITY ||
            cfg.metric == I_I::IGNORE_LOCAL_POLARITY ||
            cfg.metric == I_I::IGNORE_GLOBAL_POLARITY);
}

bool bilinearNormalizedGradient(const cv::Mat& image, double x, double y,
                                float& gradientX, float& gradientY, float& rawMagnitude,
                                int minContrast)
{
    if (x < 1.0 || y < 1.0 || x >= image.cols - 2.0 || y >= image.rows - 2.0)
        return false;
    const int x0 = static_cast<int>(std::floor(x));
    const int y0 = static_cast<int>(std::floor(y));
    const float ax = static_cast<float>(x - x0);
    const float ay = static_cast<float>(y - y0);
    float gx[4], gy[4], magnitudes[4];
    int index = 0;
    for (int yy = 0; yy <= 1; ++yy)
        for (int xx = 0; xx <= 1; ++xx, ++index)
        {
            const int px = x0 + xx;
            const int py = y0 + yy;
            const float dx = static_cast<float>(image.at<unsigned char>(py, px + 1)) -
                             image.at<unsigned char>(py, px - 1);
            const float dy = static_cast<float>(image.at<unsigned char>(py + 1, px)) -
                             image.at<unsigned char>(py - 1, px);
            const float magnitude = std::sqrt(dx * dx + dy * dy);
            magnitudes[index] = magnitude;
            gx[index] = magnitude > 1e-6f ? dx / magnitude : 0.0f;
            gy[index] = magnitude > 1e-6f ? dy / magnitude : 0.0f;
        }
    gradientX = (1.0f - ay) * ((1.0f - ax) * gx[0] + ax * gx[1]) +
                ay * ((1.0f - ax) * gx[2] + ax * gx[3]);
    gradientY = (1.0f - ay) * ((1.0f - ax) * gy[0] + ax * gy[1]) +
                ay * ((1.0f - ax) * gy[2] + ax * gy[3]);
    rawMagnitude = (1.0f - ay) * ((1.0f - ax) * magnitudes[0] + ax * magnitudes[1]) +
                   ay * ((1.0f - ax) * magnitudes[2] + ax * magnitudes[3]);
    if (!(rawMagnitude > 1e-6f && rawMagnitude >= static_cast<float>(minContrast)))
        gradientX = gradientY = 0.0f;
    return true;
}

struct SubpixelScore
{
    double score = -std::numeric_limits<double>::infinity();
    double visible_ratio = 0.0;
    double matched_ratio = 0.0;
};

SubpixelScore subpixelShapeScore(const T_T::ShapeAngle& shape,
                          const cv::Mat& image, const cv::Mat& mask,
                          double centerX, double centerY,
                          double minVisibleRatio, int minContrast, I_I::Metric metric)
{
    SubpixelScore result;
    double sum = 0.0;
    size_t visible = 0;
    size_t matched = 0;
    const size_t sampleStep = std::max<size_t>(1, (shape.shape_point.size() + 511) / 512);
    size_t sampled = 0;
    for (size_t pointIndex = 0; pointIndex < shape.shape_point.size(); pointIndex += sampleStep)
    {
        const auto& point = shape.shape_point[pointIndex];
        ++sampled;
        const double x = centerX + point.x;
        const double y = centerY + point.y;
        const int nearestX = cvRound(x);
        const int nearestY = cvRound(y);
        if (nearestX < 0 || nearestX >= mask.cols || nearestY < 0 || nearestY >= mask.rows ||
            mask.at<unsigned char>(nearestY, nearestX) != 255) continue;
        ++visible;
        float sx = 0.0f, sy = 0.0f, magnitude = 0.0f;
        if (!bilinearNormalizedGradient(image, x, y, sx, sy, magnitude, minContrast) ||
            (sx == 0.0f && sy == 0.0f)) continue;
        ++matched;
        double dot = std::max(-1.0, std::min(1.0,
            static_cast<double>(sx * point.edge_dx + sy * point.edge_dy)));
        sum += metric == I_I::IGNORE_LOCAL_POLARITY ? std::abs(dot) : dot;
    }
    result.visible_ratio = sampled ? static_cast<double>(visible) / sampled : 0.0;
    result.matched_ratio = visible ? static_cast<double>(matched) / visible : 0.0;
    if (sampled == 0 || static_cast<double>(visible) / sampled < minVisibleRatio || visible == 0)
        return result;
    double score = sum / visible;
    if (metric == I_I::IGNORE_GLOBAL_POLARITY) score = std::abs(score);
    result.score = std::max(0.0, std::min(1.0, score));
    return result;
}

void refineSubpixelPosition(T_T::MatchResult& result, const T_T::ShapeInfo::Ptr& shapeInfo,
                            const cv::Mat& image, const cv::Mat& mask,
                            double minVisibleRatio, int minContrast, I_I::Metric metric,
                            double angleStep, double angleStart, double angleEnd)
{
    if (!shapeInfo || shapeInfo->shape_angle.empty()) return;
    const auto& canonical = *shapeInfo->shape_angle.front();
    const auto rotatedShape = [&](double angle) {
        T_T::ShapeAngle rotated;
        rotated.angle = angle;
        rotated.shape_point.reserve(canonical.shape_point.size());
        const double radians = -angle * CV_PI / 180.0;
        const double c = std::cos(radians), s = std::sin(radians);
        for (const auto& source : canonical.shape_point)
        {
            T_T::ShapePoint point;
            point.x = source.x * c + source.y * s;
            point.y = -source.x * s + source.y * c;
            point.edge_dx = static_cast<float>(source.edge_dx * c + source.edge_dy * s);
            point.edge_dy = static_cast<float>(-source.edge_dx * s + source.edge_dy * c);
            rotated.shape_point.push_back(point);
        }
        return rotated;
    };
    const auto score = [&](double x, double y, double angle) {
        const T_T::ShapeAngle rotated = rotatedShape(angle);
        return subpixelShapeScore(rotated, image, mask, x, y, minVisibleRatio,
                                  minContrast, metric);
    };
    const T_T::MatchResult original = result;
    double refinedAngle = result.pose.angle;
    const SubpixelScore centerResult = score(result.pose.x, result.pose.y, refinedAngle);
    const double center = centerResult.score;
    if (!std::isfinite(center)) return;
    const double h = std::max(0.25, std::min(5.0, std::abs(angleStep)));
    if (refinedAngle - h >= angleStart && refinedAngle + h <= angleEnd)
    {
        const double before = score(result.pose.x, result.pose.y, refinedAngle - h).score;
        const double after = score(result.pose.x, result.pose.y, refinedAngle + h).score;
        const double denominator = before - 2.0 * center + after;
        if (std::isfinite(before) && std::isfinite(after) && denominator < -1e-9)
        {
            const double delta = std::max(-0.75, std::min(0.75,
                0.5 * (before - after) / denominator));
            const double candidateAngle = refinedAngle + delta * h;
            const SubpixelScore candidate = score(result.pose.x, result.pose.y, candidateAngle);
            if (candidate.score + 1e-6 >= center) refinedAngle = candidateAngle;
        }
    }
    const double angleCenter = score(result.pose.x, result.pose.y, refinedAngle).score;
    const auto offset = [&](double before, double after) {
        if (!std::isfinite(before) || !std::isfinite(after)) return 0.0;
        const double denominator = before - 2.0 * angleCenter + after;
        if (denominator >= -1e-9) return 0.0;
        return std::max(-0.75, std::min(0.75, 0.5 * (before - after) / denominator));
    };
    const double dx = offset(score(result.pose.x - 1.0, result.pose.y, refinedAngle).score,
                             score(result.pose.x + 1.0, result.pose.y, refinedAngle).score);
    const double dy = offset(score(result.pose.x, result.pose.y - 1.0, refinedAngle).score,
                             score(result.pose.x, result.pose.y + 1.0, refinedAngle).score);
    const SubpixelScore refined = score(result.pose.x + dx, result.pose.y + dy, refinedAngle);
    if (std::isfinite(refined.score) && refined.score + 1e-6 >= center)
    {
        result.pose.x += dx;
        result.pose.y += dy;
        result.pose.angle = refinedAngle;
        // Keep the discrete match score stable: interpolation refines the pose,
        // while its bilinear/sample-limited objective is only an internal optimizer.
        result.score = original.score;
        result.visible_ratio = refined.visible_ratio;
        result.matched_ratio = refined.matched_ratio;
    }
    else result = original;
}
}

bool SearchTemplate::searchTemplate(cv::Mat image, cv::Mat s_mask_image,
                                    T_T::Template::Ptr model_id,
                                    int angle_start, int angle_extent,
                                    float min_score, int num_matches,
                                    float max_overlap, int num_levels,
                                    float greediness, bool sort_by_y,
                                    const T_T::ScaleSearchCfg& scale_cfg,
                                    std::vector<T_T::MatchResult>& result_list)
{
    return searchTemplate(image, s_mask_image, ROI(), model_id, angle_start, angle_extent,
                          min_score, num_matches, max_overlap, num_levels, greediness,
                          sort_by_y, scale_cfg, result_list);
}

bool SearchTemplate::_prepareSearchBatch(
    cv::Mat image, cv::Mat mask, const ROI& roi,
    const T_T::ScaleSearchCfg& scale_cfg,
    std::vector<PreparedScaleInput>& prepared) const
{
    cv::Mat workImage = image;
    cv::Mat workMask = mask;
    if (!roi.empty())
    {
        if (!roi.crop(image, workImage)) return false;
        if (!mask.empty() && !roi.crop(mask, workMask)) return false;
    }

    prepared.clear();
    const double epsilon = scale_cfg.scale_step * 1e-6;
    for (double scale = scale_cfg.scale_min; scale <= scale_cfg.scale_max + epsilon;
         scale += scale_cfg.scale_step)
    {
        PreparedScaleInput input;
        input.scale = scale;
        const double inverseScale = 1.0 / scale;
        cv::resize(workImage, input.image, cv::Size(), inverseScale, inverseScale,
                   inverseScale < 1.0 ? cv::INTER_AREA : cv::INTER_LINEAR);
        if (!workMask.empty())
            cv::resize(workMask, input.mask, input.image.size(), 0.0, 0.0, cv::INTER_NEAREST);
        prepared.push_back(std::move(input));
    }
    return true;
}

bool SearchTemplate::searchTemplate(cv::Mat image, cv::Mat s_mask_image, ROI roi,
                                    T_T::Template::Ptr model_id,
                                    int angle_start, int angle_extent,
                                    float min_score, int num_matches,
                                    float max_overlap, int num_levels,
                                    float greediness, bool sort_by_y,
                                    const T_T::ScaleSearchCfg& scale_cfg,
                                    std::vector<T_T::MatchResult>& result_list)
{
    if (image.empty() || !model_id || !validScaleCfg(scale_cfg)) return false;
    std::lock_guard<std::mutex> searchGuard(search_mutex_);
    std::vector<PreparedScaleInput> prepared;
    if (!_prepareSearchBatch(image, s_mask_image, roi, scale_cfg, prepared)) return false;

    const double oldVisibleRatio = min_visible_ratio_;
    const bool oldSubpixelRefine = subpixel_refine_;
    const int oldSearchMinContrast = search_min_contrast_;
    const I_I::Metric oldMetric = metric_;
    const bool oldVariableVisibility = variable_visibility_;
    min_visible_ratio_ = scale_cfg.min_visible_ratio;
    subpixel_refine_ = scale_cfg.subpixel_refine;
    search_min_contrast_ = scale_cfg.min_contrast;
    metric_ = static_cast<I_I::Metric>(scale_cfg.metric);
    variable_visibility_ = scale_cfg.min_visible_ratio < 1.0 || !s_mask_image.empty();
    const auto restore = [&]() {
        min_visible_ratio_ = oldVisibleRatio;
        subpixel_refine_ = oldSubpixelRefine;
        search_min_contrast_ = oldSearchMinContrast;
        metric_ = oldMetric;
        variable_visibility_ = oldVariableVisibility;
    };
    auto stateGuard = makeScopeExit(restore);
    return _searchTemplatePrepared(prepared, roi, model_id, angle_start,
        angle_extent, min_score, num_matches, max_overlap, num_levels, greediness,
        sort_by_y, scale_cfg, result_list);
}

bool SearchTemplate::_searchTemplatePrepared(
    const std::vector<PreparedScaleInput>& prepared, const ROI& roi,
    T_T::Template::Ptr model_id, int angle_start, int angle_extent,
    float min_score, int num_matches, float max_overlap, int num_levels,
    float greediness, bool sort_by_y, const T_T::ScaleSearchCfg& scale_cfg,
    std::vector<T_T::MatchResult>& result_list)
{
    std::vector<T_T::MatchResult> all;
    const double epsilon = scale_cfg.scale_step * 1e-6;
    for (const auto& input : prepared)
    {
        const double scale = input.scale;

        std::vector<T_T::MatchResult> perScale;
        if (!_searchTemplateSingleScale(input.image, input.mask, ROI(), model_id,
                                        angle_start, angle_extent, min_score, -1,
                                        max_overlap, num_levels, greediness,
                                        sort_by_y, perScale)) continue;
        for (auto& result : perScale)
        {
            result.pose.x *= scale;
            result.pose.y *= scale;
            result.scale = scale;
            result.template_id = model_id->template_cfg.id;
            if (!roi.empty())
            {
                double x, y, angle;
                if (roi.toImageCoord(result.pose.x, result.pose.y, result.pose.angle,
                                     x, y, angle))
                {
                    result.pose.x = x;
                    result.pose.y = y;
                    result.pose.angle = angle;
                }
            }
            all.push_back(result);
        }
    }

    std::sort(all.begin(), all.end(), [](const T_T::MatchResult& a, const T_T::MatchResult& b)
    {
        return a.score > b.score;
    });
    std::vector<T_T::MatchResult> kept;
    for (const auto& candidate : all)
    {
        bool suppressed = false;
        for (const auto& accepted : kept)
        {
            // 同一模板不同尺度的重复候选按较小框归一化，避免小尺度重复漏抑制。
            const cv::RotatedRect ra(cv::Point2f(candidate.pose.x, candidate.pose.y),
                                     cv::Size2f(model_id->template_cfg.image_width * candidate.scale,
                                                model_id->template_cfg.image_height * candidate.scale),
                                     -candidate.pose.angle);
            const cv::RotatedRect rb(cv::Point2f(accepted.pose.x, accepted.pose.y),
                                     cv::Size2f(model_id->template_cfg.image_width * accepted.scale,
                                                model_id->template_cfg.image_height * accepted.scale),
                                     -accepted.pose.angle);
            std::vector<cv::Point2f> overlap;
            if (cv::rotatedRectangleIntersection(ra, rb, overlap) != cv::INTERSECT_NONE && !overlap.empty()) {
                const double inter = std::abs(cv::contourArea(overlap));
                const double denom = std::min(ra.size.area(), rb.size.area());
                if (denom > 0.0 && inter / denom > max_overlap) suppressed = true;
            }
            if (suppressed) break;
        }
        if (!suppressed) kept.push_back(candidate);
    }

    // HALCON-style interpolation in the scale dimension. The expensive image
    // searches remain discrete; only NMS survivors are associated with the same
    // spatial peak at the immediately adjacent scales and fitted quadratically.
    if (scale_cfg.subpixel_refine &&
        scale_cfg.scale_max - scale_cfg.scale_min >= 2.0 * scale_cfg.scale_step - epsilon)
    {
        const double spatialTolerance = std::max(
            4.0, std::hypot(static_cast<double>(model_id->template_cfg.image_width),
                            static_cast<double>(model_id->template_cfg.image_height)) *
                     scale_cfg.scale_step * 1.5);
        const double angleTolerance = std::max(5.0, model_id->template_cfg.angle_step * 3.0);
        for (auto& candidate : kept)
        {
            const auto adjacent = [&](double wantedScale) -> const T_T::MatchResult* {
                const T_T::MatchResult* best = nullptr;
                double bestDistance = std::numeric_limits<double>::infinity();
                for (const auto& sample : all)
                {
                    if (std::abs(sample.scale - wantedScale) > std::max(1e-6, epsilon)) continue;
                    const double distance = std::hypot(sample.pose.x - candidate.pose.x,
                                                       sample.pose.y - candidate.pose.y);
                    if (distance > spatialTolerance ||
                        std::abs(sample.pose.angle - candidate.pose.angle) > angleTolerance) continue;
                    if (distance < bestDistance ||
                        (std::abs(distance - bestDistance) < 1e-9 &&
                         (!best || sample.score > best->score)))
                    {
                        best = &sample;
                        bestDistance = distance;
                    }
                }
                return best;
            };
            const T_T::MatchResult* lower = adjacent(candidate.scale - scale_cfg.scale_step);
            const T_T::MatchResult* upper = adjacent(candidate.scale + scale_cfg.scale_step);
            if (!lower || !upper) continue;
            const double denominator = lower->score - 2.0 * candidate.score + upper->score;
            if (denominator >= -1e-9) continue;
            const double delta = std::max(-0.75, std::min(
                0.75, 0.5 * (lower->score - upper->score) / denominator));
            candidate.scale = std::max(scale_cfg.scale_min, std::min(
                scale_cfg.scale_max, candidate.scale + delta * scale_cfg.scale_step));
        }
    }
    if (num_matches >= 0 && kept.size() > static_cast<size_t>(num_matches))
        kept.resize(num_matches);
    if (sort_by_y)
        std::sort(kept.begin(), kept.end(), [](const T_T::MatchResult& a, const T_T::MatchResult& b)
        { return a.pose.y < b.pose.y; });
    else
        std::sort(kept.begin(), kept.end(), [](const T_T::MatchResult& a, const T_T::MatchResult& b)
        { return a.pose.x < b.pose.x; });
    result_list.insert(result_list.end(), kept.begin(), kept.end());
    return true;
}

bool SearchTemplate::searchTemplate(cv::Mat image, cv::Mat s_mask_image,
                                    const std::vector<T_T::Template::Ptr>& models,
                                    int angle_start, int angle_extent,
                                    float min_score, int num_matches,
                                    float max_overlap, int num_levels,
                                    float greediness, bool sort_by_y,
                                    const T_T::ScaleSearchCfg& scale_cfg,
                                    std::vector<T_T::MatchResult>& result_list)
{
    return searchTemplate(image, s_mask_image, ROI(), models, angle_start, angle_extent,
                          min_score, num_matches, max_overlap, num_levels, greediness,
                          sort_by_y, scale_cfg, result_list);
}

bool SearchTemplate::searchTemplate(cv::Mat image, cv::Mat s_mask_image, ROI roi,
                                    const std::vector<T_T::Template::Ptr>& models,
                                    int angle_start, int angle_extent,
                                    float min_score, int num_matches,
                                    float max_overlap, int num_levels,
                                    float greediness, bool sort_by_y,
                                    const T_T::ScaleSearchCfg& scale_cfg,
                                    std::vector<T_T::MatchResult>& result_list)
{
    if (image.empty() || models.empty() || !validScaleCfg(scale_cfg)) return false;
    std::set<int> templateIds;
    for (const auto& model : models)
    {
        if (!model || model->template_cfg.id <= 0 ||
            !templateIds.insert(model->template_cfg.id).second)
            return false;
    }

    std::lock_guard<std::mutex> searchGuard(search_mutex_);
    std::vector<PreparedScaleInput> prepared;
    if (!_prepareSearchBatch(image, s_mask_image, roi, scale_cfg, prepared)) return false;

    const double oldVisibleRatio = min_visible_ratio_;
    const bool oldSubpixelRefine = subpixel_refine_;
    const int oldSearchMinContrast = search_min_contrast_;
    const I_I::Metric oldMetric = metric_;
    const bool oldVariableVisibility = variable_visibility_;
    min_visible_ratio_ = scale_cfg.min_visible_ratio;
    subpixel_refine_ = scale_cfg.subpixel_refine;
    search_min_contrast_ = scale_cfg.min_contrast;
    metric_ = static_cast<I_I::Metric>(scale_cfg.metric);
    variable_visibility_ = scale_cfg.min_visible_ratio < 1.0 || !s_mask_image.empty();
    const auto restore = [&]() {
        min_visible_ratio_ = oldVisibleRatio;
        subpixel_refine_ = oldSubpixelRefine;
        search_min_contrast_ = oldSearchMinContrast;
        metric_ = oldMetric;
        variable_visibility_ = oldVariableVisibility;
    };
    auto stateGuard = makeScopeExit(restore);

    std::vector<T_T::MatchResult> merged;
    for (const auto& model : models)
    {
        std::vector<T_T::MatchResult> current;
        if (!_searchTemplatePrepared(prepared, roi, model, angle_start, angle_extent,
                                     min_score, -1, max_overlap, num_levels, greediness,
                                     sort_by_y, scale_cfg, current)) return false;
        merged.insert(merged.end(), current.begin(), current.end());
    }
    std::sort(merged.begin(), merged.end(), [](const T_T::MatchResult& a,
                                               const T_T::MatchResult& b)
    { return a.score > b.score; });
    // 不同模板允许重叠；仅在各模板内部抑制重复候选。
    std::vector<T_T::MatchResult> kept;
    for (const auto& candidate : merged)
    {
        bool suppressed = false;
        for (const auto& accepted : kept)
        {
            if (candidate.template_id != accepted.template_id) continue;
            const T_T::Template::Ptr* candidateModel = nullptr;
            for (const auto& model : models)
                if (model && model->template_cfg.id == candidate.template_id) { candidateModel = &model; break; }
            if (!candidateModel) continue;
            cv::RotatedRect candidateRect(cv::Point2f(candidate.pose.x, candidate.pose.y),
                cv::Size2f((*candidateModel)->template_cfg.image_width * candidate.scale,
                           (*candidateModel)->template_cfg.image_height * candidate.scale), -candidate.pose.angle);
            cv::RotatedRect acceptedRect(cv::Point2f(accepted.pose.x, accepted.pose.y),
                cv::Size2f((*candidateModel)->template_cfg.image_width * accepted.scale,
                           (*candidateModel)->template_cfg.image_height * accepted.scale), -accepted.pose.angle);
            std::vector<cv::Point2f> intersection;
            if (cv::rotatedRectangleIntersection(candidateRect, acceptedRect, intersection) ==
                    cv::INTERSECT_NONE || intersection.empty()) continue;
            const double inter = std::abs(cv::contourArea(intersection));
            const double denom = std::min(candidateRect.size.area(), acceptedRect.size.area());
            if (denom > 0.0 && inter / denom > max_overlap)
            { suppressed = true; break; }
        }
        if (!suppressed) kept.push_back(candidate);
    }
    if (num_matches >= 0 && kept.size() > static_cast<size_t>(num_matches))
        kept.resize(num_matches);
    if (sort_by_y)
        std::sort(kept.begin(), kept.end(), [](const T_T::MatchResult& a, const T_T::MatchResult& b)
        { return a.pose.y < b.pose.y; });
    else
        std::sort(kept.begin(), kept.end(), [](const T_T::MatchResult& a, const T_T::MatchResult& b)
        { return a.pose.x < b.pose.x; });
    result_list.insert(result_list.end(), kept.begin(), kept.end());
    return true;
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
        std::cout << "Failed to load model." << std::endl;
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
    std::cout << "Template data loaded successfully [JSON]." << std::endl;

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
                    double rOrigX, rOrigY;
                    float X, Y, T;
                    //通过坐标变化，将坐标原点0在左上角的图像坐标系转换为笛卡尔坐标系（原点在图像中心，x朝右，y朝上）
                    X = model_id_cp->templates[index]->shape_angle[0]->shape_point[j].x;
                    Y = -model_id_cp->templates[index]->shape_angle[0]->shape_point[j].y;
                    T = X;
                    X = X * std::cos(rad) - Y * std::sin(rad); // 逆时针旋转
                    Y = T * std::sin(rad) + Y * std::cos(rad); // 逆时针旋转

                    rOrigX = X + xOffSet;
                    rOrigY = yOffSet - Y;

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

            // JSON 只保存 0° 特征；旋转重建后必须同步重建每个角度的边界框。
            // 粗匹配依赖该边界框确定合法搜索区域。
            for (const auto& shape_angle : model_id_cp->templates[index]->shape_angle)
            {
                if (shape_angle->shape_point.empty()) { continue; }
                int min_x = static_cast<int>(std::floor(shape_angle->shape_point[0].x));
                int max_x = static_cast<int>(std::ceil(shape_angle->shape_point[0].x));
                int min_y = static_cast<int>(std::floor(shape_angle->shape_point[0].y));
                int max_y = static_cast<int>(std::ceil(shape_angle->shape_point[0].y));
                for (const auto& point : shape_angle->shape_point)
                {
                    min_x = std::min(min_x, static_cast<int>(std::floor(point.x)));
                    max_x = std::max(max_x, static_cast<int>(std::ceil(point.x)));
                    min_y = std::min(min_y, static_cast<int>(std::floor(point.y)));
                    max_y = std::max(max_y, static_cast<int>(std::ceil(point.y)));
                }
                shape_angle->bbx = {min_x, min_y, max_x, max_y};
            }
        }

        // 将更新后的模型数据返回
        temp.reset();
        temp = model_id_cp;
        model_id_cp.reset();
        std::cout << "Template angle data expanded successfully [JSON]." << std::endl;
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
        std::cerr << "Failed to open file for loading: " << path << std::endl;
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
    std::cout << "Template data loaded successfully [binary]." << std::endl;

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
                    double rOrigX, rOrigY;
                    float X, Y, T;
                    //通过坐标变化，将坐标原点0在左上角的图像坐标系转换为笛卡尔坐标系（原点在图像中心，x朝右，y朝上）
                    X = model_id_cp->templates[index]->shape_angle[0]->shape_point[j].x;
                    Y = -model_id_cp->templates[index]->shape_angle[0]->shape_point[j].y;
                    T = X;
                    X = X * std::cos(rad) - Y * std::sin(rad); // 逆时针旋转
                    Y = T * std::sin(rad) + Y * std::cos(rad); // 逆时针旋转

                    rOrigX = X + xOffSet;
                    rOrigY = yOffSet - Y;

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
        std::cout << "Template angle data expanded successfully [binary]." << std::endl;
    }

    // ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑
    return temp;
}


// 辅助函数：检查点是否位于图像边界内
bool isPointInBounds(const cv::Point2f& pt, int cols, int rows) {
    return pt.x >= 0 && pt.x < cols && pt.y >= 0 && pt.y < rows;
}
// 辅助函数：将点裁剪到图像边界内
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
            // 转换为 cv::Point 后调用 clipLine
            cv::Point p1 = cv::Point(static_cast<int>(p1_f.x), static_cast<int>(p1_f.y));
            cv::Point p2 = cv::Point(static_cast<int>(p2_f.x), static_cast<int>(p2_f.y));
            if (cv::clipLine(cv::Rect(0, 0, image.cols, image.rows), p1, p2)) {
                // 转回 cv::Point2f 用于绘制
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
        // 转换为 cv::Point 后调用 clipLine
        cv::Point start = cv::Point(static_cast<int>(start_point.x), static_cast<int>(start_point.y));
        cv::Point end = cv::Point(static_cast<int>(end_point.x), static_cast<int>(end_point.y));
        if (cv::clipLine(cv::Rect(0, 0, image.cols, image.rows), start, end)) {
            // 转回 cv::Point2f 用于绘制
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

namespace
{
void calculateDrawingGradients(const cv::Mat& image, cv::Mat& gradientX, cv::Mat& gradientY)
{
    cv::Mat gray;
    if (image.channels() == 1) gray = image;
    else cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);

    cv::Sobel(gray, gradientX, CV_32F, 1, 0, 1);
    cv::Sobel(gray, gradientY, CV_32F, 0, 1, 1);
    cv::Mat magnitude;
    cv::magnitude(gradientX, gradientY, magnitude);
    cv::Mat valid = magnitude > 1e-6f;
    cv::divide(gradientX, magnitude, gradientX, 1.0, CV_32F);
    cv::divide(gradientY, magnitude, gradientY, 1.0, CV_32F);
    gradientX.setTo(0.0f, ~valid);
    gradientY.setTo(0.0f, ~valid);
}

void drawEdgeLabel(cv::Mat& image, cv::Point2f a, cv::Point2f b,
                   const cv::Point2f& frameCenter, const std::string& text,
                   const cv::Scalar& color)
{
    cv::Point2f tangent = b - a;
    const float edgeLength = std::sqrt(tangent.dot(tangent));
    if (edgeLength < 1.0f || image.empty()) return;
    tangent *= 1.0f / edgeLength;
    // Keep labels readable from left to right.  Image y grows downwards, whereas
    // getRotationMatrix2D uses a positive mathematical (counter-clockwise) angle.
    if (tangent.x < 0.0f)
    {
        std::swap(a, b);
        tangent = -tangent;
    }
    const double screenAngle = std::atan2(static_cast<double>(tangent.y), tangent.x) *
                               180.0 / CV_PI;
    const double fontScale = 0.45;
    const int thickness = 1;
    int baseline = 0;
    const cv::Size textSize = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX,
                                               fontScale, thickness, &baseline);
    const int pad = 3;
    cv::Mat label(textSize.height + baseline + 2 * pad, textSize.width + 2 * pad,
                  CV_8UC3, cv::Scalar(30, 30, 30));
    cv::putText(label, text, cv::Point(pad, pad + textSize.height),
                cv::FONT_HERSHEY_SIMPLEX, fontScale, color, thickness, cv::LINE_AA);
    cv::Mat mask(label.size(), CV_8UC1, cv::Scalar(255));
    const cv::Point2f center(label.cols * 0.5f, label.rows * 0.5f);
    const double opencvAngle = -screenAngle;
    cv::Mat rotation = cv::getRotationMatrix2D(center, opencvAngle, 1.0);
    const cv::Rect2f bounds = cv::RotatedRect(center, label.size(), opencvAngle).boundingRect2f();
    rotation.at<double>(0, 2) += bounds.width * 0.5 - center.x;
    rotation.at<double>(1, 2) += bounds.height * 0.5 - center.y;
    cv::Mat rotatedLabel, rotatedMask;
    cv::warpAffine(label, rotatedLabel, rotation, bounds.size(), cv::INTER_LINEAR,
                   cv::BORDER_CONSTANT, cv::Scalar());
    cv::warpAffine(mask, rotatedMask, rotation, bounds.size(), cv::INTER_NEAREST,
                   cv::BORDER_CONSTANT, cv::Scalar());
    const cv::Point2f midpoint = (a + b) * 0.5f;
    cv::Point2f outward = midpoint - frameCenter;
    const float outwardLength = std::sqrt(outward.dot(outward));
    if (outwardLength > 1e-3f) outward *= 1.0f / outwardLength;
    else outward = cv::Point2f(-tangent.y, tangent.x);
    const cv::Point2f anchor = midpoint + outward * (0.5f * label.rows + 5.0f);
    cv::Rect target(cvRound(anchor.x - rotatedLabel.cols * 0.5f),
                    cvRound(anchor.y - rotatedLabel.rows * 0.5f),
                    rotatedLabel.cols, rotatedLabel.rows);
    const cv::Rect clipped = target & cv::Rect(0, 0, image.cols, image.rows);
    if (clipped.empty()) return;
    const cv::Rect source(clipped.x - target.x, clipped.y - target.y,
                          clipped.width, clipped.height);
    rotatedLabel(source).copyTo(image(clipped), rotatedMask(source));
}
}

void SearchTemplate::drawMatchResults(cv::Mat& image,
                                      const std::vector<T_T::MatchResult>& results,
                                      const T_T::Template::Ptr& model)
{
    cv::Mat gradientX, gradientY;
    calculateDrawingGradients(image, gradientX, gradientY);
    _drawMatchResultsImpl(image, gradientX, gradientY, results, model);
}

void SearchTemplate::_drawMatchResultsImpl(cv::Mat& image, const cv::Mat& gradientX,
                                           const cv::Mat& gradientY,
                                           const std::vector<T_T::MatchResult>& results,
                                           const T_T::Template::Ptr& model)
{
    if (image.empty() || !model || model->templates.empty()) return;
    const auto& shapeInfo = model->templates[0];
    for (size_t index = 0; index < results.size(); ++index)
    {
        const auto& result = results[index];
        if (result.score <= 0.0) continue;
        if (result.template_id != -1 && result.template_id != model->template_cfg.id) continue;
        int r, g, b;
        _hsvToRgb(&r, &g, &b,
                  results.empty() ? 0 : static_cast<int>(360.0 * index / results.size()), 100, 100);
        const cv::Scalar color(b, g, r);
        const T_T::ShapeAngle::Ptr* selected = nullptr;
        double bestAngleDifference = std::numeric_limits<double>::max();
        for (const auto& angle : shapeInfo->shape_angle)
        {
            const double difference = std::abs(angle->angle + result.pose.angle);
            if (difference < bestAngleDifference)
            {
                bestAngleDifference = difference;
                selected = &angle;
            }
        }
        if (selected)
        {
            for (const auto& point : (*selected)->shape_point)
            {
                const int x = cvRound(result.pose.x + point.x * result.scale);
                const int y = cvRound(result.pose.y + point.y * result.scale);
                if (x >= 0 && x < image.cols && y >= 0 && y < image.rows)
                {
                    const float sx = gradientX.at<float>(y, x);
                    const float sy = gradientY.at<float>(y, x);
                    const float similarity = sx * point.edge_dx + sy * point.edge_dy;
                    // Semantic contour colors are independent of the per-result box color:
                    // green = strong, yellow = usable, red/larger = poor or missing edge.
                    cv::Scalar pointColor;
                    int radius = 1;
                    if (similarity >= 0.8f) pointColor = cv::Scalar(0, 255, 0);
                    else if (similarity >= 0.4f) pointColor = cv::Scalar(0, 255, 255);
                    else
                    {
                        pointColor = cv::Scalar(0, 0, 255);
                        radius = 2;
                    }
                    cv::circle(image, cv::Point(x, y), radius, pointColor,
                               cv::FILLED, cv::LINE_AA);
                }
            }
        }

        cv::RotatedRect frame(cv::Point2f(result.pose.x, result.pose.y),
                              cv::Size2f(model->template_cfg.image_width * result.scale,
                                         model->template_cfg.image_height * result.scale),
                              -result.pose.angle);
        cv::Point2f vertices[4];
        frame.points(vertices);
        double longest = -1.0;
        cv::Point2f labelA, labelB;
        for (int edge = 0; edge < 4; ++edge)
        {
            const cv::Point2f originalA = vertices[edge];
            const cv::Point2f originalB = vertices[(edge + 1) % 4];
            cv::Point a(cvRound(originalA.x), cvRound(originalA.y));
            cv::Point bpt(cvRound(originalB.x), cvRound(originalB.y));
            if (!cv::clipLine(cv::Rect(0, 0, image.cols, image.rows), a, bpt)) continue;
            cv::line(image, a, bpt, color, 2, cv::LINE_AA);
            const double length = cv::norm(originalA - originalB);
            if (length > longest)
            {
                longest = length;
                labelA = originalA;
                labelB = originalB;
            }
        }

        // The arrow is the template's positive x-axis and therefore makes the
        // otherwise symmetric rotated rectangle's angle direction unambiguous.
        const double direction = -result.pose.angle * CV_PI / 180.0;
        const float arrowLength = std::max(18.0f, std::min(80.0f,
            static_cast<float>(model->template_cfg.image_width * result.scale * 0.35)));
        cv::Point arrowStart(cvRound(frame.center.x), cvRound(frame.center.y));
        cv::Point arrowEnd(cvRound(frame.center.x + arrowLength * std::cos(direction)),
                           cvRound(frame.center.y + arrowLength * std::sin(direction)));
        if (cv::clipLine(cv::Rect(0, 0, image.cols, image.rows), arrowStart, arrowEnd))
        {
            cv::circle(image, arrowStart, 3, color, cv::FILLED, cv::LINE_AA);
            cv::arrowedLine(image, arrowStart, arrowEnd, color, 2, cv::LINE_AA, 0, 0.22);
        }
        if (longest > 0.0)
        {
            char label[128];
            std::snprintf(label, sizeof(label), "#%zu T:%d %.3f s:%.2f",
                          index, result.template_id, result.score, result.scale);
            drawEdgeLabel(image, labelA, labelB, frame.center, label, color);
        }
    }
}

void SearchTemplate::drawMatchResults(cv::Mat& image,
                                      const std::vector<T_T::MatchResult>& results,
                                      const std::vector<T_T::Template::Ptr>& models)
{
    if (image.empty()) return;
    cv::Mat gradientX, gradientY;
    calculateDrawingGradients(image, gradientX, gradientY);
    for (const auto& model : models)
    {
        if (!model) continue;
        _drawMatchResultsImpl(image, gradientX, gradientY, results, model);
    }
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
