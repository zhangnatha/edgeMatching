#include "MakeTemplateV1.h"
#include <omp.h>
#include <thread>
#include <fstream>

using namespace SM_V1;

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

    std::vector<uint8_t> pBufOut(buffer_size);
    std::vector<int16_t> pBufGradX(buffer_size); //存取x方向偏导数dx
    std::vector<int16_t> pBufGradY(buffer_size); //存取y方向偏导数dy
    std::vector<int32_t> pBufOrien(buffer_size); //存取梯度方向
    std::vector<float> pBufMag(buffer_size); //存取梯度模

    std::vector<T_T::TemplateFeatures> TF0degree, TF0degree_temp;

    //===================================================================================
    // step 0：获取图像的高斯模糊后的梯度信息[grad_x_edge,grad_y_edge] & 未做模糊的梯度信息[grad_x,grad_y]
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
    // step 1：获取图像的梯度方向[Make Direction]
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
    } // END [S1:Make Direction]

    //===================================================================================
    // step 2：非最大值抑制[NMS]
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
    } // END [S2:NMS]

    //===================================================================================
    // step 3：滞后阈值，双阈值
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
            // step 4：保存数据集
            //===================================================================================
            if (flag != 0) //强边缘标志
            {
                if (fdx != 0 || fdy != 0)
                {
                    float magnitude = (!(std::fabs(magnitude_origin) < 1e-6)) ? (1 / magnitude_origin) : 0;
                    TF0degree.push_back({double(i), double(j), (float)fdx, (float)fdy, magnitude});
                }
            }
        }
    }
    /*
        //===================================================================================
        // step 4：特征数过滤: 用特征点数百分比计算特征点数：Max*_features_rate
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
        angle_info_data->shape_point[m].x = TF0degree_temp[m].x - static_cast<float>(image_data.cols) / 2;
        angle_info_data->shape_point[m].y = TF0degree_temp[m].y - static_cast<float>(image_data.rows) / 2;

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
void CreateTemplate::_initialShapeModel(T_T::Template::Ptr model_id)
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
        float rad = (double)((angle * CV_PI) / 180); // 180/π =angle/rad

        for (int j = 0; j < shapeSize; j++) //轮廓点数量
        {
            //坐标x,y变化
            int rOrigX, rOrigY;
            float X, Y, T;
            //通过坐标变化，将坐标原点0在左上角的图像坐标系转换为笛卡尔坐标系（原点在图像中心，x朝右，y朝上）
            X = shape_info_vec->shape_angle[0]->shape_point[j].x;
            Y = -shape_info_vec->shape_angle[0]->shape_point[j].y;
            T = X;
            X = X * std::cos(rad) - Y * std::sin(rad); // 逆时针旋转
            Y = T * std::sin(rad) + Y * std::cos(rad); // 逆时针旋转

            rOrigX = (X + xOffSet > 0.0) ? (X + xOffSet + 0.5) : (X + xOffSet - 0.5); //四舍五入取整数
            rOrigY = (yOffSet - Y > 0.0) ? (yOffSet - Y + 0.5) : (yOffSet - Y - 0.5); //四舍五入取整数

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
            // 更新旋转后的值mag ---- 不变
            //            shape_info_vec->shape_angle[i]->shape_point[j].edge_mag = shape_info_vec->shape_angle[0]->shape_point[j].edge_mag;
            //            //保持不变
        }
    }
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

    int xOffSet = image_data.cols / 2;
    int yOffSet = image_data.rows / 2;

    // Rotated features
    _rotatedShapeInfo(shape_info_vec, xOffSet, yOffSet);

    // S2-求特征的外接最大矩形
    for (const auto item : shape_info_vec->shape_angle)
    {
        if (item->shape_point.size() != 0) //如果该层没有特征点则跳出
        {
            T_T::BboundingBox bbx;
            // 使用排序计算外接最大矩形框(左上,右下点)
            std::sort(item->shape_point.begin(), item->shape_point.end(),
                      [](const T_T::ShapePoint& pt1s, const T_T::ShapePoint& pt2s) { return pt1s.x < pt2s.x; });
            bbx.lt_x = item->shape_point[0].x;
            std::sort(item->shape_point.begin(), item->shape_point.end(),
                      [](const T_T::ShapePoint& pt1s, const T_T::ShapePoint& pt2s) { return pt1s.y < pt2s.y; });
            bbx.lt_y = item->shape_point[0].y;
            std::sort(item->shape_point.begin(), item->shape_point.end(),
                      [](const T_T::ShapePoint& pt1s, const T_T::ShapePoint& pt2s) { return pt1s.x > pt2s.x; });
            bbx.rb_x = item->shape_point[0].x;
            std::sort(item->shape_point.begin(), item->shape_point.end(),
                      [](const T_T::ShapePoint& pt1s, const T_T::ShapePoint& pt2s) { return pt1s.y > pt2s.y; });
            bbx.rb_y = item->shape_point[0].y;
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
                    cv::pyrDown(template_imgPy1, template_imgPy2,
                                cv::Size(template_imgPy1.cols / 2, template_imgPy1.rows / 2));
                    cv::pyrDown(mask_imgPy1, mask_imgPy2, cv::Size(mask_imgPy1.cols / 2, mask_imgPy1.rows / 2));

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
                    cv::pyrDown(template_imgPy2, template_imgPy3,
                                cv::Size(template_imgPy2.cols / 2, template_imgPy2.rows / 2));
                    cv::pyrDown(mask_imgPy2, mask_imgPy3, cv::Size(mask_imgPy2.cols / 2, mask_imgPy2.rows / 2));

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
                    cv::pyrDown(template_imgPy3, template_imgPy4,
                                cv::Size(template_imgPy3.cols / 2, template_imgPy3.rows / 2));
                    cv::pyrDown(mask_imgPy3, mask_imgPy4, cv::Size(mask_imgPy3.cols / 2, mask_imgPy3.rows / 2));

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
                    cv::pyrDown(template_imgPy4, template_imgPy5,
                                cv::Size(template_imgPy4.cols / 2, template_imgPy4.rows / 2));
                    cv::pyrDown(mask_imgPy4, mask_imgPy5, cv::Size(mask_imgPy4.cols / 2, mask_imgPy4.rows / 2));

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
                    cv::pyrDown(template_imgPy5, template_imgPy6,
                                cv::Size(template_imgPy5.cols / 2, template_imgPy5.rows / 2));
                    cv::pyrDown(mask_imgPy5, mask_imgPy6, cv::Size(mask_imgPy5.cols / 2, mask_imgPy5.rows / 2));

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
                    cv::pyrDown(template_imgPy6, template_imgPy7,
                                cv::Size(template_imgPy6.cols / 2, template_imgPy6.rows / 2));
                    cv::pyrDown(mask_imgPy6, mask_imgPy7, cv::Size(mask_imgPy6.cols / 2, mask_imgPy6.rows / 2));

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
            } // switch case
        } // end for:金字塔层数

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
    T_T::Template::Ptr model_id)
{
    cv::Mat tempMat, maskMat;
    tempMat = temp.clone();
    maskMat = mask.clone();

    // 如果彩色图像，转灰度图像
    if (tempMat.channels() == 3) { cv::cvtColor(tempMat, tempMat, cv::COLOR_BGR2GRAY); }
    if (maskMat.channels() == 3) { cv::cvtColor(maskMat, maskMat, cv::COLOR_BGR2GRAY); }
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
    model_id->template_cfg.min_contrast = min_contrast;
    model_id->template_cfg.max_contrast = max_contrast;
    model_id->template_cfg.id = 1;
    model_id->template_cfg.image_width = tempMat.cols;
    model_id->template_cfg.image_height = tempMat.rows;
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
    else // num_levels = 0,1,2,3,4,5,6,7
    {
        model_id->template_cfg.num_levels = num_levels;
    }
    // 初始化model_id
    CreateTemplate::_initialShapeModel(model_id);

    // model_id存储的特征点坐标（模板中心点为原点坐标）
    CreateTemplate::_createModel(tempMat, maskMat, model_id);

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
    return true;
}

// 保存模板文件json
bool CreateTemplate::saveModelFile2Json(T_T::Template::Ptr model_id, std::string path)
{
    int num_pyramid = model_id->templates.size();
    std::string model_name = path;
    cv::FileStorage fs(model_name, cv::FileStorage::WRITE);

    // shape Match model
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
                //                for(int j = 0; j < templ_templates->shape_angle.size(); j++)    //每个角度
                //                {
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
                            << /*templ_feature.edge_mag <<*/ "]";
                    }
                }
                fs << "]";
                fs << "}";
                //                }
            }
            fs << "]";
            fs << "}";
        }
    }
    fs << "]";
    fs << "}";

    fs.release();
    std::cout << "保存模板文件成功[Json]!" << std::endl;
    return true;
}

// 保存模板为二进制文件
bool CreateTemplate::saveModelFile2Binary(T_T::Template::Ptr model_id, std::string path)
{
    int num_pyramid = model_id->templates.size();

    std::ofstream ofs(path, std::ios::binary);
    if (!ofs.is_open())
    {
        std::cerr << "无法打开文件 " << path << " 来保存数据!" << std::endl;
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

    ofs.close(); // 关闭文件流
    std::cout << "保存模板文件成功[Binary]!" << std::endl;
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
    //    printf("Pyramid Level: Current->%d,Max->%d\n", num_level, model_id->template_cfg.num_levels);
    if (num_level <= model_id->template_cfg.num_levels)
    {
        for (int i = 0; i < model_id->templates[num_level]->shape_angle[0]->shape_point.size(); i++)
        {
            cv::Point2d point_one(
                model_id->template_cfg.image_width / 2 + model_id->templates[num_level]->shape_angle[0]->shape_point[i].
                x,
                model_id->template_cfg.image_height / 2 + model_id->templates[num_level]->shape_angle[0]->shape_point[i]
                .y);
            result_points.push_back(point_one);
        }
    }
    else
    {
        result_points.clear();
    }

    return result_points;
}

