#pragma once
#include <mutex>
#include <opencv2/opencv.hpp>
#include "Type.h"

namespace SM_V1
{
    /**
     * @brief 模板创建类，用于生成和保存模板数据。
     */
    class CreateTemplate
    {
    public:
        /**
         * @brief 构造函数
         */
        CreateTemplate();

        /**
         * @brief 析构函数
         */
        ~CreateTemplate();

        /**
         * @brief 创建模板
         *
         * @param temp 输入模板图像
         * @param mask 输入掩模图像
         * @param num_levels 金字塔层数
         * @param angle_start 模板旋转起始角度
         * @param angle_end 模板旋转终止角度
         * @param angle_step 模板旋转角度步长
         * @param create_otsu 是否使用大津阈值法
         * @param min_contrast 最小对比度
         * @param max_contrast 最大对比度
         * @param model_id 输出模板对象指针
         * @return true 创建成功
         * @return false 创建失败
         */
        bool createTemplate(cv::Mat temp, cv::Mat mask, int num_levels, int angle_start, int angle_end,
                            double angle_step, bool create_otsu, int min_contrast,
                            int max_contrast, T_T::Template::Ptr model_id);

        /**
         * @brief 将模板保存为 JSON 文件
         * @param model_id 模板对象指针
         * @param path 保存路径
         * @return true 保存成功
         * @return false 保存失败
         */
        bool saveModelFile2Json(T_T::Template::Ptr model_id, std::string path);

        /**
         * @brief 将模板保存为二进制文件
         * @param model_id 模板对象指针
         * @param path 保存路径
         * @return true 保存成功
         * @return false 保存失败
         */
        bool saveModelFile2Binary(T_T::Template::Ptr model_id, std::string path);

        /**
         * @brief 获取指定金字塔层的模板点
         * @param model_id 模板对象指针
         * @param num_level 金字塔层数
         * @return 模板点集合
         */
        std::vector<cv::Point2d> getTemplatePointPyramid(T_T::Template::Ptr model_id, int num_level);

        /**
         * @brief 可视化模板和匹配结果
         * @param model_id 模板对象指针
         * @param src 输入图像
         * @param pts 匹配点集合
         * @return true 可视化成功
         * @return false 可视化失败
         */
        bool visualizeTemplateAndMatches(T_T::Template::Ptr model_id, cv::Mat src, std::vector<cv::Point3f> pts);

    private:
        /**
         * @brief 初始化形状模型
         */
        void _initialShapeModel(T_T::Template::Ptr model_id);

        /**
         * @brief 初始化金字塔形状模型
         */
        void _initialShapeModelPyd(T_T::ShapeInfo::Ptr shape_info_vec, int angle_start, int angle_stop,
                                   double angle_step);

        /**
         * @brief 创建模型
         */
        bool _createModel(cv::Mat template_img, cv::Mat mask_img, T_T::Template::Ptr model_id);

        /**
         * @brief 构建模型列表
         */
        bool _buildModelList(T_T::ShapeInfo::Ptr shape_info_vec, cv::Mat image_data, cv::Mat mask_data,
                             int min_contrast, int max_contrast);

        /**
         * @brief 提取形状信息
         */
        void _extractShapeInfo(cv::Mat image_data, uint8_t* mask_data, T_T::ShapeAngle::Ptr shape_info_data,
                               int min_contrast, int max_contrast);

        /**
         * @brief 转换长度
         */
        int _convertLength(int length_src);

        /**
         * @brief 从候选特征中选择分散的特征点
         */
        bool _selectScatteredFeatures(
            std::vector<T_T::TemplateFeatures> candidates, std::vector<T_T::TemplateFeatures>& templ,
            int featuresmaxnumber, float distance);

        /**
         * @brief 对形状信息进行旋转偏移
         */
        bool _rotatedShapeInfo(T_T::ShapeInfo::Ptr shape_info_vec, int xOffSet, int yOffSet);

        /**
         * @brief 高斯滤波
         */
        static void _gaussianFilter(uint8_t* corrupted, uint8_t* smooth, int width, int height);

        /**
         * @brief 使用 Canny 边缘检测提取模板特征
         */
        std::vector<T_T::TemplateFeatures> _canny(const cv::Mat& image, const cv::Mat& mask,
                                                  int min_contrast, int max_contrast);
    };
} // namespace SM_V1

