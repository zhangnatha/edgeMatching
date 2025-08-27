#pragma once
#include <mutex>
#include "Type.h"
#include "ROI.h"

#include <opencv2/opencv.hpp>

namespace SM_V1
{
    /**
     * @brief 模板匹配搜索类，用于在图像中查找已训练的模板。
     */
    class SearchTemplate
    {
    public:
        /**
         * @brief 构造函数
         */
        SearchTemplate();

        /**
         * @brief 析构函数
         */
        ~SearchTemplate();

        /**
         * @brief 在图像中进行模板匹配搜索
         *
         * @param image 输入搜索图像
         * @param s_mask_image 输入掩模图像
         * @param model_id 模板指针
         * @param angle_start 搜索起始角度
         * @param angle_extent 搜索角度范围
         * @param min_score 匹配的最小得分阈值
         * @param num_matches 最大匹配数量
         * @param max_overlap 允许的最大重叠度
         * @param num_levels 金字塔层数
         * @param greediness 搜索贪婪度
         * @param sort_by_y 是否按y坐标排序
         * @param result_list 输出的匹配结果列表
         * @return true 搜索成功
         * @return false 搜索失败
         */
        bool searchTemplate(cv::Mat image, cv::Mat s_mask_image,
                            T_T::Template::Ptr model_id,
                            int angle_start, int angle_extent,
                            float min_score, int num_matches,
                            float max_overlap, int num_levels,
                            float greediness, bool sort_by_y,
                            std::vector<T_T::MatchResult>& result_list);

        /**
         * @brief 在指定ROI区域内进行模板匹配搜索
         *
         * @param image 输入搜索图像
         * @param s_mask_image 输入掩模图像
         * @param roi 限定的ROI区域
         * @param temp 模板指针
         * @param angle_start 搜索起始角度
         * @param angle_extent 搜索角度范围
         * @param min_score 匹配的最小得分阈值
         * @param num_matches 最大匹配数量
         * @param max_overlap 允许的最大重叠度
         * @param num_levels 金字塔层数
         * @param greediness 搜索贪婪度
         * @param sort_by_y 是否按y坐标排序
         * @param result_list 输出的匹配结果列表
         * @return true 搜索成功
         * @return false 搜索失败
         */
        bool searchTemplate(cv::Mat image, cv::Mat s_mask_image, ROI roi, T_T::Template::Ptr temp, int angle_start,
                            int angle_extent, float min_score, int num_matches,
                            float max_overlap, int num_levels, float greediness, bool sort_by_y,
                            std::vector<T_T::MatchResult>& result_list);

        /**
         * @brief 从二进制文件加载模板模型
         * @param path 模型文件路径
         * @return 模板对象指针
         */
        T_T::Template::Ptr loadModelFileFromBinary(std::string path);

        /**
         * @brief 从JSON文件加载模板模型
         * @param path 模型文件路径
         * @return 模板对象指针
         */
        T_T::Template::Ptr loadModelFileFromJson(std::string path);

        /**
         * @brief 在图像上绘制匹配结果
         *
         * @param image 输入/输出图像
         * @param results 匹配结果列表
         * @param shapeInfo 模板形状信息
         * @param sort_by_y 是否按y坐标排序
         */
        void drawMatchResults(cv::Mat& image, const std::vector<T_T::MatchResult>& results,
                              T_T::ShapeInfo::Ptr shapeInfo);

    private:
        /**
         * @brief 粗匹配
         */
        void _coarseMatching(cv::Mat search_image, cv::Mat mask_image, T_T::ShapeInfo::Ptr shape_info_vec, int width,
                             int height, int model_width,
                             int model_height, int left, int top, float min_score, float greediness, float max_overlap,
                             T_T::SearchCfg search_region, std::vector<T_T::MatchResult>& resultList);

        /**
         * @brief 提取特征（梯度信息）
         */
        void _getFeature(cv::Mat search_image, cv::Mat mask_image, int width, int height,
        std::vector<float>& p_buf_gradX, std::vector<float>& p_buf_gradY, bool useSIMD);

        /**
         * @brief 粗匹配到精匹配的转换
         */
        bool _coarse2FineMatching(cv::Mat p_image_py, cv::Mat mask_image, T_T::Template::Ptr model_id, int py_levels,
                                  int width, int height, float min_score,
                                  float greediness, T_T::MatchResult* result_list_high,
                                  T_T::MatchResult* result_list_low, int Numpyl);

        /**
         * @brief 精匹配
         */
        void _fineMatching(cv::Mat search_image, cv::Mat mask_image, T_T::ShapeInfo::Ptr shape_info_vec, int py_levels,
                           int width, int height, float min_score,
                           float greediness, T_T::SearchCfg search_region, T_T::MatchResult* result_list, bool useSIMD);

        /**
         * @brief 过滤掉邻近候选点
         */
        std::vector<T_T::MatchResult> _filterNearCandidates(const std::vector<T_T::MatchResult>& input);

        /**
         * @brief 过滤掉重叠过大的候选点
         */
        std::vector<T_T::MatchResult> _filterMaxOverLapCandidates(
            const std::vector<T_T::MatchResult>& input, float max_ovelap, int model_height, int model_width);

        /**
         * @brief 高斯滤波
         */
        void _gaussianFilter(uint8_t* corrupted, uint8_t* smooth, int width, int height, bool useSIMD);

        /**
         * @brief 转换长度
         */
        int _convertLength(int length_src);

        /**
         * @brief 判断两个旋转矩形是否重叠超过阈值
         */
        bool _maxOverlap(const cv::RotatedRect rect1, const cv::RotatedRect& rect2, float overlap);

        /**
         * @brief HSV 转 RGB
         */
        void _hsvToRgb(int* r, int* g, int* b, int h, int s, int v);

    private:
        int thread_num_;       /**< 使用的线程数量 */
        int min_contrast_ = 20; /**< 模板匹配的最小对比度 */
        int max_contrast_ = 75; /**< 模板匹配的最大对比度 */
        int start_angle_;      /**< 搜索起始角度 */
        int stop_angle_;       /**< 搜索终止角度 */
        int image_width_ = 0;  /**< 输入图像宽度 */
        int image_height_ = 0; /**< 输入图像高度 */
    };
} // namespace SM_V1

