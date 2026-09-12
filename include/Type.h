#pragma once
#include <iostream>
#include <vector>
#include <memory>
#include <opencv2/core/types.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>

#define S_PTR(TYPE)                    \
    typedef std::shared_ptr<TYPE> Ptr; \
    typedef std::weak_ptr<TYPE> WPtr;

namespace I_I
{
    enum Metric { USE_POLARITY = 0, IGNORE_LOCAL_POLARITY = 1, IGNORE_GLOBAL_POLARITY = 2 };
}

namespace T_T
{
    // The historical model coordinate system is image-centre based.  New
    // models may opt into the valid-domain centroid, while old JSON/BIN files
    // remain IMAGE_CENTER when this field is absent.
    enum TemplateOriginMode
    {
        ORIGIN_IMAGE_CENTER = 0,
        ORIGIN_DOMAIN_CENTROID = 1
    };

    enum EdgeMethod
    {
        CURRENT = 0,
        DEVERNAY = 1,
        // Appended for binary/JSON compatibility: legacy values remain stable.
        CANNY_PIXEL = 2,
        EDGE_CURRENT = CURRENT,
        EDGE_DEVERNAY = DEVERNAY,
        EDGE_CANNY_PIXEL = CANNY_PIXEL
    };

    //模板制作输入参数
    struct TemplateCfg
    {
        int num_levels;    //金字塔层数
        int angle_start;   //模板旋转起始角度
        int angle_end;     //模板旋转终止幅度
        double angle_step; //角度步长
        bool create_otsu;  //自动阈值分割标志
        int max_contrast;  //高阈值
        int min_contrast;  //低阈值
        int id;            //模板ID
        int image_width;   //原模板图像宽度
        int image_height;  //原模板图像高度
        bool is_inited;    //初始化标志
        EdgeMethod edge_method; //边缘特征提取后端
        TemplateOriginMode origin_mode; //模板坐标原点策略
        double origin_x; //原图像坐标中的模板原点 X
        double origin_y; //原图像坐标中的模板原点 Y

        // createTemplate reads id before it overwrites the remaining fields,
        // so a newly allocated configuration must be fully initialized.
        TemplateCfg()
            : num_levels(0), angle_start(0), angle_end(0), angle_step(1.0),
              create_otsu(false), max_contrast(0), min_contrast(0), id(1),
              image_width(0), image_height(0), is_inited(false), edge_method(EDGE_CURRENT),
              origin_mode(ORIGIN_IMAGE_CENTER), origin_x(0.0), origin_y(0.0)
        {
        }
    };

    //模板提取的特征点信息
    struct TemplateFeatures
    {
        double x;
        double y;
        float edge_dx;  // X方向梯度
        float edge_dy;  // Y方向梯度
        float edge_mag; //梯度模
    };

    //每角度下各特征点
    struct ShapePoint
    {
        double x, y;   //模板坐标数组（x,y）
        float edge_dx; // X方向梯度
        float edge_dy; // Y方向梯度
    };

    // 用于计算每层金字塔模板的最小外接矩形的左上、右下点
    struct BboundingBox
    {
        int lt_x; //特征外接矩形左上x
        int lt_y; //特征外接矩形左上y
        int rb_x; //特征外接矩形右下x
        int rb_y; //特征外接矩形右下y
    };

    //每层金字塔下各角度
    struct ShapeAngle
    {
        S_PTR(ShapeAngle)
        BboundingBox bbx;                    //特征最小外接矩形
        double angle;                        //旋转角度
        std::vector<ShapePoint> shape_point; //轮廓点数量
    };

    //金字塔各层
    struct ShapeInfo
    {
        S_PTR(ShapeInfo)
        std::vector<ShapeAngle::Ptr> shape_angle; //角度个数
    };

    //	模板文件结构体
    struct Template
    {
        S_PTR(Template)
        TemplateCfg template_cfg;              // 模板配置
        std::vector<ShapeInfo::Ptr> templates; //模板图像的特征信息
        bool is_empty;
        bool is_inited; /**<  初始化标志 */

        Template() : template_cfg(), is_empty(true), is_inited(false) {}
    };

    //	搜索区域结构体
    struct SearchCfg
    {
        int start_X;     // X方向起点
        int start_Y;     // Y方向起点
        int end_X;       // X方向终点
        int end_Y;       // Y方向终点
        int step_angle;  //搜索角度步长
        int range_angle; //搜索角度数目
        int start_angle; //搜索预先角度
        int stop_angle;  //搜索终止角度
    };

    struct Pose2d
    {
        double x, y, angle;
        Pose2d(double x_, double y_, double angle_) : x(x_), y(y_), angle(angle_) {}
    };

    /**
     * @brief 多尺度及边界匹配参数。
     *
     * scale_step 为相邻离散尺度的增量；当三个尺度参数均为 1 时退化为
     * 原有的同尺度匹配。min_visible_ratio 用于限制模板落在图像内的最小
     * 特征比例，1 表示要求模板完整可见。
     */
    struct ScaleSearchCfg
    {
        double scale_min;
        double scale_max;
        double scale_step;
        double min_visible_ratio;
        bool subpixel_refine;
        int min_contrast;
        int metric;
        bool use_simd;

        ScaleSearchCfg(double scale_min_ = 1.0, double scale_max_ = 1.0,
                       double scale_step_ = 1.0, double min_visible_ratio_ = 1.0,
                       bool subpixel_refine_ = false, int min_contrast_ = 0,
                       int metric_ = I_I::USE_POLARITY, bool use_simd_ = false)
            : scale_min(scale_min_), scale_max(scale_max_), scale_step(scale_step_),
              min_visible_ratio(min_visible_ratio_), subpixel_refine(subpixel_refine_),
              min_contrast(min_contrast_), metric(metric_), use_simd(use_simd_)
        {
        }
    };

    //	匹配结果结构体
    struct MatchResult
    {
        Pose2d pose;         // 匹配到的坐标（x,y,angle）
        double score;        // 匹配得分
        double scale;        // 目标相对于原始模板的尺度（1.0 为原尺寸）
        int template_id;     // 匹配到的模板 ID，-1 表示未指定
        double visible_ratio; // 落在待测图内的模板特征比例
        double matched_ratio; // 可见点中达到搜索最小对比度的比例

        MatchResult()
            : pose(0.0, 0.0, 0.0), score(0.0), scale(1.0), template_id(-1),
              visible_ratio(1.0), matched_ratio(1.0)
        {
        }

        // 保留现有 `{Pose2d(...), score}` 构造用法的语义。
        MatchResult(const Pose2d& pose_, double score_, double scale_ = 1.0,
                    int template_id_ = -1, double visible_ratio_ = 1.0,
                    double matched_ratio_ = 1.0)
            : pose(pose_), score(score_), scale(scale_), template_id(template_id_),
              visible_ratio(visible_ratio_), matched_ratio(matched_ratio_)
        {
        }
    };
} // T_T 命名空间


namespace I_I
{
    constexpr int MIN_AREA = 256;
    constexpr int CANDIDATE = 5;
    constexpr float INVALID = -1.f;
    constexpr float F_2PI = 6.283185307179586476925286766559f;
    constexpr float COS[] = {
        1.f, 0.994522f, 0.978148f, 0.951057f, 0.913545f, 0.866025f, 0.809017f, 0.743145f,
        0.669131f, 0.587785f, 0.5f, 0.406737f, 0.309017f, 0.207912f, 0.104528f, 0.f,
        -0.104529f, -0.207912f, -0.309017f, -0.406737f, -0.5f, -0.587785f, -0.669131f, -0.743145f,
        -0.809017f, -0.866025f, -0.913545f, -0.951056f, -0.978148f, -0.994522f, -1.f, -0.994522f,
        -0.978148f, -0.951056f, -0.913545f, -0.866025f, -0.809017f, -0.743145f, -0.669131f, -0.587785f,
        -0.5f, -0.406737f, -0.309017f, -0.207912f, -0.104528f, 0.f, 0.104528f, 0.207912f,
        0.309017f, 0.406737f, 0.5f, 0.587785f, 0.669131f, 0.743145f, 0.809017f, 0.866025f,
        0.913545f, 0.951056f, 0.978148f, 0.9999f, 1.f
    };

    // 枚举类型定义
    enum Reduce { NONE = 0, LOW = 10, MEDIUM = 5, HIGH = 2, AUTO };

    // 匹配姿态结果
    struct Pose {
        float x;
        float y;
        float angle;
        float score;
    };

    // 匹配候选结构（用于排序匹配得分）
    struct Candidate {
        double score;
        float angle;
        cv::Point2f pos;

        Candidate() : score(0), angle(0) {}
        Candidate(double _score, float _angle, cv::Point2f _pos) : score(_score), angle(_angle), pos(_pos) {}

        bool operator<(const Candidate& rhs) const {
            return this->score > rhs.score;
        }
    };

    // 训练模板数据结构体
    struct Template {
        float angleStep;
        float radius;
        std::vector<cv::Point2f> edges;
        std::vector<float> angles;
    };

    // 边缘提取参数结构体
    struct EdgeParam {
        float sigma;
        unsigned char low;
        unsigned char high;
        int minLength;
    };

    // 模板模型结构体
    struct Model {
        EdgeParam edgeParam;
        unsigned char minMag;
        Metric metric;
        Reduce reduce;
        float radius;
        cv::Mat source;
        std::vector<Template> templates;
        std::vector<Template> reducedTemplates;
    };
}    // I_I 命名空间
