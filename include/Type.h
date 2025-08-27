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

namespace T_T
{
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
        TemplateCfg template_cfg;              // TemplateCfg
        std::vector<ShapeInfo::Ptr> templates; //模板图像的特征信息
        bool is_empty = true;
        bool is_inited; /**<  初始化标志 */
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

    //	匹配结果结构体
    struct MatchResult
    {
        // S_PTR(MatchResult)
        Pose2d pose;  //匹配到的坐标（x,y,angle）
        double score; //匹配得分
    };
} // namespace T_T


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
    enum Metric { USE_POLARITY, IGNORE_LOCAL_POLARITY, IGNORE_GLOBAL_POLARITY };
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
}    // namespace I_I
