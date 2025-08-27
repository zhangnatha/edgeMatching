// inference.cpp
#include "FindTemplateV1.h"
#include "Timer.h"

#define USE_BINARY_MODEL 0

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

#include <opencv2/opencv.hpp>
#include <vector>
#include <random>

#include <opencv2/opencv.hpp>
#include <vector>
#include <random>

void drawMultiLineText(
    cv::Mat& image,                            // 输入/输出图像
    const std::vector<std::string>& texts,     // 多行文本
    const cv::Point& topLeft,                  // 左上角起点
    const std::vector<cv::Scalar>& fontColors,  // 每行文本的字体颜色
    const std::vector<cv::Scalar>& bgColors,   // 每行文本的背景颜色
    double fontScale = 0.5,                    // 字体缩放比例
    int thickness = 2,                         // 字体粗细
    int fontFace = cv::FONT_HERSHEY_SIMPLEX,   // 字体类型
    int padding = 5                            // 文本周围内边距
) {
    if (texts.empty() || image.empty()) return;

    // 获取图像宽度用于自适应字体大小
    int imgWidth = image.cols;
    double adaptiveFontScale = imgWidth / 500.0 * fontScale;

    // 确保字体粗细至少为1
    int adaptiveThickness = std::max(1, static_cast<int>(adaptiveFontScale * thickness));

    // 随机颜色生成器（用于颜色不足时）
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 255);

    // 确保颜色向量大小与文本行数一致
    std::vector<cv::Scalar> fontColorsPadded = fontColors;
    std::vector<cv::Scalar> bgColorsPadded = bgColors;
    fontColorsPadded.resize(texts.size(), cv::Scalar(dis(gen), dis(gen), dis(gen))); // 默认随机颜色
    bgColorsPadded.resize(texts.size(), cv::Scalar(dis(gen), dis(gen), dis(gen)));   // 默认随机背景颜色

    // 计算每行文本的大小
    std::vector<cv::Size> textSizes;
    int totalHeight = 0;
    for (const auto& text : texts) {
        cv::Size textSize = cv::getTextSize(text, fontFace, adaptiveFontScale, adaptiveThickness, nullptr);
        textSizes.push_back(textSize);
        totalHeight += textSize.height + padding; // 每行间增加间距
    }
    totalHeight += padding; // 顶部和底部内边距

    // 绘制每行文本的背景和文本
    int yOffset = topLeft.y + padding;
    for (size_t i = 0; i < texts.size(); ++i) {
        // 绘制单行背景矩形，宽度根据该行文本长度
        cv::Point textPos(topLeft.x + padding, yOffset + textSizes[i].height);
        cv::Point bgTopLeft(topLeft.x, yOffset);
        cv::Point bgBottomRight(topLeft.x + textSizes[i].width + 2 * padding, yOffset + textSizes[i].height + padding);
        cv::rectangle(image, bgTopLeft, bgBottomRight, bgColorsPadded[i], cv::FILLED);

        // 绘制文本
        cv::putText(image, texts[i], textPos, fontFace, adaptiveFontScale, fontColorsPadded[i], adaptiveThickness);
        yOffset += textSizes[i].height + padding;
    }
}

int main(int argc, const char* argv[])
{
    Timer timer(TimerMethod::HighResolutionClock);

    // 主函数：加载搜索图像，读取模型并执行匹配，显示结果
    cv::Mat search_image, search_mask;
    if (argc < 2)
    {
        search_image = cv::imread("../assert/src.bmp", cv::IMREAD_GRAYSCALE);
    }
    else
    {
        search_image = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);
    }

    // 正矩形 ROI 从搜索图像裁剪出子图
    // ROI roi(cv::Rect(1377, 5, 76, 1434));

    // 匹配参数
    const double f_angle_start = -180;
    const double f_angle_stop = 180;
    const double f_min_score = 0.85;
    const int f_matche_numbers = 200;
    const double f_max_overlap = 0.5;
    const int f_pyramid_number = -1;
    const double f_greediness = 0.9;
    bool sort_by_y = true;//按y坐标排序,否则按x坐标排序

#if USE_BINARY_MODEL
    const std::string modelPath = "./model.bin";
#else
    const std::string modelPath = "./model.json";
#endif

    std::vector<T_T::MatchResult> result_list;
    SM_V1::SearchTemplate matcher;
    auto model_ptr = std::make_shared<T_T::Template>();
#if USE_BINARY_MODEL
    model_ptr = matcher.loadModelFileFromBinary(modelPath);
#else
    model_ptr = matcher.loadModelFileFromJson(modelPath);
#endif
    timer.start();
    timer.record("读取模型");

    matcher.searchTemplate(search_image, search_mask, /*roi,*/ model_ptr, f_angle_start, f_angle_stop, f_min_score,
                           f_matche_numbers, f_max_overlap, f_pyramid_number, f_greediness, sort_by_y, result_list);
    timer.record("匹配模型");
    timer.report();

    // 输出结果
    for (size_t i = 0; i < result_list.size(); i++)
    {
        printf("[%ld]:%f,%f,%f,%f\n", i, result_list[i].pose.x, result_list[i].pose.y, result_list[i].pose.angle,
               result_list[i].score);
    }

    // 显示结果
    cv::Mat color;
    cv::cvtColor(search_image, color, cv::COLOR_GRAY2RGB);
    matcher.drawMatchResults(color, result_list, model_ptr->templates[0]);

    // 在左上角绘制耗时（白色背景，黑色字体）
    std::string timeText = "Time Cost: " + std::to_string(timer.get("匹配模型")) + " ms";
    std::string infoText = "Match Number: " + std::to_string(result_list.size());
    std::vector<std::string> texts = { timeText, infoText };

    cv::Point topLeft(10, 10);
    std::vector<cv::Scalar> fontColor = {{255, 255, 255}, {255, 255, 255}};
    std::vector<cv::Scalar> bgColor{{114, 83, 52},{64,116,52}};
    drawMultiLineText(color, texts, topLeft, fontColor, bgColor, 0.5, 2, cv::FONT_HERSHEY_SIMPLEX, 5);

    //保存结果
    cv::imwrite("result.png", color);
    return 0;
}

