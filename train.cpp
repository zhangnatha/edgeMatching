// train.cpp
#include "MakeTemplateV1.h"
#include "Timer.h"
#include <cerrno>
#include <climits>
#include <cstdlib>
#include <iostream>
#include <string>

#define USE_BINARY_MODEL 0

namespace {
bool parseInt(const std::string& text, int& value)
{
    if (text.empty()) return false;
    errno = 0;
    char* end = nullptr;
    const long parsed = std::strtol(text.c_str(), &end, 10);
    if (errno == ERANGE || end == text.c_str() || *end != '\0' ||
        parsed < INT_MIN || parsed > INT_MAX) return false;
    value = static_cast<int>(parsed);
    return true;
}

void usage(const char* program)
{
    std::cout << "Usage: " << program
              << " [template_image] [--id N] [--output FILE] [--pyramid-output FILE]\n";
}
}

int main(int argc, const char* argv[])
{
    Timer timer(TimerMethod::HighResolutionClock);

    // 主函数：加载图像，执行训练和保存模型
    cv::Mat model_image, model_mask;
    std::string imagePath = "../assert/m1.png";
    std::string modelPath = "./model.json";
    std::string pyramidPath;
    int templateId = 1;
    bool imageSpecified = false;
    for (int i = 1; i < argc; ++i) {
        std::string arg(argv[i]);
        if (arg == "--help" || arg == "-h") {
            usage(argv[0]);
            return 0;
        } else if (arg == "--id") {
            if (i + 1 >= argc || !parseInt(argv[++i], templateId)) {
                usage(argv[0]);
                return 2;
            }
        } else if (arg == "--output") {
            if (i + 1 >= argc || argv[i + 1][0] == '\0' || argv[i + 1][0] == '-') {
                usage(argv[0]);
                return 2;
            }
            modelPath = argv[++i];
        } else if (arg == "--pyramid-output") {
            if (i + 1 >= argc || argv[i + 1][0] == '\0' || argv[i + 1][0] == '-') {
                usage(argv[0]);
                return 2;
            }
            pyramidPath = argv[++i];
        } else if (!arg.empty() && arg[0] != '-') {
            if (imageSpecified) {
                std::cerr << "Only one template image may be specified.\n";
                usage(argv[0]);
                return 2;
            }
            imagePath = arg;
            imageSpecified = true;
        } else {
            std::cerr << "Unknown option: " << arg << '\n';
            usage(argv[0]);
            return 2;
        }
    }
    if (templateId <= 0) { std::cerr << "Invalid template ID.\n"; return 2; }
    {
        model_image = cv::imread(imagePath, cv::IMREAD_GRAYSCALE);
        if (model_image.empty()) { std::cerr << "Failed to read template image: " << imagePath << '\n'; return 1; }
        model_mask = cv::Mat(model_image.rows, model_image.cols, CV_8UC1, cv::Scalar(255));
    }
    // 训练参数
    const int c_pyramid_number = -1;
    const double c_angle_start = -180;
    const double c_angle_end = 180;
    const double c_angle_step = 1;
    const double c_min_contrast = 25;
    const double c_max_contrast = 100;
#if USE_BINARY_MODEL
    modelPath = "./model.bin";
#endif

    // 训练模型
    SM_V1::CreateTemplate trainer;
    T_T::Template::Ptr modelId = std::make_shared<T_T::Template>();
    modelId->template_cfg.id = templateId;
    timer.start();
    if (!trainer.createTemplate(model_image, model_mask, c_pyramid_number, c_angle_start, c_angle_end,
                                c_angle_step, false, c_min_contrast, c_max_contrast, modelId)) {
        std::cerr << "Failed to create template.\n";
        return 1;
    }
    timer.record("训练模型");
#if USE_BINARY_MODEL
    if (!trainer.saveModelFile2Binary(modelId, modelPath)) {
#else
    if (!trainer.saveModelFile2Json(modelId, modelPath)) {
#endif
        std::cerr << "Failed to save model: " << modelPath << '\n';
        return 1;
    }
    timer.record("保存模型");

    if (!pyramidPath.empty()) {
        cv::Mat visualization;
        if (!trainer.drawPyramidFeatures(model_image, modelId, visualization) ||
            !cv::imwrite(pyramidPath, visualization)) {
            std::cerr << "Failed to save pyramid visualization: " << pyramidPath << '\n';
            return 1;
        }
        timer.record("保存金字塔特征图");
    }

    timer.report();

    return 0;
}
