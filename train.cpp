// train.cpp
#include "MakeTemplateV1.h"
#include "Timer.h"

#define USE_BINARY_MODEL 0

int main(int argc, const char* argv[])
{
    Timer timer(TimerMethod::HighResolutionClock);

    // 主函数：加载图像，执行训练和保存模型
    cv::Mat model_image, model_mask;
    if (argc < 2)
    {
        model_image = cv::imread("../assert/m1.png", cv::IMREAD_GRAYSCALE);
        model_mask = cv::Mat(model_image.rows, model_image.cols, CV_8UC1, cv::Scalar(255));
    }
    else
    {
        model_image = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);
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
    const std::string modelPath = "./model.bin";
#else
    const std::string modelPath = "./model.json";
#endif

    // 训练模型
    SM_V1::CreateTemplate trainer;
    T_T::Template::Ptr modelId = std::make_shared<T_T::Template>();
    timer.start();
    trainer.createTemplate(model_image, model_mask, c_pyramid_number, c_angle_start, c_angle_end, c_angle_step, false,
                           c_min_contrast, c_max_contrast, modelId);
    timer.record("训练模型");
#if USE_BINARY_MODEL
    trainer.saveModelFile2Binary(modelId, modelPath);
#else
    trainer.saveModelFile2Json(modelId, modelPath);
#endif
    timer.record("保存模型");

    timer.report();

    return 0;
}


