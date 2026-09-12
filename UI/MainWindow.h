#pragma once

#include "ImageView.h"
#include "FindTemplateV1.h"
#include "MakeTemplateV1.h"
#include "ModelIdNormalization.h"

#include <QFutureWatcher>
#include <QMainWindow>
#include <QTranslator>
#include <QSettings>
#include <QStringList>

#include <opencv2/core.hpp>
#include <functional>
#include <memory>
#include <vector>

class QCheckBox;
class QComboBox;
class QDoubleSpinBox;
class QLabel;
class QLineEdit;
class QProgressBar;
class QScrollArea;
class QSpinBox;
class QTabWidget;
class QTextEdit;
class QPushButton;
class QMenu;
class QAction;

struct TrainResult
{
    bool ok = false;
    QString error;
    std::function<QString()> errorFunc;
    double elapsed_ms = 0.0;
    T_T::Template::Ptr model;
    cv::Mat image;
    QVector<ImageView::ContourPath> contours;
    std::vector<cv::Mat> layerImages;
    std::vector<QVector<ImageView::ContourPath>> layerContours;
};

struct InferResult
{
    bool ok = false;
    QString error;
    std::function<QString()> errorFunc;
    double elapsed_ms = 0.0;
    cv::Mat image;
    std::vector<T_T::MatchResult> matches;
    std::vector<T_T::Template::Ptr> models;
    QStringList modelPaths;
    std::vector<SM_V1::ModelIdAssignment> modelIdAssignments;
    QStringList modelIdLogs;
    I_I::Metric metric = I_I::USE_POLARITY;
    T_T::ScaleSearchCfg scaleCfg;
    int angleStart = -180;
    int angleEnd = 180;
    float minScore = 0.7f;
    float maxOverlap = 0.5f;
};

class MainWindow : public QMainWindow
{
    Q_OBJECT
public:
    explicit MainWindow(QWidget* parent = nullptr);
    ~MainWindow() override;
    void log(const std::function<QString()>& textFunc);
    void log(const QString& text);
    void refreshLogView();

public slots:
    void selectLanguage(bool english);

private slots:
    void chooseTemplateImage();
    void chooseModelOutput();
    void chooseInputImage();
    void chooseModels();
    void chooseResultOutput();
    void train();
    void saveJson();
    void saveBinary();
    void infer();
    void onTrainFinished();
    void onInferFinished();
    void onTrainPyramidLayerChanged(int index);
    void viewPyramidFeatures();

private:
    void buildUi();
    QWidget* buildTrainPanel();
    QWidget* buildInferPanel();
    static QScrollArea* scrollPanel(QWidget* panel, QWidget* parent);
    void setBusy(bool busy, const std::function<QString()>& messageFunc = nullptr);
    void setBusy(bool busy, const QString& message);
    void showError(const std::function<QString()>& textFunc);
    void showError(const QString& text);
    bool validateTraining(std::function<QString()>& errorFunc) const;
    bool validateInference(std::function<QString()>& errorFunc) const;
    void updateTimingLabels();
    void saveModel(bool binary);
    void saveUiState() const;
    void restoreUiState();
    void connectPathSignals();
    void rebuildUiForLanguage();
    static QImage matToImage(const cv::Mat& image);
    QTabWidget* tabs_;
    QTabWidget* parameterTabs_;
    ImageView* trainView_;
    ImageView* inferView_;
    QLineEdit* templatePath_;
    QLineEdit* modelOutputPath_;
    QLineEdit* inputPath_;
    QLineEdit* modelPaths_;
    QLineEdit* resultOutputPath_;
    QSpinBox* templateId_;
    QSpinBox* pyramidLevels_;
    QComboBox* pyramidDisplayLevel_;
    QSpinBox* angleStart_;
    QSpinBox* angleEnd_;
    QDoubleSpinBox* angleStep_;
    QCheckBox* createOtsu_;
    QSpinBox* trainMinContrast_;
    QSpinBox* trainMaxContrast_;
    QSpinBox* roiX_;
    QSpinBox* roiY_;
    QSpinBox* roiW_;
    QSpinBox* roiH_;
    QSpinBox* searchAngleStart_;
    QSpinBox* searchAngleEnd_;
    QDoubleSpinBox* minScore_;
    QSpinBox* numMatches_;
    QDoubleSpinBox* maxOverlap_;
    QSpinBox* searchLevels_;
    QDoubleSpinBox* greediness_;
    QCheckBox* sortByY_;
    QDoubleSpinBox* scaleMin_;
    QDoubleSpinBox* scaleMax_;
    QDoubleSpinBox* scaleStep_;
    QDoubleSpinBox* minVisibleRatio_;
    QSpinBox* searchMinContrast_;
    QComboBox* metric_;
    QCheckBox* subpixel_;
    QCheckBox* simd_;
    QComboBox* edgeMethod_;
    QPushButton* trainButton_;
    QPushButton* pyramidFeaturesButton_;
    QPushButton* inferButton_;
    QPushButton* saveJsonButton_;
    QPushButton* saveBinaryButton_;
    QTextEdit* logView_;
    QProgressBar* progress_;

    cv::Mat templateImage_;
    cv::Mat inputImage_;
    T_T::Template::Ptr model_;
    std::vector<cv::Mat> trainLayerImages_;
    std::vector<QVector<ImageView::ContourPath>> trainLayerContours_;
    QStringList selectedModelPaths_;
    bool modelOutputPathUserSet_;
    bool resultOutputPathUserSet_;

    QFutureWatcher<TrainResult> trainWatcher_;
    QFutureWatcher<InferResult> inferWatcher_;
    QLabel* trainTimingLabel_;
    QLabel* inferTimingLabel_;
    QMenu* languageMenu_;
    QAction* chineseAction_;
    QAction* englishAction_;
    QTranslator translator_;
    bool english_;
    std::vector<std::function<QString()>> logHistory_;
    double lastTrainElapsedMs_ = -1.0;
    double lastInferElapsedMs_ = -1.0;
    cv::Mat lastInferImage_;
    QVector<ImageView::OverlayPath> lastInferOverlays_;
};
