#include "MainWindow.h"
#include "ContourBuilder.h"
#include "MatchOverlay.h"
#include "ModelIdNormalization.h"

#include <QtConcurrent/QtConcurrent>
#include <QCheckBox>
#include <QComboBox>
#include <QDoubleSpinBox>
#include <QDir>
#include <QCoreApplication>
#include <QFile>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QFileDialog>
#include <QFileInfo>
#include <QFormLayout>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QMenu>
#include <QMenuBar>
#include <QAction>
#include <QMessageBox>
#include <QProgressBar>
#include <QPushButton>
#include <QScrollArea>
#include <QSignalBlocker>
#include <QSpinBox>
#include <QSplitter>
#include <QTabWidget>
#include <QTextEdit>
#include <QVBoxLayout>
#include <QApplication>
#include <QSettings>

#include <opencv2/imgproc.hpp>

#include <cmath>
#include <algorithm>
#include <chrono>
#include <functional>
#include <limits>

namespace {

QSpinBox* intBox(int min, int max, int value, QWidget* parent)
{
    QSpinBox* box = new QSpinBox(parent);
    box->setRange(min, max);
    box->setValue(value);
    return box;
}

QDoubleSpinBox* realBox(double min, double max, double step, double value, QWidget* parent)
{
    QDoubleSpinBox* box = new QDoubleSpinBox(parent);
    box->setRange(min, max);
    box->setSingleStep(step);
    box->setDecimals(4);
    box->setValue(value);
    return box;
}

void setPath(QLineEdit* edit, const QString& path)
{
    edit->setText(QDir::toNativeSeparators(path));
}

QString runtimeOutputDir()
{
    return QCoreApplication::applicationDirPath();
}

// UI 输出字段历史上可能包含 UI/results/foo.json 等路径；保留用户明确选择的绝对路径，
// 但将所有相对产物解析到可执行文件旁，使结果文件不依赖进程工作目录。
QString normalizedOutputPath(const QString& candidate,
                             const QString& fallbackName,
                             const QString& defaultExtension,
                             bool forceExtension)
{
    QString value = candidate.trimmed();
    if (value.isEmpty()) value = fallbackName;
    QFileInfo input(value);
    QString base = input.completeBaseName();
    if (base.isEmpty()) base = input.fileName();
    if (base.isEmpty()) base = QStringLiteral("result");

    QString fileName;
    if (forceExtension || input.suffix().isEmpty())
        fileName = base + defaultExtension;
    else
        fileName = input.fileName();

    const QString path = input.isAbsolute()
        ? QDir(input.absolutePath()).absoluteFilePath(fileName)
        : QDir(runtimeOutputDir()).absoluteFilePath(fileName);
    return QDir::cleanPath(path);
}

bool ensureOutputParent(const QString& path, QString* error)
{
    const QString absolute = QFileInfo(path).absoluteFilePath();
    const QString parent = QFileInfo(absolute).absolutePath();
    if (QDir().mkpath(parent)) return true;
    if (error) *error = QObject::tr("无法创建输出目录：%1").arg(parent);
    return false;
}

bool savePyramidPreview(const QString& modelPath,
                        const std::vector<cv::Mat>& layerImages,
                        const std::vector<QVector<ImageView::ContourPath>>& layerContours,
                        QString* error)
{
    try {
        if (modelPath.isEmpty() || layerImages.empty() || layerImages.size() != layerContours.size()) {
            if (error) *error = QObject::tr("金字塔层数据为空或数量不一致");
            return false;
        }
        const int margin = 18, gap = 24;
        int width = margin * 2, height = margin * 2;
        for (const cv::Mat& layer : layerImages) {
            if (layer.empty()) return false;
            width += layer.cols + gap;
            height = std::max(height, layer.rows + 2 * margin + 28);
        }
        cv::Mat canvas(height, width, CV_8UC3, cv::Scalar(18, 18, 18));
        int x = margin;
        constexpr int contourShift = 8;
        for (size_t level = 0; level < layerImages.size(); ++level) {
            cv::Mat tile;
            if (layerImages[level].channels() == 1)
                cv::cvtColor(layerImages[level], tile, cv::COLOR_GRAY2BGR);
            else if (layerImages[level].channels() == 3)
                tile = layerImages[level].clone();
            else {
                if (error) *error = QObject::tr("金字塔图像通道数不支持");
                return false;
            }
            for (const auto& contour : layerContours[level]) {
                if (contour.points.size() < 2) continue;
                std::vector<cv::Point> points;
                points.reserve(contour.points.size());
                bool valid = true;
                for (const QPointF& point : contour.points) {
                    if (!std::isfinite(point.x()) || !std::isfinite(point.y())) {
                        valid = false;
                        break;
                    }
                    points.emplace_back(cvRound(point.x() * (1 << contourShift)),
                                        cvRound(point.y() * (1 << contourShift)));
                }
                if (!valid || points.size() < 2) continue;
                cv::polylines(tile, points, contour.closed, cv::Scalar(0, 255, 0),
                              1, cv::LINE_AA, contourShift);
            }
            tile.copyTo(canvas(cv::Rect(x, margin, tile.cols, tile.rows)));
            cv::putText(canvas, "L" + std::to_string(level) + " " +
                        std::to_string(tile.cols) + "x" + std::to_string(tile.rows),
                        cv::Point(x, margin + tile.rows + 18), cv::FONT_HERSHEY_SIMPLEX,
                        0.42, cv::Scalar(230, 230, 230), 1, cv::LINE_AA);
            x += tile.cols + gap;
        }
        const QFileInfo info(modelPath);
        const QString outputPath = info.path() + QLatin1String("/") +
            info.completeBaseName() + QLatin1String(".pyramid.png");
        if (!ensureOutputParent(outputPath, error)) return false;
        if (!cv::imwrite(outputPath.toStdString(), canvas)) {
            if (error) *error = QObject::tr("OpenCV 未写入文件");
            return false;
        }
        return true;
    } catch (const cv::Exception& ex) {
        if (error) *error = QObject::tr("OpenCV 异常：%1").arg(QString::fromLocal8Bit(ex.what()));
        return false;
    } catch (const std::exception& ex) {
        if (error) *error = QObject::tr("异常：%1").arg(QString::fromLocal8Bit(ex.what()));
        return false;
    }
}

}

MainWindow::MainWindow(QWidget* parent)
    : QMainWindow(parent), tabs_(nullptr), parameterTabs_(nullptr), trainView_(nullptr), inferView_(nullptr),
      templatePath_(nullptr), modelOutputPath_(nullptr),
      inputPath_(nullptr), modelPaths_(nullptr), resultOutputPath_(nullptr),
      templateId_(nullptr), pyramidLevels_(nullptr), pyramidDisplayLevel_(nullptr), angleStart_(nullptr), angleEnd_(nullptr),
      angleStep_(nullptr), createOtsu_(nullptr), trainMinContrast_(nullptr),
      trainMaxContrast_(nullptr), roiX_(nullptr), roiY_(nullptr), roiW_(nullptr), roiH_(nullptr),
      searchAngleStart_(nullptr), searchAngleEnd_(nullptr), minScore_(nullptr),
      numMatches_(nullptr), maxOverlap_(nullptr), searchLevels_(nullptr), greediness_(nullptr),
      sortByY_(nullptr), scaleMin_(nullptr), scaleMax_(nullptr), scaleStep_(nullptr),
      minVisibleRatio_(nullptr), searchMinContrast_(nullptr), metric_(nullptr),
      subpixel_(nullptr), simd_(nullptr), edgeMethod_(nullptr), trainButton_(nullptr), pyramidFeaturesButton_(nullptr), inferButton_(nullptr), saveJsonButton_(nullptr),
      saveBinaryButton_(nullptr), logView_(nullptr), progress_(nullptr),
      modelOutputPathUserSet_(false), resultOutputPathUserSet_(false),
      trainTimingLabel_(nullptr), inferTimingLabel_(nullptr),
      languageMenu_(nullptr), chineseAction_(nullptr), englishAction_(nullptr),
      english_(false)
{
    QSettings settings;
    english_ = settings.value(QStringLiteral("ui/language"), QStringLiteral("zh")).toString() == QStringLiteral("en");
    if (english_)
        translator_.load(QStringLiteral("shape_match_en.qm"), QCoreApplication::applicationDirPath());
    if (english_) qApp->installTranslator(&translator_);
    buildUi();
    QDir().mkpath(runtimeOutputDir());
    connectPathSignals();
    connect(&trainWatcher_, &QFutureWatcher<TrainResult>::finished,
            this, &MainWindow::onTrainFinished);
    connect(&inferWatcher_, &QFutureWatcher<InferResult>::finished,
            this, &MainWindow::onInferFinished);
    setWindowTitle(tr("Shape Match - Qt5 客户端"));
    languageMenu_ = menuBar()->addMenu(tr("语言 / Language"));
    chineseAction_ = languageMenu_->addAction(QStringLiteral("中文"));
    englishAction_ = languageMenu_->addAction(QStringLiteral("English"));
    chineseAction_->setCheckable(true);
    englishAction_->setCheckable(true);
    chineseAction_->setChecked(!english_);
    englishAction_->setChecked(english_);
    connect(chineseAction_, &QAction::triggered, this, [this] { selectLanguage(false); });
    connect(englishAction_, &QAction::triggered, this, [this] { selectLanguage(true); });
    restoreUiState();
    resize(1280, 780);
    log([this]() { return tr("就绪。可在训练/推理页设置参数并执行。"); });
}

MainWindow::~MainWindow()
{
    trainWatcher_.waitForFinished();
    inferWatcher_.waitForFinished();
}

void MainWindow::buildUi()
{
    QWidget* trainPage = new QWidget(this);
    QVBoxLayout* trainLayout = new QVBoxLayout(trainPage);
    QHBoxLayout* trainBar = new QHBoxLayout();
    trainBar->addWidget(new QLabel(tr("训练模板（L0 canonical 边缘叠加）"), trainPage));
    trainTimingLabel_ = new QLabel(tr("纯训练耗时: --"), trainPage);
    trainBar->addWidget(trainTimingLabel_);
    trainBar->addStretch();
    QPushButton* fitTrain = new QPushButton(tr("适应窗口"), trainPage);
    QPushButton* resetTrain = new QPushButton(tr("1:1"), trainPage);
    pyramidFeaturesButton_ = new QPushButton(tr("查看金字塔各层特征"), trainPage);
    pyramidFeaturesButton_->setEnabled(false);
    trainBar->addWidget(fitTrain);
    trainBar->addWidget(resetTrain);
    trainBar->addWidget(pyramidFeaturesButton_);
    connect(pyramidFeaturesButton_, &QPushButton::clicked, this, &MainWindow::viewPyramidFeatures);
    trainLayout->addLayout(trainBar);
    trainView_ = new ImageView(trainPage);
    trainLayout->addWidget(trainView_, 1);

    QWidget* inferPage = new QWidget(this);
    QVBoxLayout* inferLayout = new QVBoxLayout(inferPage);
    QHBoxLayout* resultBar = new QHBoxLayout();
    resultBar->addWidget(new QLabel(tr("推理结果（滚轮缩放，拖拽平移）"), inferPage));
    inferTimingLabel_ = new QLabel(tr("纯推理耗时: --"), inferPage);
    resultBar->addWidget(inferTimingLabel_);
    resultBar->addStretch();
    QPushButton* fitInfer = new QPushButton(tr("适应窗口"), inferPage);
    QPushButton* resetInfer = new QPushButton(tr("1:1"), inferPage);
    resultBar->addWidget(fitInfer);
    resultBar->addWidget(resetInfer);
    inferLayout->addLayout(resultBar);
    inferView_ = new ImageView(inferPage);
    inferLayout->addWidget(inferView_, 1);

    tabs_ = new QTabWidget(this);
    tabs_->addTab(trainPage, tr("训练模板"));
    tabs_->addTab(inferPage, tr("推理结果"));
    connect(fitTrain, &QPushButton::clicked, trainView_, &ImageView::fitImage);
    connect(resetTrain, &QPushButton::clicked, trainView_, &ImageView::resetZoom);
    connect(fitInfer, &QPushButton::clicked, inferView_, &ImageView::fitImage);
    connect(resetInfer, &QPushButton::clicked, inferView_, &ImageView::resetZoom);

    QWidget* panel = new QWidget(this);
    QVBoxLayout* panelLayout = new QVBoxLayout(panel);
    parameterTabs_ = new QTabWidget(panel);
    parameterTabs_->addTab(scrollPanel(buildTrainPanel(), parameterTabs_), tr("训练参数"));
    parameterTabs_->addTab(scrollPanel(buildInferPanel(), parameterTabs_), tr("推理参数"));
    panelLayout->addWidget(parameterTabs_, 1);
    progress_ = new QProgressBar(panel);
    progress_->setRange(0, 0);
    progress_->setVisible(false);
    panelLayout->addWidget(progress_);
    panelLayout->addWidget(new QLabel(tr("状态日志"), panel));
    logView_ = new QTextEdit(panel);
    logView_->setReadOnly(true);
    logView_->setMinimumHeight(130);
    panelLayout->addWidget(logView_, 0);
    QScrollArea* scroll = new QScrollArea(this);
    scroll->setWidgetResizable(true);
    scroll->setWidget(panel);
    scroll->setMinimumWidth(390);

    QSplitter* splitter = new QSplitter(Qt::Horizontal, this);
    splitter->addWidget(tabs_);
    splitter->addWidget(scroll);
    splitter->setStretchFactor(0, 1);
    splitter->setStretchFactor(1, 0);
    setCentralWidget(splitter);

    // 让图像视图和参数页保持同一工作流；阻断相互信号以避免 currentChanged 递归。
    connect(tabs_, &QTabWidget::currentChanged, this, [this](int index) {
        if (parameterTabs_ && parameterTabs_->currentIndex() != index) {
            const QSignalBlocker blocker(parameterTabs_);
            parameterTabs_->setCurrentIndex(index);
        }
    });
    connect(parameterTabs_, &QTabWidget::currentChanged, this, [this](int index) {
        if (tabs_ && tabs_->currentIndex() != index) {
            const QSignalBlocker blocker(tabs_);
            tabs_->setCurrentIndex(index);
        }
    });
}

QScrollArea* MainWindow::scrollPanel(QWidget* panel, QWidget* parent)
{
    QScrollArea* scroll = new QScrollArea(parent);
    scroll->setWidgetResizable(true);
    scroll->setFrameShape(QFrame::NoFrame);
    scroll->setWidget(panel);
    return scroll;
}

QWidget* MainWindow::buildTrainPanel()
{
    QGroupBox* group = new QGroupBox(tr("模板训练参数"), this);
    QFormLayout* form = new QFormLayout(group);
    auto pathRow = [this](QLineEdit*& edit, const QString& buttonText,
                          const std::function<void()>& slot) {
        QWidget* row = new QWidget(this);
        QHBoxLayout* layout = new QHBoxLayout(row);
        layout->setContentsMargins(0, 0, 0, 0);
        edit = new QLineEdit(row);
        QPushButton* button = new QPushButton(buttonText, row);
        layout->addWidget(edit, 1);
        layout->addWidget(button);
        connect(button, &QPushButton::clicked, this, slot);
        return row;
    };
    form->addRow(tr("模板图像"), pathRow(templatePath_, tr("选择"), [this] { chooseTemplateImage(); }));
    templateId_ = intBox(1, 1000000, 1, group);
    form->addRow(tr("Template ID"), templateId_);
    pyramidLevels_ = intBox(-1, 7, -1, group);
    pyramidLevels_->setToolTip(tr("-1=自动，0..7=固定层数"));
    form->addRow(tr("金字塔层数"), pyramidLevels_);
    pyramidDisplayLevel_ = new QComboBox(group);
    pyramidDisplayLevel_->setEnabled(false);
    pyramidDisplayLevel_->setToolTip(tr("训练完成后选择要显示的金字塔层，默认 L0"));
    form->addRow(tr("显示金字塔层"), pyramidDisplayLevel_);
    connect(pyramidDisplayLevel_, static_cast<void (QComboBox::*)(int)>(&QComboBox::currentIndexChanged),
            this, &MainWindow::onTrainPyramidLayerChanged);
    angleStart_ = intBox(-360, 360, -180, group);
    angleEnd_ = intBox(-360, 360, 180, group);
    angleStep_ = realBox(0.01, 360.0, 1.0, 1.0, group);
    form->addRow(tr("角度起始/终止"), angleStart_);
    form->addRow(tr("角度终止（度）"), angleEnd_);
    form->addRow(tr("角度步长（度）"), angleStep_);
    createOtsu_ = new QCheckBox(tr("使用 Otsu 自动阈值"), group);
    form->addRow(QString(), createOtsu_);
    trainMinContrast_ = intBox(0, 255, 25, group);
    trainMaxContrast_ = intBox(0, 255, 100, group);
    form->addRow(tr("最小对比度"), trainMinContrast_);
    form->addRow(tr("最大对比度"), trainMaxContrast_);
    edgeMethod_ = new QComboBox(group);
    edgeMethod_->addItem(tr("Canny + 抛物线亚像素边缘"), static_cast<int>(T_T::EDGE_CURRENT));
    edgeMethod_->addItem(tr("Devernay 亚像素边缘"), static_cast<int>(T_T::EDGE_DEVERNAY));
    edgeMethod_->addItem(tr("Canny 像素级边缘"), static_cast<int>(T_T::EDGE_CANNY_PIXEL));
    form->addRow(tr("边缘特征算法"), edgeMethod_);
    form->addRow(tr("模型输出"), pathRow(modelOutputPath_, tr("选择"), [this] { chooseModelOutput(); }));
    QHBoxLayout* buttons = new QHBoxLayout();
    trainButton_ = new QPushButton(tr("开始训练"), group);
    saveJsonButton_ = new QPushButton(tr("保存 JSON"), group);
    saveBinaryButton_ = new QPushButton(tr("保存 BIN"), group);
    saveJsonButton_->setEnabled(false);
    saveBinaryButton_->setEnabled(false);
    buttons->addWidget(trainButton_);
    buttons->addWidget(saveJsonButton_);
    buttons->addWidget(saveBinaryButton_);
    form->addRow(buttons);
    connect(trainButton_, &QPushButton::clicked, this, &MainWindow::train);
    connect(saveJsonButton_, &QPushButton::clicked, this, &MainWindow::saveJson);
    connect(saveBinaryButton_, &QPushButton::clicked, this, &MainWindow::saveBinary);
    return group;
}

QWidget* MainWindow::buildInferPanel()
{
    QGroupBox* group = new QGroupBox(tr("模板推理参数"), this);
    QFormLayout* form = new QFormLayout(group);
    auto pathRow = [this](QLineEdit*& edit, const QString& buttonText,
                          const std::function<void()>& slot) {
        QWidget* row = new QWidget(this);
        QHBoxLayout* layout = new QHBoxLayout(row);
        layout->setContentsMargins(0, 0, 0, 0);
        edit = new QLineEdit(row);
        QPushButton* button = new QPushButton(buttonText, row);
        layout->addWidget(edit, 1);
        layout->addWidget(button);
        connect(button, &QPushButton::clicked, this, slot);
        return row;
    };
    form->addRow(tr("输入图像"), pathRow(inputPath_, tr("选择"), [this] { chooseInputImage(); }));
    form->addRow(tr("模型（可多选）"), pathRow(modelPaths_, tr("选择"), [this] { chooseModels(); }));
    QWidget* roi = new QWidget(group);
    QHBoxLayout* roiLayout = new QHBoxLayout(roi);
    roiLayout->setContentsMargins(0, 0, 0, 0);
    roiX_ = intBox(0, 100000, 0, roi); roiY_ = intBox(0, 100000, 0, roi);
    roiW_ = intBox(0, 100000, 0, roi); roiH_ = intBox(0, 100000, 0, roi);
    roiLayout->addWidget(new QLabel("x", roi)); roiLayout->addWidget(roiX_);
    roiLayout->addWidget(new QLabel("y", roi)); roiLayout->addWidget(roiY_);
    roiLayout->addWidget(new QLabel("w", roi)); roiLayout->addWidget(roiW_);
    roiLayout->addWidget(new QLabel("h", roi)); roiLayout->addWidget(roiH_);
    form->addRow(tr("ROI（0宽高=全图）"), roi);
    searchAngleStart_ = intBox(-360, 360, -180, group);
    searchAngleEnd_ = intBox(-360, 360, 180, group);
    form->addRow(tr("搜索角度起始"), searchAngleStart_);
    form->addRow(tr("搜索角度终止"), searchAngleEnd_);
    minScore_ = realBox(0.0, 1.0, 0.01, 0.7, group);
    numMatches_ = intBox(1, 10000, 200, group);
    maxOverlap_ = realBox(0.0, 1.0, 0.05, 0.5, group);
    searchLevels_ = intBox(-1, 7, -1, group);
    greediness_ = realBox(0.0, 1.0, 0.05, 0.9, group);
    form->addRow(tr("最小得分"), minScore_);
    form->addRow(tr("最大匹配数"), numMatches_);
    form->addRow(tr("最大重叠"), maxOverlap_);
    form->addRow(tr("搜索层数"), searchLevels_);
    form->addRow(tr("贪婪度"), greediness_);
    sortByY_ = new QCheckBox(tr("结果按 Y 排序"), group);
    sortByY_->setChecked(true);
    form->addRow(QString(), sortByY_);
    scaleMin_ = realBox(0.05, 10.0, 0.05, 1.0, group);
    scaleMax_ = realBox(0.05, 10.0, 0.05, 1.0, group);
    scaleStep_ = realBox(0.001, 5.0, 0.01, 1.0, group);
    minVisibleRatio_ = realBox(0.0, 1.0, 0.05, 1.0, group);
    searchMinContrast_ = intBox(0, 255, 0, group);
    form->addRow(tr("尺度最小/最大/步长"), scaleMin_);
    form->addRow(tr("尺度最大"), scaleMax_);
    form->addRow(tr("尺度步长"), scaleStep_);
    form->addRow(tr("最小可见比例"), minVisibleRatio_);
    form->addRow(tr("搜索最小对比度"), searchMinContrast_);
    metric_ = new QComboBox(group);
    metric_->addItem(tr("使用极性"), I_I::USE_POLARITY);
    metric_->addItem(tr("忽略局部极性"), I_I::IGNORE_LOCAL_POLARITY);
    metric_->addItem(tr("忽略全局极性"), I_I::IGNORE_GLOBAL_POLARITY);
    form->addRow(tr("匹配度量"), metric_);
    subpixel_ = new QCheckBox(tr("启用亚像素精修"), group);
    subpixel_->setToolTip(tr("对最终候选执行亚像素位姿精修"));
    form->addRow(QString(), subpixel_);
    simd_ = new QCheckBox(tr("请求 AVX2 SIMD 加速"), group);
    simd_->setEnabled(SM_V1::SearchTemplate::isSimdAvailable());
    simd_->setToolTip(simd_->isEnabled() ? tr("当前 CPU 支持 AVX2") : tr("当前 CPU 不支持 AVX2，将使用标量路径"));
    form->addRow(QString(), simd_);
    form->addRow(tr("结果图输出"), pathRow(resultOutputPath_, tr("选择"), [this] { chooseResultOutput(); }));
    inferButton_ = new QPushButton(tr("开始推理"), group);
    form->addRow(inferButton_);
    connect(inferButton_, &QPushButton::clicked, this, &MainWindow::infer);
    return group;
}

void MainWindow::chooseTemplateImage()
{
    QDir root(QCoreApplication::applicationDirPath()); root.cdUp(); root.cdUp();
    const QString path = QFileDialog::getOpenFileName(this, tr("选择模板图像"), root.absoluteFilePath("assert"),
                                                       tr("图像 (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)"));
    if (path.isEmpty()) return;
    // 保留源像素用于预览；CreateTemplate 接受 1/3/4 通道 8 位图像，并在内部将
    // 私有副本转换为灰度。
    cv::Mat image = cv::imread(path.toStdString(), cv::IMREAD_UNCHANGED);
    if (image.empty()) {
        showError([this, path]() { return tr("无法读取模板图像：%1").arg(path); });
        return;
    }
    templateImage_ = image;
    setPath(templatePath_, path);
    if (!modelOutputPathUserSet_)
    {
        const QSignalBlocker blocker(modelOutputPath_);
        setPath(modelOutputPath_, QDir(runtimeOutputDir()).absoluteFilePath(
            QFileInfo(path).completeBaseName() + ".json"));
    }
    const int cols = image.cols;
    const int rows = image.rows;
    log([this, path, cols, rows]() {
        return tr("已载入模板图像 %1（%2 x %3）").arg(path).arg(cols).arg(rows);
    });
}

void MainWindow::chooseModelOutput()
{
    const QString initial = normalizedOutputPath(modelOutputPath_->text(),
                                                 QStringLiteral("model.json"),
                                                 QStringLiteral(".json"), false);
    const QString path = QFileDialog::getSaveFileName(this, tr("模型输出路径"), initial,
                                                       tr("JSON (*.json);;Binary (*.bin);;所有文件 (*)"));
    if (!path.isEmpty()) setPath(modelOutputPath_, path);
}

void MainWindow::chooseInputImage()
{
    QDir root(QCoreApplication::applicationDirPath()); root.cdUp(); root.cdUp();
    const QString path = QFileDialog::getOpenFileName(this, tr("选择推理图像"), root.absoluteFilePath("assert"),
                                                       tr("图像 (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)"));
    if (path.isEmpty()) return;
    // 保证搜索像素与加载 IMREAD_GRAYSCALE 的 CLI 推理路径逐字节等价；仅在写入彩色
    // 结果图时将该副本转换回 BGR。
    cv::Mat image = cv::imread(path.toStdString(), cv::IMREAD_GRAYSCALE);
    if (image.empty()) {
        showError([this, path]() { return tr("无法读取输入图像：%1").arg(path); });
        return;
    }
    inputImage_ = image;
    setPath(inputPath_, path);
    if (!resultOutputPathUserSet_) {
        const QSignalBlocker blocker(resultOutputPath_);
        setPath(resultOutputPath_, QDir(runtimeOutputDir()).absoluteFilePath(
            QFileInfo(path).completeBaseName() + ".result.png"));
    }
    const int cols = image.cols;
    const int rows = image.rows;
    log([this, path, cols, rows]() {
        return tr("已载入推理图像 %1（%2 x %3）").arg(path).arg(cols).arg(rows);
    });
}

void MainWindow::chooseModels()
{
    const QStringList paths = QFileDialog::getOpenFileNames(this, tr("选择模板模型"), runtimeOutputDir(),
                                                             tr("模型 (*.json *.bin);;所有文件 (*)"));
    if (paths.isEmpty()) return;
    selectedModelPaths_ = paths;
    modelPaths_->setText(paths.join(QLatin1String("; ")));
    const int count = paths.size();
    log([this, count]() {
        return tr("已选择 %1 个模型").arg(count);
    });
}

void MainWindow::chooseResultOutput()
{
    const QString initial = normalizedOutputPath(resultOutputPath_->text(),
                                                 QStringLiteral("result.png"),
                                                 QStringLiteral(".png"), false);
    const QString path = QFileDialog::getSaveFileName(this, tr("结果图输出路径"), initial,
                                                       tr("图像 (*.png *.jpg *.bmp);;所有文件 (*)"));
    if (!path.isEmpty()) setPath(resultOutputPath_, path);
}

bool MainWindow::validateTraining(std::function<QString()>& errorFunc) const
{
    errorFunc = nullptr;
    if (templateImage_.empty()) errorFunc = [this]() { return tr("请先选择模板图像"); };
    else if (templateId_->value() <= 0) errorFunc = [this]() { return tr("Template ID 必须为正数"); };
    else if (angleEnd_->value() < angleStart_->value()) errorFunc = [this]() { return tr("训练角度终止必须不小于起始"); };
    else if (angleStep_->value() <= 0) errorFunc = [this]() { return tr("角度步长必须大于 0"); };
    else if (!createOtsu_->isChecked() && trainMaxContrast_->value() < trainMinContrast_->value())
        errorFunc = [this]() { return tr("最大对比度必须不小于最小对比度"); };
    return errorFunc == nullptr;
}

bool MainWindow::validateInference(std::function<QString()>& errorFunc) const
{
    errorFunc = nullptr;
    if (inputImage_.empty()) errorFunc = [this]() { return tr("请先选择输入图像"); };
    else if (selectedModelPaths_.isEmpty() && !model_) errorFunc = [this]() { return tr("请至少选择一个模型文件"); };
    else if (searchAngleEnd_->value() < searchAngleStart_->value()) errorFunc = [this]() { return tr("搜索角度终止必须不小于起始"); };
    else if (minScore_->value() < 0 || minScore_->value() > 1) errorFunc = [this]() { return tr("最小得分必须在 0..1"); };
    else if (scaleMax_->value() < scaleMin_->value() || scaleStep_->value() <= 0)
        errorFunc = [this]() { return tr("尺度范围或步长无效"); };
    else if (minVisibleRatio_->value() < 0 || minVisibleRatio_->value() > 1)
        errorFunc = [this]() { return tr("最小可见比例必须在 0..1"); };
    else if (roiW_->value() < 0 || roiH_->value() < 0) errorFunc = [this]() { return tr("ROI 尺寸无效"); };
    else if ((roiW_->value() > 0 || roiH_->value() > 0) &&
             (roiW_->value() <= 0 || roiH_->value() <= 0)) errorFunc = [this]() { return tr("ROI 宽高必须同时为正数"); };
    return errorFunc == nullptr;
}

void MainWindow::train()
{
    std::function<QString()> errorFunc;
    if (!validateTraining(errorFunc)) { showError(errorFunc); return; }
    if (trainWatcher_.isRunning() || inferWatcher_.isRunning()) return;
    model_.reset();
    trainLayerImages_.clear();
    trainLayerContours_.clear();
    pyramidFeaturesButton_->setEnabled(false);
    {
        const QSignalBlocker blocker(pyramidDisplayLevel_);
        pyramidDisplayLevel_->clear();
        pyramidDisplayLevel_->setEnabled(false);
    }
    trainView_->clearContours();
    const cv::Mat image = templateImage_.clone();
    const int id = templateId_->value();
    const int levels = pyramidLevels_->value();
    const int start = angleStart_->value();
    const int end = angleEnd_->value();
    const double step = angleStep_->value();
    const bool otsu = createOtsu_->isChecked();
    const int minContrast = trainMinContrast_->value();
    const int maxContrast = trainMaxContrast_->value();
    const T_T::EdgeMethod edgeMethod = static_cast<T_T::EdgeMethod>(edgeMethod_->currentData().toInt());
    lastTrainElapsedMs_ = -1.0;
    updateTimingLabels();
    log([this]() { return tr("纯训练耗时: --（仅统计 CreateTemplate::createTemplate）"); });
    setBusy(true, [this]() { return tr("正在训练模板…"); });
    trainWatcher_.setFuture(QtConcurrent::run([image, id, levels, start, end, step, otsu,
                                                minContrast, maxContrast, edgeMethod]() {
        TrainResult result;
        cv::Mat mask(image.rows, image.cols, CV_8UC1, cv::Scalar(255));
        SM_V1::CreateTemplate trainer;
        result.model = std::make_shared<T_T::Template>();
        result.model->template_cfg.id = id;
        const auto trainBegin = std::chrono::steady_clock::now();
        const bool created = trainer.createTemplate(image, mask, levels, start, end, step, otsu,
                                                    minContrast, maxContrast, result.model, edgeMethod);
        result.elapsed_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - trainBegin).count();
        if (!created) {
            result.errorFunc = []() { return QObject::tr("核心库创建模板失败"); };
            result.model.reset();
            return result;
        }
        // 重现核心金字塔尺寸（每次 pyrDown 后取整数半尺寸）；交给 Qt 前保持每个
        // 副本独立，避免工作线程 cv::Mat 存储与 GUI 共享。
        const int configuredLayers = std::max(1, result.model->template_cfg.num_levels + 1);
        const int layerCount = std::min(configuredLayers,
                                       static_cast<int>(result.model->templates.size()));
        cv::Mat layer = image.clone();
        result.layerImages.reserve(layerCount);
        result.layerContours.reserve(layerCount);
        for (int level = 0; level < layerCount; ++level) {
            if (level > 0) {
                cv::Mat next;
                cv::pyrDown(layer, next, cv::Size(layer.cols / 2, layer.rows / 2));
                if (next.empty()) break;
                layer = next;
            }
            result.layerImages.push_back(layer.clone());
            result.layerContours.push_back(
                ContourBuilder::buildTemplateContours(result.layerImages.back(), result.model, level));
        }
        if (result.layerImages.empty()) {
            result.errorFunc = []() { return QObject::tr("模板创建成功，但无法生成金字塔可视化"); };
            result.model.reset();
            return result;
        }
        result.image = result.layerImages.front().clone();
        result.contours = result.layerContours.front();
        if (result.image.empty()) {
            result.errorFunc = []() { return QObject::tr("模板创建成功，但无法生成模板可视化"); };
            result.model.reset();
            return result;
        }
        result.ok = true;
        return result;
    }));
}

void MainWindow::saveModel(bool binary)
{
    if (!model_) {
        showError([this]() { return tr("请先完成训练"); });
        return;
    }
    QString path = modelOutputPath_->text().trimmed();
    if (path.isEmpty()) {
        const QString initial = QDir(runtimeOutputDir()).absoluteFilePath(
            QStringLiteral("model") + (binary ? QStringLiteral(".bin") : QStringLiteral(".json")));
        path = QFileDialog::getSaveFileName(this, tr("保存模型"), initial,
                                            binary ? tr("Binary (*.bin)") : tr("JSON (*.json)"));
        if (path.isEmpty()) return;
    }
    const QString fallback = QStringLiteral("model") +
        (binary ? QStringLiteral(".bin") : QStringLiteral(".json"));
    path = normalizedOutputPath(path, fallback, binary ? QStringLiteral(".bin") : QStringLiteral(".json"), true);
    {
        const QSignalBlocker blocker(modelOutputPath_);
        setPath(modelOutputPath_, path);
    }

    QString outputError;
    if (!ensureOutputParent(path, &outputError)) {
        log([this, path, outputError]() {
            return tr("保存模型失败：%1（%2）").arg(path, outputError);
        });
        showError([this, path]() {
            return tr("保存模型失败：%1").arg(path);
        });
        return;
    }

    bool ok = false;
    QString exceptionMsg;
    bool isOpenCvEx = false;
    try {
        SM_V1::CreateTemplate trainer;
        ok = binary ? trainer.saveModelFile2Binary(model_, path.toStdString())
                    : trainer.saveModelFile2Json(model_, path.toStdString());
    } catch (const cv::Exception& ex) {
        isOpenCvEx = true;
        exceptionMsg = QString::fromLocal8Bit(ex.what());
    } catch (const std::exception& ex) {
        exceptionMsg = QString::fromLocal8Bit(ex.what());
    }
    if (!ok) {
        log([this, path, isOpenCvEx, exceptionMsg]() {
            QString detail;
            if (isOpenCvEx) detail = tr("OpenCV 异常：%1").arg(exceptionMsg);
            else if (!exceptionMsg.isEmpty()) detail = tr("异常：%1").arg(exceptionMsg);
            else detail = tr("模型写入接口返回失败");
            return tr("保存模型失败：%1（%2）").arg(path, detail);
        });
        showError([this, path]() {
            return tr("保存模型失败：%1").arg(path);
        });
        return;
    }

    QString pyramidError;
    const bool pyramidOk = savePyramidPreview(path, trainLayerImages_, trainLayerContours_, &pyramidError);
    if (!pyramidOk) {
        const QString base = QFileInfo(path).completeBaseName();
        log([this, base, pyramidError]() {
            return tr("警告：金字塔特征图写入失败：%1.pyramid.png（%2）").arg(base, pyramidError);
        });
    } else {
        const QString pyrPath = QFileInfo(path).absolutePath() + QLatin1String("/") +
                                QFileInfo(path).completeBaseName() + QLatin1String(".pyramid.png");
        log([this, pyrPath]() {
            return tr("金字塔特征图已保存：%1").arg(pyrPath);
        });
    }
    log([this, path]() {
        return tr("模型已保存：%1").arg(path);
    });
}

void MainWindow::saveJson() { saveModel(false); }
void MainWindow::saveBinary() { saveModel(true); }

void MainWindow::infer()
{
    std::function<QString()> errorFunc;
    if (!validateInference(errorFunc)) { showError(errorFunc); return; }
    if (trainWatcher_.isRunning() || inferWatcher_.isRunning()) return;
    const cv::Mat image = inputImage_.clone();
    const QStringList paths = selectedModelPaths_;
    const T_T::Template::Ptr fallback = model_;
    const int aStart = searchAngleStart_->value();
    const int aEnd = searchAngleEnd_->value();
    const float score = static_cast<float>(minScore_->value());
    const int count = numMatches_->value();
    const float overlap = static_cast<float>(maxOverlap_->value());
    const int levels = searchLevels_->value();
    const float greed = static_cast<float>(greediness_->value());
    const bool sortY = sortByY_->isChecked();
    T_T::ScaleSearchCfg scale(scaleMin_->value(), scaleMax_->value(), scaleStep_->value(),
                              minVisibleRatio_->value(), subpixel_->isChecked(),
                              searchMinContrast_->value(), metric_->currentData().toInt(),
                              simd_->isChecked());
    const ROI searchRoi = (roiW_->value() > 0 && roiH_->value() > 0)
        ? ROI(cv::Rect(roiX_->value(), roiY_->value(), roiW_->value(), roiH_->value())) : ROI();
    lastInferElapsedMs_ = -1.0;
    updateTimingLabels();
    log([this]() { return tr("纯推理耗时: --（仅统计 SearchTemplate::searchTemplate）"); });
    const bool subpix = scale.subpixel_refine;
    const bool simdReq = scale.use_simd;
    const bool simdAvail = SM_V1::SearchTemplate::isSimdAvailable();
    log([this, subpix, simdReq, simdAvail]() {
        return tr("推理配置：亚像素=%1，SIMD请求=%2（AVX2能力=%3）。")
            .arg(subpix ? tr("开") : tr("关"))
            .arg(simdReq ? tr("开") : tr("关"))
            .arg(simdAvail ? tr("有") : tr("无"));
    });
    setBusy(true, [this]() { return tr("正在推理…"); });
    inferWatcher_.setFuture(QtConcurrent::run([image, paths, fallback, aStart, aEnd, score, count,
                                                overlap, levels, greed, sortY, scale, searchRoi]() {
        InferResult result;
        SM_V1::SearchTemplate matcher;
        std::vector<T_T::Template::Ptr> models;
        if (paths.isEmpty() && fallback) models.push_back(fallback);
        for (const QString& path : paths) {
            T_T::Template::Ptr loaded;
            const std::string p = path.toStdString();
            if (path.endsWith(QLatin1String(".bin"), Qt::CaseInsensitive))
                loaded = matcher.loadModelFileFromBinary(p);
            else loaded = matcher.loadModelFileFromJson(p);
            if (!loaded) {
                result.errorFunc = [path]() { return QObject::tr("无法加载模型：%1").arg(path); };
                return result;
            }
            models.push_back(loaded);
        }
        if (models.empty()) {
            result.errorFunc = []() { return QObject::tr("没有可用模型"); };
            return result;
        }
        std::vector<SM_V1::ModelIdAssignment> assignments;
        std::string normalizationError;
        if (!SM_V1::normalizeTemplateIds(models, assignments, &normalizationError)) {
            const QString normErr = QString::fromStdString(normalizationError);
            result.errorFunc = [normErr]() {
                return QObject::tr("模型ID规范化失败：%1").arg(normErr);
            };
            return result;
        }
        result.modelIdAssignments = assignments;
        std::vector<T_T::MatchResult> matches;
        const auto inferBegin = std::chrono::steady_clock::now();
        const bool ok = matcher.searchTemplate(image, cv::Mat(), searchRoi, models,
                                                // 历史公开 API 将此字段称为 angle_extent，但实现和 CLI 语义将其作为绝对终止角。
                                                aStart, aEnd, score, count, overlap,
                                                levels, greed, sortY, scale, matches);
        result.elapsed_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - inferBegin).count();
        if (!ok) {
            result.errorFunc = []() { return QObject::tr("核心库匹配失败"); };
            return result;
        }
        result.image = image.clone();
        result.matches = matches;
        result.models = models;
        result.modelPaths = paths;
        result.metric = static_cast<I_I::Metric>(scale.metric);
        result.scaleCfg = scale;
        result.angleStart = aStart;
        result.angleEnd = aEnd;
        result.minScore = score;
        result.maxOverlap = overlap;
        result.ok = true;
        return result;
    }));
}

void MainWindow::onTrainFinished()
{
    const TrainResult result = trainWatcher_.result();
    setBusy(false);
    lastTrainElapsedMs_ = result.elapsed_ms;
    updateTimingLabels();
    const double trainMs = result.elapsed_ms;
    log([this, trainMs]() {
        return tr("纯训练耗时: %1 ms").arg(trainMs, 0, 'f', 3);
    });
    if (!result.ok) {
        if (result.errorFunc) showError(result.errorFunc);
        else showError(result.error);
        return;
    }
    model_ = result.model;
    trainLayerImages_ = result.layerImages;
    trainLayerContours_ = result.layerContours;
    {
        const QSignalBlocker blocker(pyramidDisplayLevel_);
        pyramidDisplayLevel_->clear();
        for (int level = 0; level < static_cast<int>(trainLayerImages_.size()); ++level) {
            const int featureCount = (level < static_cast<int>(model_->templates.size()) &&
                                       model_->templates[level] &&
                                       !model_->templates[level]->shape_angle.empty() &&
                                       model_->templates[level]->shape_angle[0])
                ? static_cast<int>(model_->templates[level]->shape_angle[0]->shape_point.size()) : 0;
            const cv::Mat& layer = trainLayerImages_[level];
            pyramidDisplayLevel_->addItem(
                tr("L%1 (%2×%3, %4 特征)").arg(level).arg(layer.cols).arg(layer.rows).arg(featureCount),
                level);
        }
        pyramidDisplayLevel_->setCurrentIndex(0);
        pyramidDisplayLevel_->setEnabled(!trainLayerImages_.empty());
    }
    trainView_->setImage(matToImage(trainLayerImages_.front()));
    trainView_->setContours(trainLayerContours_.front());
    tabs_->setCurrentIndex(0);
    saveJsonButton_->setEnabled(true);
    saveBinaryButton_->setEnabled(true);
    pyramidFeaturesButton_->setEnabled(true);
    const T_T::EdgeMethod edgeMethod = model_->template_cfg.edge_method;
    const size_t layerCount = trainLayerImages_.size();
    log([this, layerCount, edgeMethod]() {
        const QString edgeName = edgeMethod == T_T::EDGE_DEVERNAY
            ? tr("Devernay 亚像素")
            : (edgeMethod == T_T::EDGE_CANNY_PIXEL
                ? tr("Canny 像素级") : tr("Canny + 抛物线亚像素"));
        return tr("训练完成：默认显示 L0 canonical 轮廓（共 %1 层，边缘算法=%2）。")
            .arg(layerCount).arg(edgeName);
    });
}

void MainWindow::onTrainPyramidLayerChanged(int index)
{
    if (index < 0 || index >= static_cast<int>(trainLayerImages_.size()) ||
        index >= static_cast<int>(trainLayerContours_.size())) return;
    trainView_->setImage(matToImage(trainLayerImages_[index]));
    trainView_->setContours(trainLayerContours_[index]);
    const int cols = trainLayerImages_[index].cols;
    const int rows = trainLayerImages_[index].rows;
    const int contourCount = trainLayerContours_[index].size();
    log([this, index, cols, rows, contourCount]() {
        return tr("训练预览切换至 L%1（%2×%3，%4 个特征）。")
            .arg(index).arg(cols).arg(rows).arg(contourCount);
    });
}

void MainWindow::viewPyramidFeatures()
{
    if (!model_ || templateImage_.empty()) return;
    const int margin = 18, gap = 24;
    int width = margin * 2, height = margin * 2;
    for (const auto& layer : trainLayerImages_) { width += layer.cols + gap; height = std::max(height, layer.rows + 2 * margin + 30); }
    if (trainLayerImages_.empty()) return;
    cv::Mat visualization(height, width, CV_8UC3, cv::Scalar(18,18,18));
    QVector<ImageView::OverlayPath> overlays;
    int x = margin;
    for (size_t level = 0; level < trainLayerImages_.size(); ++level) {
        cv::Mat tile;
        const cv::Mat& src = trainLayerImages_[level];
        if (src.channels() == 1) cv::cvtColor(src, tile, cv::COLOR_GRAY2BGR); else tile = src.clone();
        const auto& contours = trainLayerContours_[level];
        for (const auto& contour : contours) {
            if (contour.points.size() < 2) continue;
            ImageView::OverlayPath path = contour;
            for (QPointF& p : path.points) p += QPointF(x, margin);
            path.color = QColor(0, 255, 0); path.width = 1.0; path.cosmetic = true;
            overlays.push_back(path);
        }
        tile.copyTo(visualization(cv::Rect(x, margin, tile.cols, tile.rows)));
        cv::putText(visualization, "L" + std::to_string(level) + " " + std::to_string(tile.cols) + "x" + std::to_string(tile.rows),
                    cv::Point(x, margin + tile.rows + 18), cv::FONT_HERSHEY_SIMPLEX, 0.42, cv::Scalar(230,230,230), 1, cv::LINE_AA);
        x += tile.cols + gap;
    }
    trainView_->setImage(matToImage(visualization));
    trainView_->setOverlays(overlays);
    log([this]() { return tr("已显示金字塔各层特征综合图"); });
}

void MainWindow::onInferFinished()
{
    const InferResult result = inferWatcher_.result();
    setBusy(false);
    lastInferElapsedMs_ = result.elapsed_ms;
    updateTimingLabels();
    const double inferMs = result.elapsed_ms;
    log([this, inferMs]() {
        return tr("纯推理耗时: %1 ms").arg(inferMs, 0, 'f', 3);
    });
    if (!result.ok) {
        if (result.errorFunc) showError(result.errorFunc);
        else showError(result.error);
        return;
    }
    for (size_t i = 0; i < result.modelIdAssignments.size(); ++i) {
        const auto& assign = result.modelIdAssignments[i];
        const QString source = i < static_cast<size_t>(result.modelPaths.size())
            ? result.modelPaths[static_cast<int>(i)] : QString();
        const int origId = assign.original_id;
        const int runId = assign.runtime_id;
        log([this, origId, runId, source]() {
            const QString src = source.isEmpty() ? tr("当前训练模型") : source;
            return tr("模型ID映射：原ID=%1 -> 运行时ID=%2，文件=%3")
                .arg(origId).arg(runId).arg(src);
        });
    }
    lastInferImage_ = result.image.clone();
    lastInferOverlays_ = MatchOverlay::buildMatchOverlays(result.image, result.matches,
                                                          result.models, result.metric);
    inferView_->setImage(matToImage(lastInferImage_));
    inferView_->setOverlays(lastInferOverlays_);
    tabs_->setCurrentIndex(1);
    const QString resultBase = inputPath_->text().isEmpty()
        ? QStringLiteral("result") : QFileInfo(inputPath_->text()).completeBaseName();
    const QString output = normalizedOutputPath(resultOutputPath_->text(),
                                                resultBase + QStringLiteral(".result.png"),
                                                QStringLiteral(".png"), false);
    {
        const QSignalBlocker blocker(resultOutputPath_);
        setPath(resultOutputPath_, output);
    }
    QString outputError;
    if (!ensureOutputParent(output, &outputError)) {
        log([this, output, outputError]() {
            return tr("结果图保存失败：%1（%2）").arg(output, outputError);
        });
    }
    cv::Mat rendered = result.image.clone();
    if (rendered.channels() == 1)
        cv::cvtColor(rendered, rendered, cv::COLOR_GRAY2BGR);
    SM_V1::SearchTemplate renderer;
    renderer.drawMatchResults(rendered, result.matches, result.models);
    if (outputError.isEmpty()) {
        try {
            if (!cv::imwrite(output.toStdString(), rendered)) {
                log([this, output]() {
                    return tr("结果图写入失败：%1（OpenCV 未写入文件）").arg(output);
                });
            } else {
                log([this, output]() {
                    return tr("结果图已保存：%1").arg(output);
                });
            }
        } catch (const cv::Exception& ex) {
            const QString what = QString::fromLocal8Bit(ex.what());
            log([this, output, what]() {
                return tr("结果图写入失败：%1（OpenCV 异常：%2）").arg(output, what);
            });
        } catch (const std::exception& ex) {
            const QString what = QString::fromLocal8Bit(ex.what());
            log([this, output, what]() {
                return tr("结果图写入失败：%1（异常：%2）").arg(output, what);
            });
        }
    }
    if (outputError.isEmpty()) {
        QFileInfo outputInfo(output);
        const QString jsonPath = outputInfo.path() + QLatin1String("/") +
            outputInfo.completeBaseName() + QLatin1String(".json");
        QJsonObject root;
        QJsonObject imageInfo;
        imageInfo["path"] = inputPath_->text();
        imageInfo["width"] = result.image.cols;
        imageInfo["height"] = result.image.rows;
        root["image"] = imageInfo;
        QJsonObject searchConfig;
        searchConfig["scale_min"] = result.scaleCfg.scale_min;
        searchConfig["scale_max"] = result.scaleCfg.scale_max;
        searchConfig["scale_step"] = result.scaleCfg.scale_step;
        searchConfig["angle_start"] = result.angleStart;
        searchConfig["angle_end"] = result.angleEnd;
        searchConfig["min_score"] = result.minScore;
        searchConfig["max_overlap"] = result.maxOverlap;
        searchConfig["min_visible_ratio"] = result.scaleCfg.min_visible_ratio;
        searchConfig["min_contrast"] = result.scaleCfg.min_contrast;
        searchConfig["metric"] = result.metric == I_I::USE_POLARITY ? "use-polarity" :
            (result.metric == I_I::IGNORE_LOCAL_POLARITY ? "ignore-local-polarity" : "ignore-global-polarity");
        searchConfig["subpixel_refine"] = result.scaleCfg.subpixel_refine;
        searchConfig["simd_requested"] = result.scaleCfg.use_simd;
        searchConfig["simd_actual"] = result.scaleCfg.use_simd && SM_V1::SearchTemplate::isSimdAvailable();
        root["search_config"] = searchConfig;
        QJsonObject summary;
        summary["total_matches"] = static_cast<int>(result.matches.size());
        root["summary"] = summary;
        QJsonArray arr;
        for (size_t n = 0; n < result.matches.size(); ++n) {
            const auto& m = result.matches[n];
            QJsonObject o;
            o["index"] = static_cast<int>(n + 1);
            o["template_id"] = m.template_id;
            o["score"] = m.score;
            o["scale"] = m.scale;
            QJsonObject pose;
            pose["x"] = m.pose.x; pose["y"] = m.pose.y; pose["angle"] = m.pose.angle;
            o["pose"] = pose;
            o["visible_ratio"] = m.visible_ratio;
            o["matched_ratio"] = m.matched_ratio;
            arr.append(o);
        }
        root["results"] = arr;
        if (!ensureOutputParent(jsonPath, &outputError)) {
            log([this, jsonPath, outputError]() {
                return tr("结果 JSON 保存失败：%1（%2）").arg(jsonPath, outputError);
            });
        } else {
            QFile jf(jsonPath);
            if (!jf.open(QIODevice::WriteOnly)) {
                const QString errStr = jf.errorString();
                log([this, jsonPath, errStr]() {
                    return tr("结果 JSON 保存失败：%1（%2）").arg(jsonPath, errStr);
                });
            } else if (jf.write(QJsonDocument(root).toJson(QJsonDocument::Indented)) < 0) {
                const QString errStr = jf.errorString();
                log([this, jsonPath, errStr]() {
                    return tr("结果 JSON 写入失败：%1（%2）").arg(jsonPath, errStr);
                });
            } else {
                const QString absPath = QFileInfo(jsonPath).absoluteFilePath();
                log([this, absPath]() {
                    return tr("结果 JSON 已保存：%1").arg(absPath);
                });
            }
        }
    }
    const size_t matchCount = result.matches.size();
    log([this, matchCount]() {
        return tr("推理完成：匹配 %1 个，结果图已显示。").arg(matchCount);
    });
    QString equivalent = QStringLiteral("inference \"%1\"").arg(inputPath_->text());
    for (const QString& modelPath : result.modelPaths)
        equivalent += QStringLiteral(" \"%1\"").arg(modelPath);
    equivalent += QStringLiteral(" --min-score %1 --max-overlap %2 --angle-start %3 --angle-end %4"
                                 " --scale-min %5 --scale-max %6 --scale-step %7 --min-visible-ratio %8"
                                 " --min-contrast %9 --metric %10%11%12 --output \"%13\"")
        .arg(result.minScore, 0, 'f', 4).arg(result.maxOverlap, 0, 'f', 4)
        .arg(result.angleStart).arg(result.angleEnd)
        .arg(result.scaleCfg.scale_min, 0, 'f', 4).arg(result.scaleCfg.scale_max, 0, 'f', 4)
        .arg(result.scaleCfg.scale_step, 0, 'f', 4).arg(result.scaleCfg.min_visible_ratio, 0, 'f', 4)
        .arg(result.scaleCfg.min_contrast)
        .arg(result.metric == I_I::USE_POLARITY ? QStringLiteral("use-polarity") :
             (result.metric == I_I::IGNORE_LOCAL_POLARITY ? QStringLiteral("ignore-local-polarity") :
              QStringLiteral("ignore-global-polarity")))
        .arg(result.scaleCfg.subpixel_refine ? QStringLiteral(" --subpixel") : QString())
        .arg(result.scaleCfg.use_simd ? QStringLiteral(" --simd") : QString())
        .arg(output);
    log([this, equivalent]() {
        return tr("等价命令：%1").arg(equivalent);
    });
    for (size_t i = 0; i < result.matches.size(); ++i) {
        const T_T::MatchResult m = result.matches[i];
        const size_t idx = i + 1;
        log([this, idx, m]() {
            return tr("#%1 id=%2 score=%3 scale=%4 pose=(%5, %6, %7) visible=%8")
                .arg(idx).arg(m.template_id).arg(m.score, 0, 'f', 4).arg(m.scale, 0, 'f', 4)
                .arg(m.pose.x, 0, 'f', 3).arg(m.pose.y, 0, 'f', 3).arg(m.pose.angle, 0, 'f', 3)
                .arg(m.visible_ratio, 0, 'f', 3);
        });
    }
}

void MainWindow::connectPathSignals()
{
    connect(modelOutputPath_, &QLineEdit::textChanged, this, [this](const QString& text) {
        modelOutputPathUserSet_ = !text.trimmed().isEmpty();
    });
    connect(resultOutputPath_, &QLineEdit::textChanged, this, [this](const QString& text) {
        resultOutputPathUserSet_ = !text.trimmed().isEmpty();
    });
}

void MainWindow::saveUiState() const
{
    QSettings settings;
    settings.beginGroup(QStringLiteral("ui/parameters"));
    auto text = [&settings](const char* key, const QLineEdit* widget) {
        settings.setValue(QLatin1String(key), widget ? widget->text() : QString());
    };
    auto integer = [&settings](const char* key, const QSpinBox* widget) {
        if (widget) settings.setValue(QLatin1String(key), widget->value());
    };
    auto real = [&settings](const char* key, const QDoubleSpinBox* widget) {
        if (widget) settings.setValue(QLatin1String(key), widget->value());
    };
    auto check = [&settings](const char* key, const QCheckBox* widget) {
        if (widget) settings.setValue(QLatin1String(key), widget->isChecked());
    };
    text("templatePath", templatePath_); text("modelOutputPath", modelOutputPath_);
    text("inputPath", inputPath_); text("modelPaths", modelPaths_);
    text("resultOutputPath", resultOutputPath_);
    integer("templateId", templateId_); integer("pyramidLevels", pyramidLevels_);
    if (pyramidDisplayLevel_) settings.setValue(QStringLiteral("pyramidDisplayLevel"), pyramidDisplayLevel_->currentIndex());
    integer("angleStart", angleStart_); integer("angleEnd", angleEnd_);
    real("angleStep", angleStep_); check("createOtsu", createOtsu_);
    integer("trainMinContrast", trainMinContrast_); integer("trainMaxContrast", trainMaxContrast_);
    integer("roiX", roiX_); integer("roiY", roiY_); integer("roiW", roiW_); integer("roiH", roiH_);
    integer("searchAngleStart", searchAngleStart_); integer("searchAngleEnd", searchAngleEnd_);
    real("minScore", minScore_); integer("numMatches", numMatches_); real("maxOverlap", maxOverlap_);
    integer("searchLevels", searchLevels_); real("greediness", greediness_); check("sortByY", sortByY_);
    real("scaleMin", scaleMin_); real("scaleMax", scaleMax_); real("scaleStep", scaleStep_);
    real("minVisibleRatio", minVisibleRatio_); integer("searchMinContrast", searchMinContrast_);
    if (metric_) settings.setValue(QStringLiteral("metric"), metric_->currentData());
    check("subpixel", subpixel_); check("simd", simd_);
    if (edgeMethod_) settings.setValue(QStringLiteral("edgeMethod"), edgeMethod_->currentData());
    settings.endGroup();
}

void MainWindow::restoreUiState()
{
    QSettings settings;
    settings.beginGroup(QStringLiteral("ui/parameters"));
    auto text = [&settings](const char* key, QLineEdit* widget) {
        if (widget && settings.contains(QLatin1String(key))) widget->setText(settings.value(QLatin1String(key)).toString());
    };
    auto integer = [&settings](const char* key, QSpinBox* widget) {
        if (widget && settings.contains(QLatin1String(key))) widget->setValue(settings.value(QLatin1String(key)).toInt());
    };
    auto real = [&settings](const char* key, QDoubleSpinBox* widget) {
        if (widget && settings.contains(QLatin1String(key))) widget->setValue(settings.value(QLatin1String(key)).toDouble());
    };
    auto check = [&settings](const char* key, QCheckBox* widget) {
        if (widget && settings.contains(QLatin1String(key))) widget->setChecked(settings.value(QLatin1String(key)).toBool());
    };
    text("templatePath", templatePath_); text("modelOutputPath", modelOutputPath_);
    text("inputPath", inputPath_); text("modelPaths", modelPaths_);
    text("resultOutputPath", resultOutputPath_);
    integer("templateId", templateId_); integer("pyramidLevels", pyramidLevels_);
    if (pyramidDisplayLevel_ && settings.contains(QStringLiteral("pyramidDisplayLevel")))
        pyramidDisplayLevel_->setCurrentIndex(settings.value(QStringLiteral("pyramidDisplayLevel")).toInt());
    integer("angleStart", angleStart_); integer("angleEnd", angleEnd_);
    real("angleStep", angleStep_); check("createOtsu", createOtsu_);
    integer("trainMinContrast", trainMinContrast_); integer("trainMaxContrast", trainMaxContrast_);
    integer("roiX", roiX_); integer("roiY", roiY_); integer("roiW", roiW_); integer("roiH", roiH_);
    integer("searchAngleStart", searchAngleStart_); integer("searchAngleEnd", searchAngleEnd_);
    real("minScore", minScore_); integer("numMatches", numMatches_); real("maxOverlap", maxOverlap_);
    integer("searchLevels", searchLevels_); real("greediness", greediness_); check("sortByY", sortByY_);
    real("scaleMin", scaleMin_); real("scaleMax", scaleMax_); real("scaleStep", scaleStep_);
    real("minVisibleRatio", minVisibleRatio_); integer("searchMinContrast", searchMinContrast_);
    auto combo = [&settings](const char* key, QComboBox* widget) {
        if (!widget || !settings.contains(QLatin1String(key))) return;
        const int index = widget->findData(settings.value(QLatin1String(key)));
        if (index >= 0) widget->setCurrentIndex(index);
    };
    combo("metric", metric_); check("subpixel", subpixel_); check("simd", simd_);
    combo("edgeMethod", edgeMethod_);
    settings.endGroup();
    selectedModelPaths_ = modelPaths_ ? modelPaths_->text().split(QLatin1String(";"), Qt::SkipEmptyParts) : QStringList();
    for (QString& path : selectedModelPaths_) path = path.trimmed();
}

void MainWindow::selectLanguage(bool english)
{
    if (english_ == english || trainWatcher_.isRunning() || inferWatcher_.isRunning()) return;
    saveUiState();
    english_ = english;
    QSettings settings;
    settings.setValue(QStringLiteral("ui/language"), english_ ? QStringLiteral("en") : QStringLiteral("zh"));
    qApp->removeTranslator(&translator_);
    if (english_ && translator_.load(QStringLiteral("shape_match_en.qm"), QCoreApplication::applicationDirPath()))
        qApp->installTranslator(&translator_);
    rebuildUiForLanguage();
}

void MainWindow::rebuildUiForLanguage()
{
    const int tabIndex = tabs_ ? tabs_->currentIndex() : 0;
    const int paramTabIndex = parameterTabs_ ? parameterTabs_->currentIndex() : 0;
    const int pyramidIndex = pyramidDisplayLevel_ ? pyramidDisplayLevel_->currentIndex() : 0;

    QWidget* oldCentral = takeCentralWidget();
    delete oldCentral;
    buildUi();
    connectPathSignals();
    restoreUiState();
    setWindowTitle(tr("Shape Match - Qt5 客户端"));
    if (languageMenu_) languageMenu_->setTitle(tr("语言 / Language"));
    if (chineseAction_) { chineseAction_->setText(QStringLiteral("中文")); chineseAction_->setChecked(!english_); }
    if (englishAction_) { englishAction_->setText(QStringLiteral("English")); englishAction_->setChecked(english_); }

    // 恢复金字塔层级下拉框
    if (pyramidDisplayLevel_) {
        const QSignalBlocker blocker(pyramidDisplayLevel_);
        pyramidDisplayLevel_->clear();
        for (int level = 0; level < static_cast<int>(trainLayerImages_.size()); ++level) {
            const int featureCount = (model_ && level < static_cast<int>(model_->templates.size()) &&
                                       model_->templates[level] &&
                                       !model_->templates[level]->shape_angle.empty() &&
                                       model_->templates[level]->shape_angle[0])
                ? static_cast<int>(model_->templates[level]->shape_angle[0]->shape_point.size()) : 0;
            const cv::Mat& layer = trainLayerImages_[level];
            pyramidDisplayLevel_->addItem(
                tr("L%1 (%2×%3, %4 特征)").arg(level).arg(layer.cols).arg(layer.rows).arg(featureCount),
                level);
        }
        if (!trainLayerImages_.empty()) {
            pyramidDisplayLevel_->setCurrentIndex(pyramidIndex >= 0 && pyramidIndex < static_cast<int>(trainLayerImages_.size()) ? pyramidIndex : 0);
            pyramidDisplayLevel_->setEnabled(true);
        } else {
            pyramidDisplayLevel_->setEnabled(false);
        }
    }

    // 恢复图像视图显示
    if (trainView_ && !trainLayerImages_.empty()) {
        int idx = (pyramidDisplayLevel_ && pyramidDisplayLevel_->currentIndex() >= 0)
            ? pyramidDisplayLevel_->currentIndex() : 0;
        if (idx < static_cast<int>(trainLayerImages_.size())) {
            trainView_->setImage(matToImage(trainLayerImages_[idx]));
            if (idx < static_cast<int>(trainLayerContours_.size())) {
                trainView_->setContours(trainLayerContours_[idx]);
            }
        }
    }
    if (inferView_ && !lastInferImage_.empty()) {
        inferView_->setImage(matToImage(lastInferImage_));
        inferView_->setOverlays(lastInferOverlays_);
    }

    // 恢复按钮可用状态
    if (pyramidFeaturesButton_) pyramidFeaturesButton_->setEnabled(model_ != nullptr);
    if (saveJsonButton_) saveJsonButton_->setEnabled(model_ != nullptr);
    if (saveBinaryButton_) saveBinaryButton_->setEnabled(model_ != nullptr);

    // 恢复标签页
    if (tabs_ && tabIndex >= 0 && tabIndex < tabs_->count()) {
        tabs_->setCurrentIndex(tabIndex);
    }
    if (parameterTabs_ && paramTabIndex >= 0 && paramTabIndex < parameterTabs_->count()) {
        parameterTabs_->setCurrentIndex(paramTabIndex);
    }

    // 刷新日志与计时标签
    refreshLogView();
    updateTimingLabels();
}

void MainWindow::setBusy(bool busy, const std::function<QString()>& messageFunc)
{
    if (progress_) progress_->setVisible(busy);
    if (trainButton_) trainButton_->setEnabled(!busy);
    if (pyramidFeaturesButton_) pyramidFeaturesButton_->setEnabled(!busy && model_ != nullptr);
    if (inferButton_) inferButton_->setEnabled(!busy);
    if (saveJsonButton_) saveJsonButton_->setEnabled(!busy && model_ != nullptr);
    if (saveBinaryButton_) saveBinaryButton_->setEnabled(!busy && model_ != nullptr);
    if (messageFunc) log(messageFunc);
}

void MainWindow::setBusy(bool busy, const QString& message)
{
    if (message.isEmpty()) {
        setBusy(busy, std::function<QString()>());
    } else {
        setBusy(busy, [message]() { return message; });
    }
}

void MainWindow::log(const std::function<QString()>& textFunc)
{
    if (!textFunc) return;
    logHistory_.push_back(textFunc);
    if (logView_) logView_->append(textFunc());
}

void MainWindow::log(const QString& text)
{
    log([text]() { return text; });
}

void MainWindow::refreshLogView()
{
    if (!logView_) return;
    logView_->clear();
    for (const auto& func : logHistory_) {
        if (func) logView_->append(func());
    }
}

void MainWindow::showError(const std::function<QString()>& textFunc)
{
    if (!textFunc) return;
    log([this, textFunc]() {
        return tr("错误：%1").arg(textFunc());
    });
    QMessageBox::critical(this, tr("Shape Match"), textFunc());
}

void MainWindow::showError(const QString& text)
{
    showError([text]() { return text; });
}

void MainWindow::updateTimingLabels()
{
    if (trainTimingLabel_) {
        if (lastTrainElapsedMs_ >= 0.0)
            trainTimingLabel_->setText(tr("纯训练耗时: %1 ms").arg(lastTrainElapsedMs_, 0, 'f', 3));
        else
            trainTimingLabel_->setText(tr("纯训练耗时: --"));
    }
    if (inferTimingLabel_) {
        if (lastInferElapsedMs_ >= 0.0)
            inferTimingLabel_->setText(tr("纯推理耗时: %1 ms").arg(lastInferElapsedMs_, 0, 'f', 3));
        else
            inferTimingLabel_->setText(tr("纯推理耗时: --"));
    }
}

QImage MainWindow::matToImage(const cv::Mat& image)
{
    if (image.empty()) return QImage();
    cv::Mat converted;
    if (image.channels() == 1) {
        if (image.depth() == CV_8U) converted = image;
        else image.convertTo(converted, CV_8U);
        return QImage(converted.data, converted.cols, converted.rows,
                      static_cast<int>(converted.step), QImage::Format_Grayscale8).copy();
    }
    if (image.channels() == 3) {
        cv::cvtColor(image, converted, cv::COLOR_BGR2RGB);
        return QImage(converted.data, converted.cols, converted.rows,
                      static_cast<int>(converted.step), QImage::Format_RGB888).copy();
    }
    if (image.channels() == 4) {
        cv::cvtColor(image, converted, cv::COLOR_BGRA2RGBA);
        return QImage(converted.data, converted.cols, converted.rows,
                      static_cast<int>(converted.step), QImage::Format_RGBA8888).copy();
    }
    return QImage();
}
