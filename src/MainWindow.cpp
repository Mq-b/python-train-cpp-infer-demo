/**
 * @file MainWindow.cpp
 * @brief 主窗口 UI 实现
 *
 * 功能:
 *   - 加载 ONNX 模型
 *   - 加载单张图片或整个文件夹
 *   - 上一张/下一张浏览
 *   - 执行推理并显示结果
 */
#include "MainWindow.h"
#include "ImageUtils.h"
#include <QApplication>
#include <QPixmap>
#include <QDir>
#include <QDirIterator>
#include <QFileInfo>
#include <QListWidgetItem>
#include <functional>

namespace {

QImage readImageWithOpenCV(const QString &path) {
    return ImageUtils::toQImage(ImageUtils::loadColorImage(path));
}

constexpr int kImageIndexRole = Qt::UserRole;
constexpr int kClassNameRole = Qt::UserRole + 1;
constexpr int kConfidenceRole = Qt::UserRole + 2;
constexpr int kDetailRole = Qt::UserRole + 3;

QString buildScoreDetail(const OnnxClassifier::Result &result, const std::function<QString(const QString &)> &mapper) {
    QString detail = "所有类别得分:\n";
    for (const auto &score : result.allScores) {
        detail += QString("  %1: %2%\n")
            .arg(mapper(score.first))
            .arg(score.second * 100.0, 0, 'f', 2);
    }
    return detail;
}

} // namespace

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
{
    setupUI();
    setWindowTitle("YOLO 图像分类推理工具");
    resize(800, 600);
}

void MainWindow::setupUI() {
    QWidget *centralWidget = new QWidget(this);
    QVBoxLayout *mainLayout = new QVBoxLayout(centralWidget);

    // 第一行：模型选择 + 图片加载 + 文件夹加载
    QHBoxLayout *btnLayout1 = new QHBoxLayout();
    m_selectModelBtn = new QPushButton("选择 ONNX 模型", this);
    m_loadImageBtn = new QPushButton("加载图片", this);
    m_loadFolderBtn = new QPushButton("加载文件夹", this);
    btnLayout1->addWidget(m_selectModelBtn);
    btnLayout1->addWidget(m_loadImageBtn);
    btnLayout1->addWidget(m_loadFolderBtn);
    mainLayout->addLayout(btnLayout1);

    // 第二行：推理按钮 + 批量按钮 + 导航按钮
    QHBoxLayout *btnLayout2 = new QHBoxLayout();
    m_inferenceBtn = new QPushButton("推理当前图片", this);
    m_inferenceBtn->setEnabled(false);
    m_inferenceBtn->setStyleSheet("font-weight: bold; background: #4CAF50; color: white; padding: 6px;");

    m_batchInferenceBtn = new QPushButton("批量推理全部", this);
    m_batchInferenceBtn->setEnabled(false);
    m_batchInferenceBtn->setStyleSheet("font-weight: bold; background: #1E88E5; color: white; padding: 6px;");

    m_prevBtn = new QPushButton("◀ 上一张", this);
    m_prevBtn->setEnabled(false);
    m_nextBtn = new QPushButton("下一张 ▶", this);
    m_nextBtn->setEnabled(false);

    m_imageInfoLabel = new QLabel("", this);
    m_imageInfoLabel->setAlignment(Qt::AlignCenter);
    m_imageInfoLabel->setStyleSheet("color: gray;");

    btnLayout2->addWidget(m_prevBtn);
    btnLayout2->addWidget(m_inferenceBtn);
    btnLayout2->addWidget(m_batchInferenceBtn);
    btnLayout2->addWidget(m_nextBtn);
    btnLayout2->addWidget(m_imageInfoLabel);
    mainLayout->addLayout(btnLayout2);

    // 模型路径提示
    m_modelLabel = new QLabel("未加载模型", this);
    m_modelLabel->setStyleSheet("color: gray;");
    mainLayout->addWidget(m_modelLabel);

    m_statusLabel = new QLabel("状态: 就绪", this);
    m_statusLabel->setStyleSheet("color: #555;");
    mainLayout->addWidget(m_statusLabel);

    // 图片显示区域
    m_imageLabel = new QLabel("未加载图片", this);
    m_imageLabel->setAlignment(Qt::AlignCenter);
    m_imageLabel->setMinimumHeight(300);
    m_imageLabel->setStyleSheet("border: 1px solid gray; background: #f0f0f0;");
    mainLayout->addWidget(m_imageLabel);

    // 推理结果
    m_resultLabel = new QLabel("结果: --", this);
    m_resultLabel->setStyleSheet("font-size: 16px; font-weight: bold;");
    mainLayout->addWidget(m_resultLabel);

    m_confidenceLabel = new QLabel("置信度: --", this);
    m_confidenceLabel->setStyleSheet("font-size: 14px;");
    mainLayout->addWidget(m_confidenceLabel);

    m_resultList = new QListWidget(this);
    m_resultList->setMinimumHeight(180);
    m_resultList->setAlternatingRowColors(true);
    mainLayout->addWidget(m_resultList);

    setCentralWidget(centralWidget);

    // 信号槽连接
    connect(m_selectModelBtn, &QPushButton::clicked, this, &MainWindow::selectModel);
    connect(m_loadImageBtn, &QPushButton::clicked, this, &MainWindow::loadImage);
    connect(m_loadFolderBtn, &QPushButton::clicked, this, &MainWindow::loadFolder);
    connect(m_inferenceBtn, &QPushButton::clicked, this, &MainWindow::runInference);
    connect(m_batchInferenceBtn, &QPushButton::clicked, this, &MainWindow::runBatchInference);
    connect(m_prevBtn, &QPushButton::clicked, this, &MainWindow::prevImage);
    connect(m_nextBtn, &QPushButton::clicked, this, &MainWindow::nextImage);
    connect(m_resultList, &QListWidget::itemClicked, this, &MainWindow::handleResultItemClicked);
}

void MainWindow::selectModel() {
    QString path = QFileDialog::getOpenFileName(
        this, "选择 ONNX 模型文件", "", "ONNX Model (*.onnx)"
    );
    if (path.isEmpty()) return;

    m_modelPath = path;

    if (m_classifier.loadModel(path)) {
        m_classifier.setClassNames({"cat", "dog"});
        m_modelLabel->setText("模型: " + path);
        m_modelLabel->setStyleSheet("color: green;");
        m_inferenceBtn->setEnabled(m_currentIndex >= 0);
        m_batchInferenceBtn->setEnabled(!m_imagePaths.isEmpty());
        setStatusText("状态: 模型加载成功");
    } else {
        m_modelLabel->setText("模型加载失败: " + path);
        m_modelLabel->setStyleSheet("color: red;");
        setStatusText("状态: 模型加载失败");
        QMessageBox::critical(this, "错误", "ONNX 模型加载失败！");
    }
}

void MainWindow::loadImage() {
    QString path = QFileDialog::getOpenFileName(
        this, "选择图片", "", "Images (*.png *.jpg *.jpeg *.bmp *.webp)"
    );
    if (path.isEmpty()) return;

    m_imagePaths = QStringList{path};
    m_currentIndex = 0;
    clearResultList();
    showCurrentImage();
    m_inferenceBtn->setEnabled(m_classifier.isLoaded());
    m_batchInferenceBtn->setEnabled(m_classifier.isLoaded());
    setStatusText("状态: 已加载 1 张图片");
}

void MainWindow::loadFolder() {
    QString dir = QFileDialog::getExistingDirectory(this, "选择图片文件夹");
    if (dir.isEmpty()) return;

    QStringList filters;
    filters << "*.png" << "*.jpg" << "*.jpeg" << "*.bmp" << "*.webp";
    QStringList paths;
    QDirIterator it(dir, filters, QDir::Files, QDirIterator::Subdirectories);
    while (it.hasNext()) {
        paths << QDir::toNativeSeparators(it.next());
    }
    paths.sort(Qt::CaseInsensitive);
    m_imagePaths = paths;

    if (m_imagePaths.isEmpty()) {
        QMessageBox::warning(this, "警告", "文件夹中没有找到图片文件！");
        setStatusText("状态: 文件夹中没有图片");
        return;
    }

    m_currentIndex = 0;
    clearResultList();
    showCurrentImage();
    m_inferenceBtn->setEnabled(m_classifier.isLoaded());
    m_batchInferenceBtn->setEnabled(m_classifier.isLoaded());
    setStatusText(QString("状态: 已递归加载 %1 张图片").arg(m_imagePaths.size()));
}

void MainWindow::runInference() {
    if (m_currentIndex < 0 || m_currentIndex >= m_imagePaths.size()) return;
    if (!m_classifier.isLoaded()) {
        QMessageBox::warning(this, "警告", "请先加载 ONNX 模型！");
        return;
    }

    const QString &imagePath = m_imagePaths[m_currentIndex];
    OnnxClassifier::Result result = m_classifier.classify(imagePath);
    if (result.allScores.empty()) {
        QMessageBox::warning(this, "警告", "图片格式不支持: " + m_imagePaths[m_currentIndex]);
        return;
    }

    displayResult(result);
    setStatusText(QString("状态: 当前图片推理完成 (%1)")
        .arg(QFileInfo(imagePath).fileName()));
}

void MainWindow::runBatchInference() {
    if (m_imagePaths.isEmpty()) {
        QMessageBox::warning(this, "警告", "请先加载图片或文件夹！");
        return;
    }
    if (!m_classifier.isLoaded()) {
        QMessageBox::warning(this, "警告", "请先加载 ONNX 模型！");
        return;
    }

    clearResultList();

    int catCount = 0;
    int dogCount = 0;
    int okCount = 0;

    for (int i = 0; i < m_imagePaths.size(); ++i) {
        const QString &path = m_imagePaths[i];
        setStatusText(QString("状态: 批量推理中 %1/%2 - %3")
            .arg(i + 1)
            .arg(m_imagePaths.size())
            .arg(QFileInfo(path).fileName()));

        OnnxClassifier::Result result = m_classifier.classify(path);
        if (result.allScores.empty()) {
            auto *item = new QListWidgetItem(
                QString("第%1张 | %2 | 读取失败").arg(i + 1).arg(QFileInfo(path).fileName())
            );
            item->setData(kImageIndexRole, i);
            m_resultList->addItem(item);
            QApplication::processEvents();
            continue;
        }

        addBatchResultItem(i, path, result);
        okCount++;

        const QString label = toChineseLabel(result.className);
        if (label == "猫") {
            catCount++;
        } else if (label == "狗") {
            dogCount++;
        }

        if (i == m_currentIndex) {
            displayResult(result);
        }
        QApplication::processEvents();
    }

    setStatusText(QString("状态: 批量推理完成，共 %1 张，成功 %2 张，猫 %3 张，狗 %4 张")
        .arg(m_imagePaths.size())
        .arg(okCount)
        .arg(catCount)
        .arg(dogCount));
}

void MainWindow::prevImage() {
    if (m_currentIndex > 0) {
        m_currentIndex--;
        showCurrentImage();
    }
}

void MainWindow::nextImage() {
    if (m_currentIndex < m_imagePaths.size() - 1) {
        m_currentIndex++;
        showCurrentImage();
    }
}

void MainWindow::showCurrentImage() {
    if (m_currentIndex < 0 || m_currentIndex >= m_imagePaths.size()) return;

    QPixmap pixmap = QPixmap::fromImage(readImageWithOpenCV(m_imagePaths[m_currentIndex]));
    if (!pixmap.isNull()) {
        m_imageLabel->setPixmap(pixmap.scaled(
            m_imageLabel->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation
        ));
    } else {
        m_imageLabel->clear();
        m_imageLabel->setText("图片读取失败");
    }

    // 更新图片信息
    m_imageInfoLabel->setText(
        QString("[%1/%2] %3").arg(m_currentIndex + 1).arg(m_imagePaths.size())
            .arg(QFileInfo(m_imagePaths[m_currentIndex]).fileName())
    );

    // 清空上次推理结果
    m_resultLabel->setText("结果: --");
    m_confidenceLabel->setText("置信度: --");
    m_confidenceLabel->setToolTip("");

    updateNavButtons();
}

void MainWindow::updateNavButtons() {
    m_prevBtn->setEnabled(m_currentIndex > 0);
    m_nextBtn->setEnabled(m_currentIndex < m_imagePaths.size() - 1);
}

void MainWindow::displayResult(const OnnxClassifier::Result &result) {
    const QString displayName = toChineseLabel(result.className);
    m_resultLabel->setText("结果: " + displayName);
    m_confidenceLabel->setText(
        QString("置信度: %1%").arg(result.confidence * 100.0, 0, 'f', 2)
    );

    const QString detail = buildScoreDetail(result, [this](const QString &name) {
        return toChineseLabel(name);
    });
    m_confidenceLabel->setToolTip(detail);
}

void MainWindow::handleResultItemClicked(QListWidgetItem *item) {
    if (!item) return;

    const int index = item->data(kImageIndexRole).toInt();
    if (index < 0 || index >= m_imagePaths.size()) return;

    m_currentIndex = index;
    showCurrentImage();

    const QString className = item->data(kClassNameRole).toString();
    if (!className.isEmpty()) {
        m_resultLabel->setText("结果: " + className);
        m_confidenceLabel->setText(
            QString("置信度: %1%").arg(item->data(kConfidenceRole).toDouble(), 0, 'f', 2)
        );
        m_confidenceLabel->setToolTip(item->data(kDetailRole).toString());
    }
}

void MainWindow::clearResultList() {
    m_resultList->clear();
}

QString MainWindow::toChineseLabel(const QString &name) const {
    const QString lowered = name.trimmed().toLower();
    if (lowered == "cat") return "猫";
    if (lowered == "dog") return "狗";
    return name;
}

void MainWindow::setStatusText(const QString &text) {
    m_statusLabel->setText(text);
}

void MainWindow::addBatchResultItem(int index, const QString &imagePath, const OnnxClassifier::Result &result) {
    const QString displayName = toChineseLabel(result.className);
    const QString itemText = QString("第%1张 | %2 | 预测: %3 | 置信度: %4%")
        .arg(index + 1)
        .arg(QFileInfo(imagePath).fileName())
        .arg(displayName)
        .arg(result.confidence * 100.0, 0, 'f', 2);

    auto *item = new QListWidgetItem(itemText);
    const QString detail = buildScoreDetail(result, [this](const QString &name) {
        return toChineseLabel(name);
    }) + "\n路径: " + imagePath;

    item->setData(kImageIndexRole, index);
    item->setData(kClassNameRole, displayName);
    item->setData(kConfidenceRole, result.confidence * 100.0);
    item->setData(kDetailRole, detail);
    item->setToolTip(detail);
    m_resultList->addItem(item);
}
