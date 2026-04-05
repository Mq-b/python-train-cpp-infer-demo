/**
 * @file MainWindow.h
 * @brief 主窗口 UI 定义
 *
 * 提供图像分类推理的图形界面，包含:
 *   - 模型选择按钮（加载 .onnx 文件）
 *   - 图片加载按钮（支持 png/jpg/jpeg/bmp）
 *   - 文件夹加载按钮（批量加载图片）
 *   - 推理按钮（执行分类推理并显示结果）
 *   - 结果展示区域（类别名称 + 置信度）
 *   - 文件夹浏览导航（上一张/下一张）
 */
#pragma once

#include <QMainWindow>
#include <QLabel>
#include <QPushButton>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QFileDialog>
#include <QMessageBox>
#include <QLineEdit>
#include <QListWidget>
#include "OnnxClassifier.h"

class MainWindow : public QMainWindow {
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = nullptr);

private slots:
    void selectModel();       ///< 打开文件对话框选择 ONNX 模型
    void loadImage();         ///< 打开文件对话框选择单张图片
    void loadFolder();        ///< 打开文件夹对话框批量加载图片
    void runInference();      ///< 执行分类推理
    void runBatchInference(); ///< 批量推理当前图片列表
    void prevImage();         ///< 上一张图片
    void nextImage();         ///< 下一张图片
    void handleResultItemClicked(QListWidgetItem *item); ///< 点击结果列表项后切换图片

private:
    void setupUI();           ///< 初始化界面布局
    void displayResult(const OnnxClassifier::Result &result);  ///< 显示分类结果
    void showCurrentImage();  ///< 显示当前索引对应的图片
    void updateNavButtons();  ///< 更新导航按钮的可用状态
    void clearResultList();   ///< 清空批量结果列表
    QString toChineseLabel(const QString &name) const; ///< 英文类别映射为中文
    void setStatusText(const QString &text); ///< 设置状态提示
    void addBatchResultItem(int index, const QString &imagePath, const OnnxClassifier::Result &result); ///< 添加批量结果项

    OnnxClassifier m_classifier;  ///< 分类器实例

    QLabel *m_imageLabel;         ///< 图片显示区域
    QLabel *m_resultLabel;        ///< 分类结果文本
    QLabel *m_confidenceLabel;    ///< 置信度文本
    QLabel *m_modelLabel;         ///< 模型路径提示
    QLabel *m_statusLabel;        ///< 当前状态/批量汇总提示
    QLabel *m_imageInfoLabel;     ///< 图片信息提示（当前索引/总数）
    QListWidget *m_resultList;    ///< 批量推理结果列表
    QPushButton *m_selectModelBtn;///< 选择模型按钮
    QPushButton *m_loadImageBtn;  ///< 加载图片按钮
    QPushButton *m_loadFolderBtn; ///< 加载文件夹按钮
    QPushButton *m_inferenceBtn;  ///< 推理按钮
    QPushButton *m_batchInferenceBtn; ///< 批量推理按钮
    QPushButton *m_prevBtn;       ///< 上一张按钮
    QPushButton *m_nextBtn;       ///< 下一张按钮

    QString m_modelPath;          ///< 当前加载的模型路径
    QStringList m_imagePaths;     ///< 图片路径列表（单张模式或文件夹模式）
    int m_currentIndex = -1;      ///< 当前图片索引
};
