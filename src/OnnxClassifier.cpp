/**
 * @file OnnxClassifier.cpp
 * @brief ONNX 模型分类器实现
 *
 * 核心流程:
 *   loadModel()  -> 创建 Ort::Env + Ort::Session
 *   classify()   -> preprocess() -> Ort::Session::Run() -> 解析输出
 */
#include "OnnxClassifier.h"
#include "ImageUtils.h"

#if __has_include(<onnxruntime_cxx_api.h>)
#include <onnxruntime_cxx_api.h>
#elif __has_include(<onnxruntime/core/session/onnxruntime_cxx_api.h>)
#include <onnxruntime/core/session/onnxruntime_cxx_api.h>
#else
#error "onnxruntime_cxx_api.h not found"
#endif
#include <QDebug>
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <array>
#include <cmath>

OnnxClassifier::OnnxClassifier() {
}

OnnxClassifier::~OnnxClassifier() {
}

bool OnnxClassifier::loadModel(const QString &modelPath) {
    m_loaded = false;
    m_session.reset();

    try {
        // 创建 ONNX Runtime 运行环境
        m_env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "YOLOClassifier");

        // 配置推理选项：4 线程 + 全图优化
        Ort::SessionOptions sessionOptions;
        sessionOptions.SetIntraOpNumThreads(4);
        sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        // 加载模型文件
#ifdef _WIN32
        const std::wstring nativeModelPath = modelPath.toStdWString();
        m_session = std::make_unique<Ort::Session>(
            *m_env, nativeModelPath.c_str(), sessionOptions
        );
#else
        const std::string nativeModelPath = modelPath.toStdString();
        m_session = std::make_unique<Ort::Session>(
            *m_env, nativeModelPath.c_str(), sessionOptions
        );
#endif

        // 验证模型输入
        size_t inputCount = m_session->GetInputCount();
        if (inputCount == 0) {
            qWarning() << "No input found in ONNX model";
            return false;
        }

        // 打印模型输入形状（调试用）
        auto inputTypeInfo = m_session->GetInputTypeInfo(0);
        auto tensorInfo = inputTypeInfo.GetTensorTypeAndShapeInfo();
        auto shape = tensorInfo.GetShape();

        if (shape.size() == 4 && shape[2] > 0 && shape[3] > 0) {
            m_inputHeight = static_cast<int>(shape[2]);
            m_inputWidth = static_cast<int>(shape[3]);
        }

        qDebug() << "Model loaded:" << modelPath;
        QStringList dims;
        for (const auto dim : shape) {
            dims << QString::number(dim);
        }

        qDebug() << "Input shape:" << dims.join(" x ");
        qDebug() << "Preprocess mode: OpenCV decode -> resize shortest edge -> center crop -> RGB -> [0,1]";

        m_loaded = true;
        return true;
    } catch (const std::exception &e) {
        qCritical() << "Failed to load model:" << e.what();
        return false;
    }
}

bool OnnxClassifier::isLoaded() const {
    return m_loaded;
}

void OnnxClassifier::setClassNames(const QStringList &names) {
    m_classNames = names;
}

OnnxClassifier::Result OnnxClassifier::classify(const QString &imagePath) {
    Result result;
    if (!m_loaded) {
        qWarning() << "Model not loaded";
        return result;
    }

    try {
        // 1. 图像预处理（缩放短边、中心裁剪、转 RGB、[0,1]）
        auto input = preprocess(imagePath);
        if (input.empty()) {
            qWarning() << "Failed to preprocess image:" << imagePath;
            return result;
        }

        // 2. 获取模型输入/输出节点名称
        Ort::AllocatorWithDefaultOptions allocator;
        auto inputName = m_session->GetInputNameAllocated(0, allocator);
        auto outputName = m_session->GetOutputNameAllocated(0, allocator);
        const char *inputNames[] = { inputName.get() };
        const char *outputNames[] = { outputName.get() };

        // 3. 创建输入 Tensor（NCHW 格式: [1, 3, H, W]）
        auto memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        const std::array<int64_t, 4> inputShape{
            1, 3, static_cast<int64_t>(m_inputHeight), static_cast<int64_t>(m_inputWidth)
        };
        auto inputTensor = Ort::Value::CreateTensor<float>(
            memoryInfo, input.data(), input.size(),
            inputShape.data(), inputShape.size()
        );

        // 4. 执行推理
        auto outputTensors = m_session->Run(
            Ort::RunOptions{nullptr},
            inputNames, &inputTensor, 1,
            outputNames, 1
        );

        // 5. 解析输出（模型输出的类别概率分布）
        auto outputData = outputTensors[0].GetTensorMutableData<float>();
        auto outputShape = outputTensors[0].GetTensorTypeAndShapeInfo().GetShape();
        int numClasses = static_cast<int>(outputShape[outputShape.size() - 1]);

        std::vector<float> scores(outputData, outputData + numClasses);

        // 6. 找出最高置信度的类别
        int bestIdx = 0;
        float bestScore = scores[0];
        for (int i = 1; i < numClasses; ++i) {
            if (scores[i] > bestScore) {
                bestScore = scores[i];
                bestIdx = i;
            }
        }

        QString className = bestIdx < m_classNames.size()
            ? m_classNames[bestIdx]
            : QString("class_%1").arg(bestIdx);

        result.className = className;
        result.confidence = bestScore;

        // 7. 收集所有类别的得分
        for (int i = 0; i < numClasses; ++i) {
            QString name = i < m_classNames.size()
                ? m_classNames[i]
                : QString("class_%1").arg(i);
            result.allScores.push_back({name, scores[i]});
        }

        return result;
    } catch (const std::exception &e) {
        qCritical() << "Classification failed:" << e.what();
        return result;
    }
}

std::vector<float> OnnxClassifier::preprocess(const QString &imagePath) {
    cv::Mat bgr = ImageUtils::loadColorImage(imagePath);
    if (bgr.empty() || bgr.cols <= 0 || bgr.rows <= 0) {
        return {};
    }

    // Ultralytics 分类推理默认流程：
    // 1) 将短边缩放到目标尺寸
    // 2) 从中间裁出目标大小
    const float scale = std::max(
        static_cast<float>(m_inputWidth) / static_cast<float>(bgr.cols),
        static_cast<float>(m_inputHeight) / static_cast<float>(bgr.rows)
    );
    const int resizedWidth = std::max(1, static_cast<int>(std::lround(bgr.cols * scale)));
    const int resizedHeight = std::max(1, static_cast<int>(std::lround(bgr.rows * scale)));

    cv::Mat resized;
    cv::resize(
        bgr,
        resized,
        cv::Size(resizedWidth, resizedHeight),
        0.0,
        0.0,
        cv::INTER_LINEAR
    );

    const int cropX = std::max(0, (resized.cols - m_inputWidth) / 2);
    const int cropY = std::max(0, (resized.rows - m_inputHeight) / 2);
    const cv::Rect roi(cropX, cropY, m_inputWidth, m_inputHeight);
    cv::Mat cropped = resized(roi);

    cv::Mat rgb;
    cv::cvtColor(cropped, rgb, cv::COLOR_BGR2RGB);

    cv::Mat normalized;
    rgb.convertTo(normalized, CV_32FC3, 1.0 / 255.0);

    std::vector<float> input(3 * m_inputWidth * m_inputHeight);
    const int planeSize = m_inputWidth * m_inputHeight;

    for (int y = 0; y < m_inputHeight; ++y) {
        const auto *row = normalized.ptr<cv::Vec3f>(y);
        for (int x = 0; x < m_inputWidth; ++x) {
            const cv::Vec3f &pixel = row[x];
            const int offset = y * m_inputWidth + x;
            input[0 * planeSize + offset] = pixel[0];
            input[1 * planeSize + offset] = pixel[1];
            input[2 * planeSize + offset] = pixel[2];
        }
    }

    return input;
}
