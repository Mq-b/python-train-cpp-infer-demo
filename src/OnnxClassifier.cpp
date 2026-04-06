/**
 * @file OnnxClassifier.cpp
 * @brief ONNX 模型分类器实现
 *
 * 核心流程:
 *   loadModel()  -> 创建 Ort::Env + Ort::Session
 *   classify()   -> preprocess() -> Ort::Session::Run() -> 解析输出
 */
#include "OnnxClassifier.h"

#if __has_include(<onnxruntime_cxx_api.h>)
#include <onnxruntime_cxx_api.h>
#elif __has_include(<onnxruntime/core/session/onnxruntime_cxx_api.h>)
#include <onnxruntime/core/session/onnxruntime_cxx_api.h>
#else
#error "onnxruntime_cxx_api.h not found"
#endif
#include <QDebug>
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
        qDebug() << "Preprocess mode: resize shortest edge -> center crop -> RGB -> [0,1]";

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

OnnxClassifier::Result OnnxClassifier::classify(const QImage &image) {
    Result result;
    if (!m_loaded) {
        qWarning() << "Model not loaded";
        return result;
    }

    try {
        // 1. 图像预处理（缩放短边、中心裁剪、转 RGB、[0,1]）
        auto input = preprocess(image);

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

        // 5. 解析输出（Softmax 概率分布）
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

std::vector<float> OnnxClassifier::preprocess(const QImage &image) {
    QImage rgb = image.convertToFormat(QImage::Format_RGB32);
    if (rgb.isNull() || rgb.width() <= 0 || rgb.height() <= 0) {
        return {};
    }

    // Ultralytics 分类推理默认流程：
    // 1) 将短边缩放到目标尺寸
    // 2) 从中间裁出目标大小
    const float scale = std::max(
        static_cast<float>(m_inputWidth) / static_cast<float>(rgb.width()),
        static_cast<float>(m_inputHeight) / static_cast<float>(rgb.height())
    );
    const int resizedWidth = std::max(1, static_cast<int>(std::lround(rgb.width() * scale)));
    const int resizedHeight = std::max(1, static_cast<int>(std::lround(rgb.height() * scale)));

    QImage resized = rgb.scaled(
        resizedWidth, resizedHeight, Qt::IgnoreAspectRatio, Qt::SmoothTransformation
    );

    const int cropX = std::max(0, (resized.width() - m_inputWidth) / 2);
    const int cropY = std::max(0, (resized.height() - m_inputHeight) / 2);
    QImage cropped = resized.copy(cropX, cropY, m_inputWidth, m_inputHeight);

    std::vector<float> input(3 * m_inputWidth * m_inputHeight);

    // 只做 [0,1] 归一化，不额外做 ImageNet mean/std 标准化。
    // 这里使用 qRed/qGreen/qBlue 读取像素，避免直接按字节解释 QImage 内存布局带来的通道偏差。
    for (int y = 0; y < m_inputHeight; ++y) {
        for (int x = 0; x < m_inputWidth; ++x) {
            const QRgb pixel = cropped.pixel(x, y);
            const float r = qRed(pixel) / 255.0f;
            const float g = qGreen(pixel) / 255.0f;
            const float b = qBlue(pixel) / 255.0f;

            input[0 * m_inputWidth * m_inputHeight + y * m_inputWidth + x] = r;
            input[1 * m_inputWidth * m_inputHeight + y * m_inputWidth + x] = g;
            input[2 * m_inputWidth * m_inputHeight + y * m_inputWidth + x] = b;
        }
    }

    return input;
}
