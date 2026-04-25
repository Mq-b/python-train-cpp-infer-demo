/**
 * @file OnnxClassifier.h
 * @brief ONNX 模型分类器封装类
 *
 * 封装 ONNX Runtime C++ API，提供简洁的图像分类接口。
 * 目前按 Ultralytics 导出的分类 ONNX 约定进行预处理：
 *   - RGB
 *   - 缩放短边到输入尺寸
 *   - 中心裁剪到模型输入尺寸
 *   - 像素值归一化到 [0, 1]
 *
 * 使用流程:
 *   1. 创建 OnnxClassifier 实例
 *   2. 调用 loadModel() 加载 .onnx 文件
 *   3. 调用 setClassNames() 设置类别名称列表
 *   4. 调用 classify() 对图片路径进行分类
 */
#pragma once

#include <QString>
#include <QStringList>
#include <vector>
#include <memory>
#include <utility>

// ONNX Runtime 前向声明，避免头文件暴露实现细节
namespace Ort {
struct Env;
struct Session;
}

class OnnxClassifier {
public:
    /** @brief 分类结果结构体 */
    struct Result {
        QString className;                              ///< 最高置信度的类别名称
        float confidence = 0.0f;                        ///< 最高置信度值 (0~1)
        std::vector<std::pair<QString, float>> allScores; ///< 所有类别的得分列表
    };

    OnnxClassifier();
    ~OnnxClassifier();

    /**
     * @brief 加载 ONNX 模型文件
     * @param modelPath .onnx 文件的绝对或相对路径
     * @return 成功返回 true，失败返回 false（控制台会输出错误信息）
     */
    bool loadModel(const QString &modelPath);

    /** @return 模型是否已成功加载 */
    bool isLoaded() const;

    /**
     * @brief 设置类别名称列表
     * @param names 类别名称数组，索引需与模型输出维度一一对应
     * @note 必须在 classify() 之前调用，否则结果中的类别名称将显示为 "class_N"
     */
    void setClassNames(const QStringList &names);

    /**
     * @brief 获取模型 metadata 中解析到的类别名称（若存在）
     * @return 类别名称列表；无可用 metadata 时返回空列表
     */
    QStringList modelClassNames() const;

    /**
     * @brief 对输入图像进行分类
     * @param imagePath 待分类图片路径
     * @return 分类结果，包含类别名称和置信度
     */
    Result classify(const QString &imagePath);

private:
    /**
     * @brief 图像预处理：OpenCV 解码 + 缩放短边 + 中心裁剪 + RGB 转换 + [0,1] 归一化
     * @param imagePath 原始图片路径
     * @return 归一化后的浮点数组，布局为 [3, H, W]（CHW 格式）
     */
    std::vector<float> preprocess(const QString &imagePath);

    std::unique_ptr<Ort::Env> m_env;        ///< ONNX Runtime 环境
    std::unique_ptr<Ort::Session> m_session; ///< ONNX 推理会话
    QStringList m_classNames;                ///< 类别名称列表
    QStringList m_modelClassNames;           ///< ONNX metadata 解析出的类别名称
    bool m_loaded = false;                   ///< 模型加载状态标志
    int m_inputWidth = 224;                  ///< 模型输入宽度
    int m_inputHeight = 224;                 ///< 模型输入高度
};
