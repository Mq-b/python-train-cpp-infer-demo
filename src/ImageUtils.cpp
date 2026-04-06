/**
 * @file ImageUtils.cpp
 * @brief 基于 OpenCV 的图片读取与 Qt 图像转换。
 */
#include "ImageUtils.h"

#include <QFile>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <vector>

namespace ImageUtils {

cv::Mat loadColorImage(const QString &imagePath) {
    // 先用 Qt 读取文件字节，避免直接把 QString 路径传给 OpenCV 时
    // 在 Windows/非 ASCII 路径场景下出现兼容性问题。
    QFile file(imagePath);
    if (!file.open(QIODevice::ReadOnly)) {
        return {};
    }

    const QByteArray encoded = file.readAll();
    if (encoded.isEmpty()) {
        return {};
    }

    // OpenCV 解码统一返回 BGR 排列的彩色图，供后续推理前处理复用。
    std::vector<uchar> buffer(encoded.begin(), encoded.end());
    return cv::imdecode(buffer, cv::IMREAD_COLOR);
}

QImage toQImage(const cv::Mat &image) {
    if (image.empty()) {
        return {};
    }

    if (image.type() == CV_8UC1) {
        return QImage(
            image.data,
            image.cols,
            image.rows,
            static_cast<int>(image.step),
            QImage::Format_Grayscale8
        ).copy();
    }

    if (image.type() == CV_8UC3) {
        cv::Mat rgb;
        // Qt 的 RGB888 与 OpenCV 的 BGR 三通道内存顺序不同，需要显式转换。
        cv::cvtColor(image, rgb, cv::COLOR_BGR2RGB);
        return QImage(
            rgb.data,
            rgb.cols,
            rgb.rows,
            static_cast<int>(rgb.step),
            QImage::Format_RGB888
        ).copy();
    }

    if (image.type() == CV_8UC4) {
        cv::Mat rgba;
        // 四通道场景同理，需要从 BGRA 转成 Qt 侧期望的 RGBA。
        cv::cvtColor(image, rgba, cv::COLOR_BGRA2RGBA);
        return QImage(
            rgba.data,
            rgba.cols,
            rgba.rows,
            static_cast<int>(rgba.step),
            QImage::Format_RGBA8888
        ).copy();
    }

    return {};
}

} // namespace ImageUtils
