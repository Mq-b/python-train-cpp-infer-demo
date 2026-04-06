/**
 * @file ImageUtils.h
 * @brief 使用 OpenCV 统一图片解码与格式转换。
 */
#pragma once

#include <QImage>
#include <QString>

#include <opencv2/core/mat.hpp>

namespace ImageUtils {

/**
 * @brief 从磁盘读取图片并解码为 BGR 三通道 Mat。
 *
 * 这里先使用 Qt 的 QFile 读取原始字节，再交给 OpenCV `imdecode()`，
 * 这样可以同时兼顾：
 *   - Qt 对 QString/中文路径的处理能力
 *   - OpenCV 的统一图像解码行为
 *
 * @param imagePath 图片路径
 * @return 成功时返回 `CV_8UC3` 的 BGR 图像；失败时返回空 Mat
 */
cv::Mat loadColorImage(const QString &imagePath);

/**
 * @brief 将 OpenCV Mat 转成可供 Qt UI 展示的 QImage。
 *
 * 当前仅处理常见的 8-bit 图像类型：
 *   - `CV_8UC1` -> `QImage::Format_Grayscale8`
 *   - `CV_8UC3` -> `QImage::Format_RGB888`
 *   - `CV_8UC4` -> `QImage::Format_RGBA8888`
 *
 * 对三通道和四通道输入会先做 BGR/BGRA 到 RGB/RGBA 的颜色顺序转换。
 *
 * @param image OpenCV 图像
 * @return 转换后的 QImage；不支持的类型返回空 QImage
 */
QImage toQImage(const cv::Mat &image);

} // namespace ImageUtils
