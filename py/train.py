"""
猫狗分类模型训练脚本 (YOLOv11 Classification)

数据集目录结构:
    assets/
    ├── train/
    │   ├── cat/   (20 张猫的训练图片)
    │   └── dog/   (20 张狗的训练图片)
    └── val/
        ├── cat/   (5 张猫的验证图片)
        └── dog/   (5 张狗的验证图片)

模型: YOLOv11n-cls (轻量级分类模型，153 万参数)
任务: 二分类 (猫 vs 狗)

导出目录: models/cat_vs_dog/
"""

import os
import shutil
from ultralytics import YOLO

# 固定的模型导出目录
EXPORT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "cat_vs_dog"
)


def main():
    # ============================================================
    # 1. 加载预训练模型
    # ============================================================
    # YOLOv11n-cls: nano 级别的分类模型，参数量最小，训练最快
    # 使用 ImageNet 预训练权重，通过迁移学习适配猫狗二分类任务
    # 模型会自动将最后的分类头从 80 类 (ImageNet) 改为 2 类 (猫/狗)
    model = YOLO("yolo11n-cls.pt")

    # ============================================================
    # 2. 开始训练
    # ============================================================
    results = model.train(
        plots=False,
        data="assets",  # 数据集根目录，Ultralytics 自动扫描 train/ 和 val/ 下的子文件夹作为类别
        epochs=200,  # 训练轮数：完整遍历训练集 10 次
        imgsz=224,  # 输入图片尺寸：缩放至 224x224 像素（分类模型标准输入尺寸）
        batch=16,  # 批次大小：每次前向传播同时处理 16 张图片
        device=0,  # 使用 GPU 0 (RTX 4060) 训练，若无 GPU 改为 "cpu"
        workers=12,  # 数据加载线程数：Windows 下设为 0 避免多进程 spawn 错误
        project="runs/classify",  # 输出项目目录：训练日志、权重、图表保存位置
        name="cat_vs_dog",  # 实验名称：结果保存在 runs/classify/cat_vs_dog/ 下
        amp=False,  # 关闭混合精度训练（避免缓存权重损坏导致的 AMP 检查失败）
        verbose=True,  # 打印每个 epoch 的训练进度和指标
        exist_ok=True,  # 允许覆盖已有实验目录，避免自动递增后缀
    )

    # ============================================================
    # 3. 验证模型性能
    # ============================================================
    # 在 val 数据集上评估模型，输出 top-1 和 top-5 准确率
    metrics = model.val()
    print(f"\n验证集 Top-1 准确率: {metrics.top1:.2%}")
    print(f"验证集 Top-5 准确率: {metrics.top5:.2%}")

    # ============================================================
    # 4. 导出模型到固定目录
    # ============================================================
    os.makedirs(EXPORT_DIR, exist_ok=True)

    # 找到训练输出的 best.pt 路径
    train_weights_dir = results.save_dir / "weights"
    best_pt = str(train_weights_dir / "best.pt")

    # 导出 ONNX 到固定目录
    onnx_path = model.export(format="onnx", imgsz=224, project=EXPORT_DIR, name=".")
    shutil.move(onnx_path, os.path.join(EXPORT_DIR, "best.onnx"))
    print(f"\nONNX 模型: {os.path.join(EXPORT_DIR, 'best.onnx')}")

    # 导出 TorchScript 到固定目录
    ts_path = model.export(
        format="torchscript", imgsz=224, project=EXPORT_DIR, name="."
    )
    shutil.move(ts_path, os.path.join(EXPORT_DIR, "best.torchscript"))
    print(f"TorchScript 模型: {os.path.join(EXPORT_DIR, 'best.torchscript')}")

    # 复制 best.pt 到固定目录
    shutil.copy2(best_pt, os.path.join(EXPORT_DIR, "best.pt"))
    print(f"PyTorch 模型: {os.path.join(EXPORT_DIR, 'best.pt')}")

    print(f"\n所有模型已导出至: {EXPORT_DIR}")
    print(f"\n所有模型已导出至: {EXPORT_DIR}")


if __name__ == "__main__":
    main()
