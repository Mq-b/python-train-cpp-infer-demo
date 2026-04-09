"""
WellColumnClassification 训练脚本 (YOLO Classification)

数据集目录结构:
    assets/WellColumnClassification/
    ├── TrainingSet/
    │   ├── 1+/
    │   ├── 2+/
    │   └── ...
    └── TestSet/
        ├── 1+/
        ├── 2+/
        └── ...

规则:
1. 类别名直接使用文件夹名，例如 `1+`、`DP`、`？`
2. 忽略名为 `_` 的文件夹
3. 忽略没有训练图片的空类别
4. 自动生成 Ultralytics 需要的 `train/val` 标准目录视图

导出目录:
    models/WellColumnClassification/
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

from ultralytics import YOLO


ROOT_DIR = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT_DIR / "assets" / "WellColumnClassification"
TRAINING_SET_DIR = DATASET_DIR / "TrainingSet"
TEST_SET_DIR = DATASET_DIR / "TestSet"
PREPARED_DATASET_DIR = ROOT_DIR / "assets" / ".prepared" / "WellColumnClassification"
EXPORT_DIR = ROOT_DIR / "models" / "WellColumnClassification"

IGNORE_CLASS_NAMES = {"_"}
IMAGE_SUFFIXES = {".bmp", ".jpg", ".jpeg", ".png", ".webp"}

MODEL_NAME = "yolo11n-cls.pt"
RUN_NAME = "WellColumnClassification"
IMG_SIZE = 224
EPOCHS = 200
BATCH_SIZE = 16
DEVICE = 0
WORKERS = 0


def iter_class_dirs(root: Path) -> list[Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"数据集目录不存在: {root}")
    return sorted(
        [path for path in root.iterdir() if path.is_dir() and path.name not in IGNORE_CLASS_NAMES],
        key=lambda path: path.name,
    )


def iter_image_files(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    return sorted(
        [
            path
            for path in root.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        ]
    )


def count_images(root: Path) -> int:
    return len(iter_image_files(root))


def link_or_copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def prepare_split(src_root: Path, dst_root: Path, class_names: list[str]) -> None:
    if dst_root.exists():
        shutil.rmtree(dst_root)
    dst_root.mkdir(parents=True, exist_ok=True)

    for class_name in class_names:
        src_class_dir = src_root / class_name
        dst_class_dir = dst_root / class_name
        dst_class_dir.mkdir(parents=True, exist_ok=True)

        for src_file in iter_image_files(src_class_dir):
            relative_path = src_file.relative_to(src_class_dir)
            link_or_copy_file(src_file, dst_class_dir / relative_path)


def build_dataset_summary() -> tuple[list[str], list[dict]]:
    all_class_names = sorted(
        {path.name for path in iter_class_dirs(TRAINING_SET_DIR)}
        | {path.name for path in iter_class_dirs(TEST_SET_DIR)}
    )

    selected_class_names: list[str] = []
    summary: list[dict] = []

    for class_name in all_class_names:
        train_count = count_images(TRAINING_SET_DIR / class_name)
        val_count = count_images(TEST_SET_DIR / class_name)

        item = {
            "name": class_name,
            "train_images": train_count,
            "val_images": val_count,
        }

        if train_count <= 0:
            item["used"] = False
            item["reason"] = "no training images"
        else:
            item["used"] = True
            selected_class_names.append(class_name)

        summary.append(item)

    if len(selected_class_names) < 2:
        raise RuntimeError(
            "可用于训练的类别少于 2 个。请检查 TrainingSet 中的类别文件夹是否存在有效图片。"
        )

    return selected_class_names, summary


def prepare_dataset() -> tuple[list[str], list[dict]]:
    class_names, summary = build_dataset_summary()

    if PREPARED_DATASET_DIR.exists():
        shutil.rmtree(PREPARED_DATASET_DIR)

    prepare_split(TRAINING_SET_DIR, PREPARED_DATASET_DIR / "train", class_names)
    prepare_split(TEST_SET_DIR, PREPARED_DATASET_DIR / "val", class_names)

    return class_names, summary


def write_metadata(class_names: list[str], summary: list[dict], train_run_dir: Path | None = None) -> None:
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    labels_text = "\n".join(class_names) + "\n"
    (EXPORT_DIR / "labels.txt").write_text(labels_text, encoding="utf-8")
    (EXPORT_DIR / "dataset_summary.json").write_text(
        json.dumps(
            {
                "dataset_root": str(DATASET_DIR),
                "prepared_dataset_root": str(PREPARED_DATASET_DIR),
                "ignored_class_names": sorted(IGNORE_CLASS_NAMES),
                "used_classes": class_names,
                "summary": summary,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    if train_run_dir is not None:
        (train_run_dir / "labels.txt").write_text(labels_text, encoding="utf-8")


def move_exported_file(exported_path: str | Path, target_path: Path) -> Path:
    exported_path = Path(exported_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    if target_path.exists() and exported_path.resolve() != target_path.resolve():
        target_path.unlink()

    if exported_path.resolve() != target_path.resolve():
        shutil.move(str(exported_path), str(target_path))

    return target_path


def export_models(best_pt_path: Path) -> None:
    best_model = YOLO(str(best_pt_path))

    onnx_path = best_model.export(
        format="onnx",
        imgsz=IMG_SIZE,
        project=str(EXPORT_DIR),
        name=".",
        exist_ok=True,
    )
    onnx_path = move_exported_file(onnx_path, EXPORT_DIR / "best.onnx")
    print(f"\nONNX 模型: {onnx_path}")

    torchscript_path = best_model.export(
        format="torchscript",
        imgsz=IMG_SIZE,
        project=str(EXPORT_DIR),
        name=".",
        exist_ok=True,
    )
    torchscript_path = move_exported_file(
        torchscript_path, EXPORT_DIR / "best.torchscript"
    )
    print(f"TorchScript 模型: {torchscript_path}")


def print_dataset_summary(summary: list[dict], class_names: list[str]) -> None:
    print("\n参与训练的类别:")
    for index, class_name in enumerate(class_names):
        print(f"  [{index}] {class_name}")

    print("\n数据集统计:")
    for item in summary:
        used_flag = "使用" if item["used"] else "跳过"
        reason = f" ({item['reason']})" if "reason" in item else ""
        print(
            f"  {used_flag:2}  {item['name']}: "
            f"train={item['train_images']}, val={item['val_images']}{reason}"
        )


def main() -> None:
    class_names, summary = prepare_dataset()
    write_metadata(class_names, summary)
    print_dataset_summary(summary, class_names)

    model = YOLO(MODEL_NAME)

    results = model.train(
        plots=False,
        data=str(PREPARED_DATASET_DIR),
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        workers=WORKERS,
        project="runs/classify",
        name=RUN_NAME,
        amp=False,
        verbose=True,
        exist_ok=True,
    )

    train_run_dir = Path(results.save_dir)
    train_weights_dir = train_run_dir / "weights"
    best_pt_path = train_weights_dir / "best.pt"
    write_metadata(class_names, summary, train_run_dir=train_run_dir)

    best_model = YOLO(str(best_pt_path))
    metrics = best_model.val(
        data=str(PREPARED_DATASET_DIR),
        split="val",
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        workers=WORKERS,
        verbose=True,
    )
    print(f"\n验证集 Top-1 准确率: {metrics.top1:.2%}")
    print(f"验证集 Top-5 准确率: {metrics.top5:.2%}")

    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_pt_path, EXPORT_DIR / "best.pt")
    print(f"\nPyTorch 模型: {EXPORT_DIR / 'best.pt'}")

    export_models(best_pt_path)

    print(f"\n所有模型已导出至: {EXPORT_DIR}")


if __name__ == "__main__":
    main()
