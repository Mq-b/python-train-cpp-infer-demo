from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image


def _safe_literal(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return ast.literal_eval(value)
    except Exception:
        return value


def _normalize_names(names_value: Any) -> list[str]:
    if isinstance(names_value, dict):
        pairs: list[tuple[int, str]] = []
        for key, value in names_value.items():
            try:
                idx = int(key)
            except Exception:
                continue
            pairs.append((idx, str(value)))
        if not pairs:
            return []
        pairs.sort(key=lambda item: item[0])
        max_index = pairs[-1][0]
        out = [f"class_{i}" for i in range(max_index + 1)]
        for idx, name in pairs:
            out[idx] = name
        return out
    if isinstance(names_value, (list, tuple)):
        return [str(x) for x in names_value]
    return []


def _parse_imgsz(imgsz_value: Any, input_shape: list[Any]) -> list[int]:
    if isinstance(imgsz_value, (list, tuple)) and len(imgsz_value) == 2:
        try:
            return [int(imgsz_value[0]), int(imgsz_value[1])]
        except Exception:
            pass

    if len(input_shape) == 4:
        h = input_shape[2]
        w = input_shape[3]
        if isinstance(h, int) and isinstance(w, int):
            return [h, w]
    return [224, 224]


def _bgr_to_chw_float01(arr_bgr: np.ndarray) -> np.ndarray:
    arr_rgb = cv2.cvtColor(arr_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return np.transpose(arr_rgb, (2, 0, 1))[None]


def preprocess_cpp_style(arr_bgr: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    h, w = arr_bgr.shape[:2]
    scale = max(target_w / float(w), target_h / float(h))
    resized_w = max(target_w, int(np.floor(w * scale)))
    resized_h = max(target_h, int(np.floor(h * scale)))
    interpolation = cv2.INTER_AREA if resized_w < w or resized_h < h else cv2.INTER_LINEAR
    resized = cv2.resize(arr_bgr, (resized_w, resized_h), interpolation=interpolation)

    crop_x = max(0, int(np.rint((resized_w - target_w) / 2.0)))
    crop_y = max(0, int(np.rint((resized_h - target_h) / 2.0)))
    cropped = resized[crop_y : crop_y + target_h, crop_x : crop_x + target_w]
    return _bgr_to_chw_float01(cropped)


def preprocess_ultralytics_reference(arr_bgr: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    arr_rgb = cv2.cvtColor(arr_bgr, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(arr_rgb)
    src_w, src_h = img.size

    if src_w <= src_h:
        resized_w = target_w
        resized_h = max(1, int(target_w * src_h / src_w))
    else:
        resized_h = target_h
        resized_w = max(1, int(target_h * src_w / src_h))

    img = img.resize((resized_w, resized_h), resample=Image.BILINEAR)
    crop_x = int(round((resized_w - target_w) / 2.0))
    crop_y = int(round((resized_h - target_h) / 2.0))
    img = img.crop((crop_x, crop_y, crop_x + target_w, crop_y + target_h))

    arr = np.asarray(img, dtype=np.float32) / 255.0
    return np.transpose(arr, (2, 0, 1))[None]


def run_probe(session: ort.InferenceSession, input_name: str, tensor: np.ndarray) -> dict[str, Any]:
    output = session.run(None, {input_name: tensor})[0]
    logits = np.asarray(output).reshape(-1).astype(np.float64)
    top1_idx = int(np.argmax(logits))
    return {
        "shape": list(np.asarray(output).shape),
        "sum": float(logits.sum()),
        "min": float(logits.min()),
        "max": float(logits.max()),
        "top1_index": top1_idx,
        "top1_score": float(logits[top1_idx]),
    }


def inspect_model(model_path: Path, image_path: Path | None = None) -> dict[str, Any]:
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    input0 = session.get_inputs()[0]
    output0 = session.get_outputs()[0]
    metadata_raw = dict(session.get_modelmeta().custom_metadata_map or {})
    metadata = {k: _safe_literal(v) for k, v in metadata_raw.items()}
    names = _normalize_names(metadata.get("names"))
    imgsz = _parse_imgsz(metadata.get("imgsz"), list(input0.shape))

    result: dict[str, Any] = {
        "model_path": str(model_path.resolve()),
        "io": {
            "input_name": input0.name,
            "input_shape": list(input0.shape),
            "input_type": input0.type,
            "output_name": output0.name,
            "output_shape": list(output0.shape),
            "output_type": output0.type,
        },
        "metadata_raw": metadata_raw,
        "metadata_parsed": metadata,
        "cpp_recommended_config": {
            "task": metadata.get("task", "classify"),
            "imgsz_hw": imgsz,
            "layout": "NCHW",
            "color_order": "RGB",
            "pixel_range": "[0, 1]",
            "normalize_mean": [0.0, 0.0, 0.0],
            "normalize_std": [1.0, 1.0, 1.0],
            "resize_rule": "short_edge_to_target_then_center_crop",
            "resize_long_edge_rounding": "floor(int(target * long / short))",
            "center_crop_rounding": "round((resized - target) / 2)",
            "interpolation_hint": "PIL.BILINEAR in YOLO; C++ approximate: INTER_AREA when downsample else INTER_LINEAR",
            "softmax_hint": "Do not add softmax again if model output already sums close to 1.",
            "class_names": names,
        },
    }

    if image_path is not None:
        buf = np.fromfile(str(image_path), dtype=np.uint8)
        arr_bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if arr_bgr is None:
            raise RuntimeError(f"Failed to read image: {image_path}")

        target_h, target_w = imgsz
        cpp_tensor = preprocess_cpp_style(arr_bgr, target_h, target_w)
        yolo_tensor = preprocess_ultralytics_reference(arr_bgr, target_h, target_w)
        probe_cpp = run_probe(session, input0.name, cpp_tensor)
        probe_yolo = run_probe(session, input0.name, yolo_tensor)

        result["probe"] = {
            "image_path": str(image_path.resolve()),
            "cpp_style": probe_cpp,
            "ultralytics_reference_style": probe_yolo,
            "top1_same": probe_cpp["top1_index"] == probe_yolo["top1_index"],
            "top1_score_abs_diff": abs(probe_cpp["top1_score"] - probe_yolo["top1_score"]),
        }

    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect Ultralytics ONNX metadata and export C++ inference parameters."
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/cat_vs_dog/best.onnx"),
        help="Path to ONNX model.",
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=None,
        help="Optional image path for probe inference comparison.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output JSON path (default: <model>.infer_params.json).",
    )
    parser.add_argument(
        "--print-only",
        action="store_true",
        help="Only print JSON, do not write file.",
    )
    args = parser.parse_args()

    payload = inspect_model(args.model, args.image)
    output_text = json.dumps(payload, ensure_ascii=False, indent=2)
    print(output_text)

    if args.print_only:
        return

    out_path = args.out or args.model.with_suffix(".infer_params.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(output_text + "\n", encoding="utf-8")
    print(f"\nSaved: {out_path.resolve()}")


if __name__ == "__main__":
    main()
