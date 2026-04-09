"""
YOLO 分类推理工具 (带 GUI)

功能:
1. 选择 YOLO 分类模型 (.pt / .onnx / .torchscript)
2. 选择单张图片进行推理
3. 选择整个文件夹批量推理
4. 显示分类结果和置信度
5. 优先读取模型同目录下的 labels.txt 作为类别名
"""

from __future__ import annotations

import os
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageTk
from ultralytics import YOLO


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class InferenceApp:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLO 分类推理工具")
        self.root.geometry("900x700")
        self.root.resizable(True, True)

        self.model = None
        self.model_path = ""
        self.class_names: list[str] = []
        self.current_image = None
        self.photo = None

        self._build_ui()

    def _build_ui(self):
        model_frame = ttk.LabelFrame(self.root, text="模型", padding=10)
        model_frame.pack(fill="x", padx=10, pady=5)

        self.model_var = tk.StringVar(value="未选择模型")
        ttk.Label(model_frame, textvariable=self.model_var).pack(
            side="left", padx=(0, 10)
        )
        ttk.Button(model_frame, text="选择模型", command=self._select_model).pack(
            side="left"
        )

        input_frame = ttk.LabelFrame(self.root, text="输入", padding=10)
        input_frame.pack(fill="x", padx=10, pady=5)

        btn_frame = ttk.Frame(input_frame)
        btn_frame.pack(fill="x")

        ttk.Button(btn_frame, text="选择图片", command=self._select_image).pack(
            side="left", padx=(0, 5)
        )
        ttk.Button(btn_frame, text="选择文件夹", command=self._select_folder).pack(
            side="left", padx=(0, 5)
        )
        ttk.Button(btn_frame, text="清空结果", command=self._clear_results).pack(
            side="left"
        )

        self.input_var = tk.StringVar(value="选择图片或文件夹进行推理")
        ttk.Label(input_frame, textvariable=self.input_var, foreground="gray").pack(
            fill="x", pady=(5, 0)
        )

        result_frame = ttk.LabelFrame(self.root, text="推理结果", padding=10)
        result_frame.pack(fill="both", expand=True, padx=10, pady=5)

        canvas = tk.Canvas(result_frame)
        scrollbar = ttk.Scrollbar(result_frame, orient="vertical", command=canvas.yview)
        self.result_inner = ttk.Frame(canvas)

        self.result_inner.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all")),
        )

        canvas.create_window((0, 0), window=self.result_inner, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        self.status_var = tk.StringVar(value="就绪")
        ttk.Label(
            self.root, textvariable=self.status_var, relief="sunken", anchor="w"
        ).pack(fill="x", padx=10, pady=5)

    def _select_model(self):
        path = filedialog.askopenfilename(
            title="选择模型文件",
            filetypes=[
                ("模型文件", "*.pt *.onnx *.torchscript"),
                ("所有文件", "*.*"),
            ],
        )
        if not path:
            return

        self.status_var.set("正在加载模型...")
        self.root.update()

        try:
            if path.lower().endswith(".onnx"):
                self.model = YOLO(path, task="classify")
            else:
                self.model = YOLO(path)

            self.model_path = path
            self.class_names = self._load_class_names(path)

            model_name = os.path.basename(path)
            if self.class_names:
                self.model_var.set(f"已加载: {model_name} ({len(self.class_names)} 类)")
            else:
                self.model_var.set(f"已加载: {model_name}")

            self.status_var.set("模型加载成功")
        except Exception as e:
            self.model = None
            self.model_path = ""
            self.class_names = []
            self.model_var.set("加载失败")
            self.status_var.set(f"模型加载失败: {e}")
            messagebox.showerror("错误", f"无法加载模型:\n{e}")

    def _select_image(self):
        if not self.model:
            messagebox.showwarning("提示", "请先选择模型")
            return

        path = filedialog.askopenfilename(
            title="选择图片",
            filetypes=[
                ("图片", "*.jpg *.jpeg *.png *.bmp *.webp"),
                ("所有文件", "*.*"),
            ],
        )
        if not path:
            return

        self._clear_results()
        self.input_var.set(f"图片: {path}")
        self.status_var.set("正在推理...")
        self.root.update()

        try:
            results = self.model(path, imgsz=224)
            self._show_single_result(path, results[0])
            self.status_var.set("推理完成")
        except Exception as e:
            self.status_var.set(f"推理失败: {e}")
            messagebox.showerror("错误", f"推理失败:\n{e}")

    def _select_folder(self):
        if not self.model:
            messagebox.showwarning("提示", "请先选择模型")
            return

        folder = filedialog.askdirectory(title="选择图片文件夹")
        if not folder:
            return

        images = [
            str(path)
            for path in sorted(Path(folder).rglob("*"))
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        ]

        if not images:
            messagebox.showinfo("提示", "文件夹中没有找到图片")
            return

        self._clear_results()
        self.input_var.set(f"文件夹: {folder} ({len(images)} 张图片)")
        self.status_var.set(f"正在批量推理 {len(images)} 张图片...")
        self.root.update()

        try:
            try:
                results = self.model(images, imgsz=224)
                for i, (img_path, result) in enumerate(zip(images, results)):
                    self._show_single_result(img_path, result)
                    self.status_var.set(f"进度: {i + 1}/{len(images)}")
                    self.root.update()
                self.status_var.set(f"批量推理完成: {len(images)} 张图片")
            except Exception as batch_err:
                self.status_var.set(f"批量推理失败，尝试逐图推理: {batch_err}")
                self.root.update()
                for i, img_path in enumerate(images):
                    try:
                        per_img = self.model(img_path, imgsz=224)
                        self._show_single_result(img_path, per_img[0])
                        self.status_var.set(f"进度: {i + 1}/{len(images)}")
                        self.root.update()
                    except Exception as e:
                        self.status_var.set(f"推理失败: {e}")
                        self.root.update()
        except Exception as e:
            self.status_var.set(f"推理失败: {e}")
            messagebox.showerror("错误", f"推理失败:\n{e}")

    def _show_single_result(self, img_path, result):
        row = ttk.Frame(self.result_inner)
        row.pack(fill="x", pady=5)

        thumb_frame = ttk.Frame(row, width=150, height=150)
        thumb_frame.pack(side="left", padx=(0, 10))
        thumb_frame.pack_propagate(False)

        try:
            img = Image.open(img_path)
            img.thumbnail((150, 150))
            photo = ImageTk.PhotoImage(img)
            lbl = ttk.Label(thumb_frame, image=photo)
            lbl.image = photo
            lbl.pack()
        except Exception:
            ttk.Label(thumb_frame, text="无法加载").pack()

        info_frame = ttk.Frame(row)
        info_frame.pack(side="left", fill="x", expand=True)

        ttk.Label(
            info_frame,
            text=os.path.basename(img_path),
            font=("Microsoft YaHei", 10, "bold"),
        ).pack(anchor="w")
        ttk.Label(
            info_frame, text=img_path, foreground="gray", font=("Consolas", 8)
        ).pack(anchor="w")

        probs = result.probs
        top1_idx = probs.top1
        top1_conf = probs.top1conf
        top5 = probs.top5
        names = result.names

        top1_name = self._resolve_class_name(top1_idx, names)
        display_name = self._format_label(top1_name)

        result_text = f"预测: {display_name} (置信度: {top1_conf:.1%})"
        ttk.Label(
            info_frame,
            text=result_text,
            font=("Microsoft YaHei", 11),
            foreground="#1a73e8",
        ).pack(anchor="w", pady=(5, 0))

        if len(top5) > 1:
            detail_lines = []
            for idx, conf in zip(top5, probs.top5conf):
                name = self._resolve_class_name(idx, names)
                detail_lines.append(f"  {self._format_label(name)}: {conf:.1%}")
            ttk.Label(
                info_frame, text="\n".join(detail_lines), font=("Consolas", 9)
            ).pack(anchor="w")

        ttk.Separator(self.result_inner, orient="horizontal").pack(fill="x", pady=5)

    def _clear_results(self):
        for widget in self.result_inner.winfo_children():
            widget.destroy()

    def _resolve_class_name(self, index, result_names):
        if 0 <= index < len(self.class_names):
            return self.class_names[index]

        if isinstance(result_names, dict):
            return result_names.get(index, str(index))

        if isinstance(result_names, (list, tuple)) and 0 <= index < len(result_names):
            return result_names[index]

        return str(index)

    @staticmethod
    def _load_class_names(model_path: str) -> list[str]:
        model_dir = Path(model_path).resolve().parent
        for filename in ("labels.txt", "class_names.txt"):
            labels_path = model_dir / filename
            if not labels_path.is_file():
                continue
            lines = [
                line.strip()
                for line in labels_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            if lines:
                return lines
        return []

    @staticmethod
    def _format_label(name):
        return str(name).strip()


def main():
    root = tk.Tk()
    InferenceApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
