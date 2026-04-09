"""
YOLO 分类推理工具 (带 GUI)

功能:
1. 选择 YOLO 分类模型 (.pt / .onnx / .torchscript)
2. 选择单张图片或整个文件夹，但不自动推理
3. 支持推理当前图片或批量推理全部
4. 批量结果采用分页列表，避免大批量图片时控件和内存爆炸
5. 优先读取模型同目录下的 labels.txt 作为类别名
"""

from __future__ import annotations

import tkinter as tk
from collections import Counter
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageTk
from ultralytics import YOLO


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_PAGE_SIZE = 20
MIN_PAGE_SIZE = 1
MAX_PAGE_SIZE = 200
PREVIEW_MIN_HEIGHT = 320


class InferenceApp:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLO 图像分类推理工具")
        self.root.geometry("1200x880")
        self.root.minsize(1000, 760)

        self.model = None
        self.model_path = ""
        self.class_names: list[str] = []

        self.image_paths: list[str] = []
        self.results: list[dict | None] = []
        self.current_index = -1
        self.current_page = 0

        self.batch_running = False
        self.batch_job_id = 0
        self.batch_next_index = 0

        self.preview_photo = None
        self.preview_refresh_job = None

        self.page_size_var = tk.IntVar(value=DEFAULT_PAGE_SIZE)
        self.model_var = tk.StringVar(value="未选择模型")
        self.status_var = tk.StringVar(value="状态: 就绪")
        self.image_info_var = tk.StringVar(value="")
        self.result_var = tk.StringVar(value="结果: --")
        self.confidence_var = tk.StringVar(value="置信度: --")
        self.page_var = tk.StringVar(value="第 0/0 页")
        self.input_var = tk.StringVar(value="选择图片或文件夹后，再决定是否推理")

        self._build_ui()
        self._update_button_states()

    def _build_ui(self):
        container = ttk.Frame(self.root, padding=12)
        container.pack(fill="both", expand=True)

        top_row = ttk.Frame(container)
        top_row.pack(fill="x", pady=(0, 8))

        self.select_model_btn = ttk.Button(
            top_row, text="选择模型", command=self._select_model
        )
        self.select_model_btn.pack(side="left", padx=(0, 8))

        self.load_image_btn = ttk.Button(
            top_row, text="加载图片", command=self._select_image
        )
        self.load_image_btn.pack(side="left", padx=(0, 8))

        self.load_folder_btn = ttk.Button(
            top_row, text="加载文件夹", command=self._select_folder
        )
        self.load_folder_btn.pack(side="left", padx=(0, 8))

        ttk.Label(top_row, textvariable=self.input_var, foreground="#666666").pack(
            side="left", padx=(12, 0)
        )

        action_row = ttk.Frame(container)
        action_row.pack(fill="x", pady=(0, 8))

        self.prev_btn = ttk.Button(action_row, text="◀ 上一张", command=self._prev_image)
        self.prev_btn.pack(side="left", padx=(0, 8))

        self.infer_current_btn = ttk.Button(
            action_row, text="推理当前图片", command=self._run_inference_current
        )
        self.infer_current_btn.pack(side="left", padx=(0, 8))

        self.batch_infer_btn = ttk.Button(
            action_row, text="批量推理全部", command=self._run_batch_inference
        )
        self.batch_infer_btn.pack(side="left", padx=(0, 8))

        self.next_btn = ttk.Button(action_row, text="下一张 ▶", command=self._next_image)
        self.next_btn.pack(side="left", padx=(0, 12))

        ttk.Label(action_row, textvariable=self.image_info_var, foreground="#666666").pack(
            side="left"
        )

        option_row = ttk.Frame(container)
        option_row.pack(fill="x", pady=(0, 8))

        ttk.Label(option_row, text="每页显示:").pack(side="left")
        self.page_size_spin = tk.Spinbox(
            option_row,
            from_=MIN_PAGE_SIZE,
            to=MAX_PAGE_SIZE,
            width=6,
            textvariable=self.page_size_var,
            command=self._apply_page_size,
        )
        self.page_size_spin.pack(side="left", padx=(6, 12))
        self.page_size_spin.bind("<Return>", self._apply_page_size)
        self.page_size_spin.bind("<FocusOut>", self._apply_page_size)

        self.page_prev_btn = ttk.Button(
            option_row, text="上一页", command=lambda: self._change_page(-1)
        )
        self.page_prev_btn.pack(side="left", padx=(0, 8))

        self.page_next_btn = ttk.Button(
            option_row, text="下一页", command=lambda: self._change_page(1)
        )
        self.page_next_btn.pack(side="left", padx=(0, 8))

        ttk.Label(option_row, textvariable=self.page_var).pack(side="left", padx=(0, 12))
        ttk.Label(
            option_row,
            text="说明: 控制下方结果列表一次显示多少张已加载图片的推理结果",
            foreground="#666666",
        ).pack(side="left")

        self.model_label = ttk.Label(container, textvariable=self.model_var, foreground="#0b7a0b")
        self.model_label.pack(fill="x", pady=(0, 4))

        self.status_label = ttk.Label(container, textvariable=self.status_var, foreground="#555555")
        self.status_label.pack(fill="x", pady=(0, 8))

        preview_frame = ttk.Frame(container)
        preview_frame.pack(fill="both", expand=True, pady=(0, 8))
        preview_frame.pack_propagate(False)
        preview_frame.configure(height=420)

        self.image_label = tk.Label(
            preview_frame,
            text="未加载图片",
            anchor="center",
            relief="solid",
            borderwidth=1,
            bg="#f5f5f5",
        )
        self.image_label.pack(fill="both", expand=True)
        self.image_label.bind("<Configure>", self._schedule_preview_refresh)

        ttk.Label(
            container, textvariable=self.result_var, font=("Microsoft YaHei", 15, "bold")
        ).pack(fill="x", pady=(0, 4))
        ttk.Label(container, textvariable=self.confidence_var, font=("Microsoft YaHei", 12)).pack(
            fill="x", pady=(0, 8)
        )

        result_frame = ttk.Frame(container)
        result_frame.pack(fill="both", expand=True)

        columns = ("index", "file", "prediction", "confidence", "status")
        self.result_tree = ttk.Treeview(
            result_frame,
            columns=columns,
            show="headings",
            height=DEFAULT_PAGE_SIZE,
        )
        self.result_tree.heading("index", text="序号")
        self.result_tree.heading("file", text="文件名")
        self.result_tree.heading("prediction", text="预测结果")
        self.result_tree.heading("confidence", text="置信度")
        self.result_tree.heading("status", text="状态")

        self.result_tree.column("index", width=70, anchor="center", stretch=False)
        self.result_tree.column("file", width=320, anchor="w")
        self.result_tree.column("prediction", width=180, anchor="center")
        self.result_tree.column("confidence", width=120, anchor="center", stretch=False)
        self.result_tree.column("status", width=120, anchor="center", stretch=False)

        tree_scroll = ttk.Scrollbar(
            result_frame, orient="vertical", command=self.result_tree.yview
        )
        self.result_tree.configure(yscrollcommand=tree_scroll.set)
        self.result_tree.pack(side="left", fill="both", expand=True)
        tree_scroll.pack(side="right", fill="y")

        self.result_tree.bind("<<TreeviewSelect>>", self._handle_result_selection)

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

        self._cancel_batch_if_needed()
        self.status_var.set("状态: 正在加载模型...")
        self.root.update_idletasks()

        try:
            if path.lower().endswith(".onnx"):
                self.model = YOLO(path, task="classify")
            else:
                self.model = YOLO(path)

            self.model_path = path
            self.class_names = self._load_class_names(path)
            if self.class_names:
                self.model_var.set(
                    f"模型: {path} ({len(self.class_names)} 类)"
                )
            else:
                self.model_var.set(f"模型: {path}")
            self.status_var.set("状态: 模型加载成功")
        except Exception as exc:
            self.model = None
            self.model_path = ""
            self.class_names = []
            self.model_var.set("模型: 加载失败")
            self.status_var.set(f"状态: 模型加载失败: {exc}")
            messagebox.showerror("错误", f"无法加载模型:\n{exc}")

        self._update_button_states()

    def _select_image(self):
        path = filedialog.askopenfilename(
            title="选择图片",
            filetypes=[
                ("图片", "*.jpg *.jpeg *.png *.bmp *.webp"),
                ("所有文件", "*.*"),
            ],
        )
        if not path:
            return

        self._set_images([path], f"已加载 1 张图片: {Path(path).name}")

    def _select_folder(self):
        folder = filedialog.askdirectory(title="选择图片文件夹")
        if not folder:
            return

        images = [
            str(path)
            for path in sorted(Path(folder).rglob("*"))
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        ]

        if not images:
            messagebox.showwarning("提示", "文件夹中没有找到图片文件")
            self.status_var.set("状态: 文件夹中没有图片")
            return

        self._set_images(images, f"已递归加载 {len(images)} 张图片: {folder}")

    def _set_images(self, image_paths: list[str], status_text: str):
        self._cancel_batch_if_needed()

        self.image_paths = image_paths
        self.results = [None] * len(image_paths)
        self.current_index = 0 if image_paths else -1
        self.current_page = 0

        source_text = (
            f"已加载单张图片: {Path(image_paths[0]).name}"
            if len(image_paths) == 1
            else f"已加载文件夹，共 {len(image_paths)} 张图片"
        )
        self.input_var.set(source_text)
        self.status_var.set(f"状态: {status_text}")

        self._clear_result_labels()
        self._render_result_page()
        if self.current_index >= 0:
            self._show_current_image()
        else:
            self._clear_preview()
        self._update_button_states()

    def _run_inference_current(self):
        if not self._ensure_ready_for_inference(require_images=True):
            return

        index = self.current_index
        path = self.image_paths[index]
        self.status_var.set(f"状态: 正在推理当前图片: {Path(path).name}")
        self.root.update_idletasks()

        result = self._infer_image(path)
        self.results[index] = result

        self._display_result(index)
        self._ensure_current_page_visible()
        self._render_result_page()
        self.status_var.set(f"状态: 当前图片推理完成: {Path(path).name}")

    def _run_batch_inference(self):
        if not self._ensure_ready_for_inference(require_images=True):
            return
        if self.batch_running:
            messagebox.showinfo("提示", "批量推理正在进行中")
            return

        self.results = [None] * len(self.image_paths)
        self.batch_running = True
        self.batch_job_id += 1
        self.batch_next_index = 0

        self._render_result_page()
        self._update_button_states()
        self.status_var.set(
            f"状态: 开始批量推理，共 {len(self.image_paths)} 张图片"
        )
        self.root.after(1, lambda: self._process_batch_step(self.batch_job_id))

    def _process_batch_step(self, job_id: int):
        if not self.batch_running or job_id != self.batch_job_id:
            return

        if self.batch_next_index >= len(self.image_paths):
            self.batch_running = False
            self._update_button_states()
            self._render_result_page()
            processed, ok_count, label_counter = self._summarize_results()
            counter_text = "，".join(
                f"{label} {count} 张" for label, count in label_counter.most_common()
            )
            suffix = f"，{counter_text}" if counter_text else ""
            self.status_var.set(
                f"状态: 批量推理完成，共 {len(self.image_paths)} 张，成功 {ok_count} 张{suffix}"
            )
            return

        index = self.batch_next_index
        image_path = self.image_paths[index]
        self.status_var.set(
            f"状态: 批量推理中 {index + 1}/{len(self.image_paths)} - {Path(image_path).name}"
        )

        result = self._infer_image(image_path)
        self.results[index] = result

        if index == self.current_index:
            self._display_result(index)

        page_size = self._get_page_size()
        if index // page_size == self.current_page or index == 0 or (index + 1) % 10 == 0:
            self._render_result_page()

        self.batch_next_index += 1
        self.root.after(1, lambda: self._process_batch_step(job_id))

    def _prev_image(self):
        if self.current_index > 0:
            self.current_index -= 1
            self._ensure_current_page_visible()
            self._show_current_image()
            self._display_result(self.current_index)
            self._render_result_page()
            self._update_button_states()

    def _next_image(self):
        if self.current_index < len(self.image_paths) - 1:
            self.current_index += 1
            self._ensure_current_page_visible()
            self._show_current_image()
            self._display_result(self.current_index)
            self._render_result_page()
            self._update_button_states()

    def _change_page(self, delta: int):
        total_pages = self._get_total_pages()
        if total_pages <= 0:
            return

        new_page = max(0, min(total_pages - 1, self.current_page + delta))
        if new_page == self.current_page:
            return

        self.current_page = new_page
        page_start = self.current_page * self._get_page_size()
        page_end = min(page_start + self._get_page_size(), len(self.image_paths))
        if self.current_index < page_start or self.current_index >= page_end:
            self.current_index = page_start if page_start < page_end else -1
            if self.current_index >= 0:
                self._show_current_image()
                self._display_result(self.current_index)
        self._render_result_page()
        self._update_button_states()

    def _apply_page_size(self, _event=None):
        page_size = self._get_page_size()
        self.page_size_var.set(page_size)
        self.result_tree.configure(height=min(max(page_size, 5), 30))
        self._ensure_current_page_visible()
        self._render_result_page()
        self._update_button_states()

    def _handle_result_selection(self, _event=None):
        selected = self.result_tree.selection()
        if not selected:
            return

        index = int(selected[0])
        if index < 0 or index >= len(self.image_paths):
            return

        self.current_index = index
        self._show_current_image()
        self._display_result(index)
        self._update_button_states()

    def _render_result_page(self):
        self.result_tree.delete(*self.result_tree.get_children())

        total_pages = self._get_total_pages()
        if not self.image_paths:
            self.page_var.set("第 0/0 页")
            return

        if self.current_page >= total_pages:
            self.current_page = max(0, total_pages - 1)

        page_size = self._get_page_size()
        start = self.current_page * page_size
        end = min(start + page_size, len(self.image_paths))

        for index in range(start, end):
            file_name = Path(self.image_paths[index]).name
            result = self.results[index]

            if result is None:
                prediction = "--"
                confidence = "--"
                status = "未推理"
            elif result["status"] == "ok":
                prediction = result["display_name"]
                confidence = f"{result['confidence']:.2%}"
                status = "成功"
            else:
                prediction = "--"
                confidence = "--"
                status = "失败"

            self.result_tree.insert(
                "",
                "end",
                iid=str(index),
                values=(index + 1, file_name, prediction, confidence, status),
            )

        self.page_var.set(f"第 {self.current_page + 1}/{total_pages} 页")
        self._sync_tree_selection()

    def _sync_tree_selection(self):
        if self.current_index < 0:
            return
        current_iid = str(self.current_index)
        if current_iid in self.result_tree.get_children():
            self.result_tree.selection_set(current_iid)
            self.result_tree.focus(current_iid)
            self.result_tree.see(current_iid)

    def _show_current_image(self):
        if self.current_index < 0 or self.current_index >= len(self.image_paths):
            self._clear_preview()
            return

        image_path = self.image_paths[self.current_index]
        self.image_info_var.set(
            f"[{self.current_index + 1}/{len(self.image_paths)}] {Path(image_path).name}"
        )

        try:
            with Image.open(image_path) as image:
                image = image.convert("RGB")
                width = max(self.image_label.winfo_width(), 640)
                height = max(self.image_label.winfo_height(), PREVIEW_MIN_HEIGHT)
                preview = image.copy()
                preview.thumbnail((width - 8, height - 8))
        except Exception as exc:
            self.preview_photo = None
            self.image_label.configure(image="", text=f"图片读取失败\n{exc}")
            return

        self.preview_photo = ImageTk.PhotoImage(preview)
        self.image_label.configure(image=self.preview_photo, text="")

    def _schedule_preview_refresh(self, _event=None):
        if self.preview_refresh_job is not None:
            self.root.after_cancel(self.preview_refresh_job)
        self.preview_refresh_job = self.root.after(120, self._refresh_preview_after_resize)

    def _refresh_preview_after_resize(self):
        self.preview_refresh_job = None
        if self.current_index >= 0:
            self._show_current_image()

    def _clear_preview(self):
        self.preview_photo = None
        self.image_info_var.set("")
        self.image_label.configure(image="", text="未加载图片")

    def _display_result(self, index: int):
        if index < 0 or index >= len(self.results):
            self._clear_result_labels()
            return

        result = self.results[index]
        if result is None:
            self._clear_result_labels()
            return

        if result["status"] != "ok":
            self.result_var.set("结果: 推理失败")
            self.confidence_var.set(f"错误: {result['message']}")
            return

        self.result_var.set(f"结果: {result['display_name']}")
        self.confidence_var.set(f"置信度: {result['confidence']:.2%}")

    def _clear_result_labels(self):
        self.result_var.set("结果: --")
        self.confidence_var.set("置信度: --")

    def _infer_image(self, image_path: str) -> dict:
        try:
            result = self.model(image_path, imgsz=224, verbose=False)[0]
            probs = result.probs
            top1_idx = int(probs.top1)
            confidence = float(probs.top1conf)
            top1_name = self._resolve_class_name(top1_idx, result.names)
            display_name = self._format_label(top1_name)

            top5_details = []
            for idx, conf in zip(probs.top5, probs.top5conf):
                class_name = self._resolve_class_name(int(idx), result.names)
                top5_details.append(
                    {
                        "name": self._format_label(class_name),
                        "confidence": float(conf),
                    }
                )

            return {
                "status": "ok",
                "display_name": display_name,
                "confidence": confidence,
                "top5": top5_details,
                "path": image_path,
            }
        except Exception as exc:
            return {
                "status": "error",
                "message": str(exc),
                "path": image_path,
            }

    def _summarize_results(self) -> tuple[int, int, Counter]:
        processed = 0
        ok_count = 0
        label_counter: Counter = Counter()

        for result in self.results:
            if result is None:
                continue
            processed += 1
            if result["status"] == "ok":
                ok_count += 1
                label_counter[result["display_name"]] += 1

        return processed, ok_count, label_counter

    def _ensure_current_page_visible(self):
        if self.current_index < 0:
            self.current_page = 0
            return
        self.current_page = self.current_index // self._get_page_size()

    def _get_page_size(self) -> int:
        try:
            value = int(self.page_size_var.get())
        except (TypeError, ValueError, tk.TclError):
            value = DEFAULT_PAGE_SIZE
        return max(MIN_PAGE_SIZE, min(MAX_PAGE_SIZE, value))

    def _get_total_pages(self) -> int:
        if not self.image_paths:
            return 0
        page_size = self._get_page_size()
        return (len(self.image_paths) + page_size - 1) // page_size

    def _cancel_batch_if_needed(self):
        if self.batch_running:
            self.batch_running = False
            self.batch_job_id += 1
            self.status_var.set("状态: 已取消上一个批量推理任务")

    def _ensure_ready_for_inference(self, require_images: bool) -> bool:
        if self.model is None:
            messagebox.showwarning("提示", "请先选择模型")
            return False
        if require_images and not self.image_paths:
            messagebox.showwarning("提示", "请先加载图片或文件夹")
            return False
        if self.batch_running:
            messagebox.showwarning("提示", "批量推理进行中，请稍候")
            return False
        return True

    def _update_button_states(self):
        has_images = bool(self.image_paths)
        has_model = self.model is not None

        self.prev_btn.configure(state="normal" if self.current_index > 0 else "disabled")
        self.next_btn.configure(
            state="normal"
            if has_images and 0 <= self.current_index < len(self.image_paths) - 1
            else "disabled"
        )

        infer_state = "normal" if has_images and has_model and not self.batch_running else "disabled"
        self.infer_current_btn.configure(state=infer_state)
        self.batch_infer_btn.configure(state=infer_state)

        total_pages = self._get_total_pages()
        self.page_prev_btn.configure(
            state="normal" if total_pages > 0 and self.current_page > 0 else "disabled"
        )
        self.page_next_btn.configure(
            state="normal"
            if total_pages > 0 and self.current_page < total_pages - 1
            else "disabled"
        )

    def _resolve_class_name(self, index: int, result_names):
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
    def _format_label(name) -> str:
        return str(name).strip()


def main():
    root = tk.Tk()
    InferenceApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
