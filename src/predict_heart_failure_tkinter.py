from __future__ import annotations

import argparse
import os
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

from pyspark.ml import PipelineModel

from heart_failure_pyspark import FEATURE_COLUMNS, create_spark
from train_heart_failure_model import DEFAULT_MODEL_PATH


NUMERIC_DEFAULTS = {
    "age": "60",
    "creatinine_phosphokinase": "250",
    "ejection_fraction": "38",
    "platelets": "265000",
    "serum_creatinine": "1.1",
    "serum_sodium": "136",
    "time": "120",
}

BINARY_OPTIONS = {
    "anaemia": {"Không": 0, "Có": 1},
    "diabetes": {"Không": 0, "Có": 1},
    "high_blood_pressure": {"Không": 0, "Có": 1},
    "sex": {"Nữ": 0, "Nam": 1},
    "smoking": {"Không": 0, "Có": 1},
}

FIELD_LABELS = {
    "age": "Tuổi",
    "anaemia": "Thiếu máu",
    "creatinine_phosphokinase": "Creatinine phosphokinase",
    "diabetes": "Tiểu đường",
    "ejection_fraction": "Ejection fraction",
    "high_blood_pressure": "Cao huyết áp",
    "platelets": "Platelets",
    "serum_creatinine": "Serum creatinine",
    "serum_sodium": "Serum sodium",
    "sex": "Giới tính",
    "smoking": "Hút thuốc",
    "time": "Thời gian theo dõi",
}

FIELD_UNITS = {
    "age": "years",
    "creatinine_phosphokinase": "mcg/L",
    "ejection_fraction": "%",
    "platelets": "kiloplatelets/mL",
    "serum_creatinine": "mg/dL",
    "serum_sodium": "mEq/L",
    "time": "days",
}

FIELD_GROUPS = [
    ("Thông tin cơ bản", ["age", "sex", "time"]),
    (
        "Tiền sử và thói quen",
        ["anaemia", "diabetes", "high_blood_pressure", "smoking"],
    ),
    (
        "Chỉ số lâm sàng",
        [
            "creatinine_phosphokinase",
            "ejection_fraction",
            "platelets",
            "serum_creatinine",
            "serum_sodium",
        ],
    ),
]

APP_BG = "#eef3f7"
CARD_BG = "#ffffff"
TEXT = "#17212b"
MUTED = "#657282"
PRIMARY = "#2563eb"
LOW = "#15803d"
MEDIUM = "#b45309"
HIGH = "#b91c1c"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Form Tkinter dự đoán suy tim.")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_PATH,
        help="Thư mục Spark ML PipelineModel đã train.",
    )
    return parser.parse_args()


class HeartFailureApp:
    def __init__(self, root: tk.Tk, model_path: str) -> None:
        self.root = root
        self.root.title("Dự đoán Suy tim")
        self.root.geometry("1040x700")
        self.root.minsize(940, 640)
        self.root.configure(bg=APP_BG)

        os.environ.setdefault("SPARK_LOCAL_IP", "127.0.0.1")
        self.spark = create_spark()
        self.model = PipelineModel.load(model_path)
        self.inputs: dict[str, tk.StringVar] = {}
        self.result_status = tk.StringVar(value="Chưa có dự đoán")
        self.result_probability = tk.StringVar(value="--")
        self.result_survival = tk.StringVar(value="--")
        self.result_detail = tk.StringVar(value="Mô hình đã sẵn sàng.")

        self._configure_styles()
        self._build_layout(model_path)
        self.root.protocol("WM_DELETE_WINDOW", self.close)

    def _configure_styles(self) -> None:
        style = ttk.Style()
        style.theme_use("clam")
        style.configure(
            "App.TEntry",
            fieldbackground="#f8fafc",
            bordercolor="#cbd5e1",
            lightcolor="#cbd5e1",
            darkcolor="#cbd5e1",
            padding=8,
        )
        style.configure(
            "App.TCombobox",
            fieldbackground="#f8fafc",
            background="#f8fafc",
            bordercolor="#cbd5e1",
            arrowcolor=PRIMARY,
            padding=8,
        )
        style.configure(
            "Primary.TButton",
            background=PRIMARY,
            foreground="#ffffff",
            borderwidth=0,
            focusthickness=0,
            padding=(18, 10),
            font=("Arial", 12, "bold"),
        )
        style.map("Primary.TButton", background=[("active", "#1d4ed8")])
        style.configure(
            "Secondary.TButton",
            background="#e2e8f0",
            foreground=TEXT,
            borderwidth=0,
            padding=(16, 10),
            font=("Arial", 12),
        )
        style.map("Secondary.TButton", background=[("active", "#cbd5e1")])

    def _build_layout(self, model_path: str) -> None:
        container = tk.Frame(self.root, bg=APP_BG, padx=24, pady=22)
        container.pack(fill="both", expand=True)

        header = tk.Frame(container, bg=APP_BG)
        header.pack(fill="x", pady=(0, 18))

        title = tk.Label(
            header,
            text="Dự đoán nguy cơ tử vong do suy tim",
            bg=APP_BG,
            fg=TEXT,
            font=("Arial", 24, "bold"),
        )
        title.pack(anchor="w")

        model_label = tk.Label(
            header,
            text=f"Model: {model_path}",
            bg=APP_BG,
            fg=MUTED,
            font=("Arial", 12),
        )
        model_label.pack(anchor="w", pady=(5, 0))

        body = tk.Frame(container, bg=APP_BG)
        body.pack(fill="both", expand=True)
        body.columnconfigure(0, weight=3)
        body.columnconfigure(1, weight=2)
        body.rowconfigure(0, weight=1)

        form_panel = self._card(body, padx=18, pady=16)
        form_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 16))
        form_panel.columnconfigure(0, weight=1)
        form_panel.rowconfigure(0, weight=1)

        form_scroll_host = tk.Frame(form_panel, bg=CARD_BG)
        form_scroll_host.grid(row=0, column=0, sticky="nsew")
        form_scroll_host.columnconfigure(0, weight=1)
        form_scroll_host.rowconfigure(0, weight=1)

        canvas = tk.Canvas(
            form_scroll_host,
            bg=CARD_BG,
            highlightthickness=0,
            bd=0,
        )
        scrollbar = ttk.Scrollbar(
            form_scroll_host,
            orient="vertical",
            command=canvas.yview,
        )
        scroll_content = tk.Frame(canvas, bg=CARD_BG)

        scroll_content.bind(
            "<Configure>",
            lambda event: canvas.configure(scrollregion=canvas.bbox("all")),
        )
        canvas.bind(
            "<Configure>",
            lambda event: canvas.itemconfigure("scroll_frame", width=event.width),
        )
        canvas.create_window(
            (0, 0),
            window=scroll_content,
            anchor="nw",
            tags="scroll_frame",
        )
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns", padx=(10, 0))

        for group_title, fields in FIELD_GROUPS:
            self._build_group(scroll_content, group_title, fields)

        button_row = tk.Frame(form_panel, bg=CARD_BG)
        button_row.grid(row=1, column=0, sticky="ew", pady=(14, 0))

        predict_button = ttk.Button(
            button_row,
            text="Dự đoán",
            style="Primary.TButton",
            command=self.predict,
        )
        predict_button.pack(side="left")

        reset_button = ttk.Button(
            button_row,
            text="Nhập lại",
            style="Secondary.TButton",
            command=self.reset,
        )
        reset_button.pack(side="left", padx=(10, 0))

        self._build_result_panel(body)

    def _card(self, parent: tk.Widget, padx: int = 14, pady: int = 12) -> tk.Frame:
        return tk.Frame(
            parent,
            bg=CARD_BG,
            highlightbackground="#dbe3ec",
            highlightthickness=1,
            padx=padx,
            pady=pady,
        )

    def _build_group(
        self, parent: tk.Widget, group_title: str, fields: list[str]
    ) -> None:
        group = tk.Frame(parent, bg=CARD_BG)
        group.pack(fill="x", pady=(0, 16))

        title = tk.Label(
            group,
            text=group_title,
            bg=CARD_BG,
            fg=TEXT,
            font=("Arial", 14, "bold"),
        )
        title.pack(anchor="w", pady=(0, 8))

        grid = tk.Frame(group, bg=CARD_BG)
        grid.pack(fill="x")
        grid.columnconfigure(0, weight=1)
        grid.columnconfigure(1, weight=1)
        grid.columnconfigure(2, weight=1)

        for index, field in enumerate(fields):
            cell = tk.Frame(grid, bg=CARD_BG)
            cell.grid(
                row=index // 3,
                column=index % 3,
                sticky="ew",
                padx=(0, 12) if index % 3 != 2 else (0, 0),
                pady=(0, 10),
            )

            label_text = FIELD_LABELS.get(field, field)
            if field in FIELD_UNITS:
                label_text = f"{label_text} ({FIELD_UNITS[field]})"

            label = tk.Label(
                cell,
                text=label_text,
                bg=CARD_BG,
                fg=MUTED,
                font=("Arial", 11),
            )
            label.pack(anchor="w", pady=(0, 4))

            value = tk.StringVar()
            self.inputs[field] = value

            if field in BINARY_OPTIONS:
                options = list(BINARY_OPTIONS[field].keys())
                value.set(options[0])
                widget = ttk.Combobox(
                    cell,
                    textvariable=value,
                    values=options,
                    state="readonly",
                    style="App.TCombobox",
                )
            else:
                value.set(NUMERIC_DEFAULTS.get(field, "0"))
                widget = ttk.Entry(cell, textvariable=value, style="App.TEntry")

            widget.pack(fill="x")

    def _build_result_panel(self, parent: tk.Widget) -> None:
        panel = self._card(parent, padx=20, pady=18)
        panel.grid(row=0, column=1, sticky="nsew")

        heading = tk.Label(
            panel,
            text="Kết quả dự đoán",
            bg=CARD_BG,
            fg=TEXT,
            font=("Arial", 18, "bold"),
        )
        heading.pack(anchor="w")

        explainer = tk.Label(
            panel,
            text=(
                "Tỷ lệ % ở đây là xác suất mô hình dự đoán bệnh nhân thuộc "
                "nhóm DEATH_EVENT = 1, tức là nhóm tử vong trong dataset."
            ),
            bg=CARD_BG,
            fg=MUTED,
            font=("Arial", 11),
            wraplength=330,
            justify="left",
        )
        explainer.pack(anchor="w", pady=(8, 0))

        self.badge = tk.Label(
            panel,
            textvariable=self.result_status,
            bg="#e2e8f0",
            fg=TEXT,
            font=("Arial", 13, "bold"),
            padx=12,
            pady=8,
        )
        self.badge.pack(anchor="w", pady=(16, 18))

        self.gauge = tk.Canvas(
            panel,
            width=260,
            height=160,
            bg=CARD_BG,
            highlightthickness=0,
        )
        self.gauge.pack(fill="x")
        self._draw_gauge(0.0, "#cbd5e1")

        probability_title = tk.Label(
            panel,
            text="Xác suất tử vong dự đoán",
            bg=CARD_BG,
            fg=MUTED,
            font=("Arial", 11),
        )
        probability_title.pack(anchor="w", pady=(8, 0))

        probability = tk.Label(
            panel,
            textvariable=self.result_probability,
            bg=CARD_BG,
            fg=TEXT,
            font=("Arial", 34, "bold"),
        )
        probability.pack(anchor="w", pady=(8, 0))

        survival = tk.Label(
            panel,
            textvariable=self.result_survival,
            bg=CARD_BG,
            fg=TEXT,
            font=("Arial", 14, "bold"),
        )
        survival.pack(anchor="w", pady=(10, 0))

        detail = tk.Label(
            panel,
            textvariable=self.result_detail,
            bg=CARD_BG,
            fg=MUTED,
            font=("Arial", 12),
            wraplength=330,
            justify="left",
        )
        detail.pack(anchor="w", pady=(8, 0))

    def _draw_gauge(self, probability: float, color: str) -> None:
        self.gauge.delete("all")
        width = 260
        x0, y0, x1, y1 = 18, 22, width - 18, 240
        self.gauge.create_arc(
            x0,
            y0,
            x1,
            y1,
            start=0,
            extent=180,
            style="arc",
            outline="#e2e8f0",
            width=18,
        )
        self.gauge.create_arc(
            x0,
            y0,
            x1,
            y1,
            start=180,
            extent=-180 * probability,
            style="arc",
            outline=color,
            width=18,
        )
        self.gauge.create_text(
            width // 2,
            134,
            text="Xác suất mô hình dự đoán tử vong",
            fill=MUTED,
            font=("Arial", 11),
        )

    def _set_result(self, probability: float, predicted_class: int) -> None:
        survival_probability = 1.0 - probability

        if probability >= 0.65:
            color = HIGH
            status = "Nguy cơ cao"
        elif probability >= 0.35:
            color = MEDIUM
            status = "Cần theo dõi"
        else:
            color = LOW
            status = "Nguy cơ thấp"

        if predicted_class == 1 and probability < 0.65:
            status = "Có nguy cơ tử vong"

        self.result_status.set(status)
        self.result_probability.set(f"{probability:.2%}")
        self.result_survival.set(
            f"Xác suất sống sót dự đoán: {survival_probability:.2%}"
        )
        self.result_detail.set(
            f"Mô hình xếp mẫu này vào {'DEATH_EVENT = 1' if predicted_class == 1 else 'DEATH_EVENT = 0'}. "
            f"Nói cách khác, tỷ lệ {probability:.2%} là xác suất tử vong dự đoán, "
            f"còn {survival_probability:.2%} là xác suất không tử vong dự đoán."
        )
        self.badge.configure(bg=color, fg="#ffffff")
        self._draw_gauge(probability, color)

    def reset(self) -> None:
        for field, variable in self.inputs.items():
            if field in BINARY_OPTIONS:
                variable.set(next(iter(BINARY_OPTIONS[field])))
            else:
                variable.set(NUMERIC_DEFAULTS.get(field, "0"))
        self.result_status.set("Chưa có dự đoán")
        self.result_probability.set("--")
        self.result_survival.set("--")
        self.result_detail.set("Mô hình đã sẵn sàng.")
        self.badge.configure(bg="#e2e8f0", fg=TEXT)
        self._draw_gauge(0.0, "#cbd5e1")

    def _read_form(self) -> dict[str, float | int]:
        row: dict[str, float | int] = {}
        for field in FEATURE_COLUMNS:
            raw_value = self.inputs[field].get().strip()
            if field in BINARY_OPTIONS:
                row[field] = BINARY_OPTIONS[field][raw_value]
            else:
                try:
                    row[field] = float(raw_value)
                except ValueError as exc:
                    raise ValueError(
                        f"Giá trị '{FIELD_LABELS.get(field, field)}' phải là số."
                    ) from exc
        return row

    def predict(self) -> None:
        try:
            row = self._read_form()
            prediction_df = self.model.transform(self.spark.createDataFrame([row]))
            prediction = prediction_df.select("prediction", "probability").first()

            death_probability = float(prediction["probability"][1])
            predicted_class = int(prediction["prediction"])
            self._set_result(death_probability, predicted_class)
        except Exception as exc:
            messagebox.showerror("Không thể dự đoán", str(exc))

    def close(self) -> None:
        self.spark.stop()
        self.root.destroy()


def main() -> None:
    args = parse_args()
    model_path = Path(args.model)
    if not model_path.exists():
        raise SystemExit(
            "Chưa có model. Hãy train trước bằng lệnh: "
            "python src/train_heart_failure_model.py"
        )

    root = tk.Tk()
    HeartFailureApp(root, str(model_path))
    root.mainloop()


if __name__ == "__main__":
    main()
