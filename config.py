import os
import torch

DB_PATH = "stroke_history.csv"
DB_DICOM_PATH = "stroke_history_dicom.csv"
FONT_PATH = "DejaVuSans.ttf"
COLUMNS = ["ID", "Снимок", "Дата", "Время", "Модель", "Вердикт",
           "Полушарие", "Плотность (HU)", "Площадь", "Достоверность", "Скорость"]
MAX_HISTORY = 100

allowed_models = ["stroke_model.pth", "stroke_model_best.pth"]
model_paths = {f"📌 {name}": name for name in allowed_models if os.path.exists(name)}
if not model_paths:
    model_paths["❌ Модели не найдены"] = None

IMPORTANT_DICOM_TAGS = {
    "SliceThickness": "Толщина среза (мм)",
    "PixelSpacing": "Размер пикселя (мм)",
    "RescaleIntercept": "Перехват шкалы (HU)",
    "RescaleSlope": "Наклон шкалы",
    "WindowCenter": "Центр окна (яркость)",
    "WindowWidth": "Ширина окна (контраст)",
    "Rows": "Высота (пиксели)",
    "Columns": "Ширина (пиксели)"
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')