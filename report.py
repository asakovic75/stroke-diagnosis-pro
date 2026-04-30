# Генерация PDF-отчётов, построение диагностических диаграмм, сохранение истории в CSV

import os
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import io
from PIL import Image
from fpdf import FPDF
from fpdf.enums import XPos, YPos
from config import FONT_PATH, IMPORTANT_DICOM_TAGS, DB_PATH, COLUMNS, MAX_HISTORY

# ---------- Работа с историей (CSV) ----------
history_list = []

def load_database():
    global history_list
    if os.path.exists(DB_PATH):
        try:
            df = pd.read_csv(DB_PATH)
            if len(df.columns) == len(COLUMNS):
                df['ID'] = pd.to_numeric(df['ID'])
                history_list = df.values.tolist()
                return history_list
        except:
            pass
    history_list = []
    return history_list

def save_history(record):
    global history_list
    history_list.insert(0, record)
    if len(history_list) > MAX_HISTORY:
        history_list = history_list[:MAX_HISTORY]
    df_history = pd.DataFrame(history_list, columns=COLUMNS)
    df_history.to_csv(DB_PATH, index=False)

# Загружаем историю при импорте
load_database()

# ---------- Генератор PDF-отчётов ----------
def generate_report_universal(results_list, output_name="Diagnosis_Report.pdf", is_batch=False):
    """
    Создаёт PDF-отчёт для одного или нескольких исследований (раздел 2.4).
    results_list: список словарей с ключами 'orig_img', 'res_img', 'info', 'meta'
    """
    pdf = FPDF()
    has_font = os.path.exists(FONT_PATH)
    if has_font:
        try:
            pdf.add_font("DejaVu", "", FONT_PATH)
            pdf.add_font("DejaVu", "B", FONT_PATH)
        except:
            has_font = False
    disclaimer = "ВНИМАНИЕ: Данный отчет сформирован системой ИИ. Он носит справочный характер и не является диагнозом."

    for idx, item in enumerate(results_list):
        orig = item['orig_img']
        res = item['res_img']
        info = item['info']
        meta = item.get('meta', {})
        pdf.add_page()

        if has_font:
            pdf.set_font("DejaVu", "B", 20)
            pdf.cell(0, 15, "МЕДИЦИНСКИЙ ОТЧЕТ АНАЛИЗА КТ", align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.set_font("DejaVu", "", 10)
            if is_batch:
                pdf.cell(0, 8, f"Файл #{idx+1}: {info['filename']}", align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            else:
                pdf.cell(0, 8, f"ID Пациента: {info['p_id']} | Файл: {info['filename']}", align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.cell(0, 8, f"Дата: {info['date']} | Время: {info['time']}", align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.ln(5)
            pdf.set_font("DejaVu", "B", 14)
            pdf.cell(0, 10, "1. РЕЗУЛЬТАТЫ ОБСЛЕДОВАНИЯ:", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.set_font("DejaVu", "", 12)

            if info['verdict_ru'] == "ИНСУЛЬТ":
                pdf.set_text_color(255, 0, 0)
            pdf.cell(0, 8, f"- Заключение: {info['verdict_ru']}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.set_text_color(0, 0, 0)
            pdf.cell(0, 8, f"- Плотность очага: {info.get('hu', 'Н/Д')}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.cell(0, 8, f"- Локализация: {info['side_ru']}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.cell(0, 8, f"- Площадь поражения: {info['area']}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.cell(0, 8, f"- Скорость анализа: {info['speed']} мс", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.cell(0, 8, f"- Уверенность системы: {info['conf']}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.cell(0, 8, f"- Модель: {info['model']}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

            if meta:
                pdf.ln(5)
                pdf.set_font("DejaVu", "B", 12)
                pdf.cell(0, 10, "2. ТЕХНИЧЕСКИЕ ПАРАМЕТРЫ (DICOM):", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
                pdf.set_font("DejaVu", "", 10)
                for tag, desc in IMPORTANT_DICOM_TAGS.items():
                    val = meta.get(tag, "Н/Д")
                    pdf.cell(0, 6, f"• {desc}: {val}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

        # Временные файлы для изображений
        temp_o = f"o_tmp_{idx}.jpg"
        temp_r = f"r_tmp_{idx}.jpg"
        cv2.imwrite(temp_o, cv2.cvtColor(orig, cv2.COLOR_RGB2BGR))
        cv2.imwrite(temp_r, cv2.cvtColor(res, cv2.COLOR_RGB2BGR))
        img_y = 180 if meta else 140
        pdf.image(temp_o, x=15, y=img_y, w=85)
        pdf.image(temp_r, x=110, y=img_y, w=85)
        os.remove(temp_o)
        os.remove(temp_r)

        pdf.set_y(-30)
        if has_font:
            pdf.set_font("DejaVu", "", 7)
            pdf.multi_cell(0, 4, disclaimer, align="C")
        else:
            pdf.set_font("Helvetica", "", 7)
            pdf.multi_cell(0, 4, "WARNING: AI Report. This is for informational purposes only.", align="C")

    pdf.output(output_name)
    return output_name

# ---------- Статистическая аналитика (графики) ----------
def create_analytics(df):
    """Строит три диагностические диаграммы (раздел 2.4)."""
    plt.close('all')
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    verdict_counts = df['Вердикт'].value_counts()
    colors_v = ['#4CAF50' if x == 'Норма' else '#D32F2F' for x in verdict_counts.index]
    axes[0].pie(verdict_counts, labels=verdict_counts.index, autopct='%1.1f%%', colors=colors_v)
    axes[0].set_title("Статус: Норма / Инсульт", fontsize=12, fontweight='bold')

    areas = df[df['Вердикт'] == 'Инсульт']['Площадь_Ч'].astype(float)
    if not areas.empty:
        n, bins, patches = axes[1].hist(areas, bins=10, edgecolor='black')
        for i, patch in enumerate(patches):
            bin_center = (bins[i] + bins[i+1]) / 2
            if bin_center < 0.5:
                patch.set_facecolor('#FFEB3B')
            elif bin_center < 2.0:
                patch.set_facecolor('#FB8C00')
            else:
                patch.set_facecolor('#D32F2F')
        axes[1].set_xlabel("Площадь поражения (%)")
    else:
        axes[1].text(0.5, 0.5, 'Данных нет', ha='center')
    axes[1].set_title("Тяжесть: Площадь поражения (%)", fontsize=12, fontweight='bold')

    stroke_df = df[df['Вердикт'] == 'Инсульт']
    if not stroke_df.empty:
        side_counts = stroke_df['Полушарие'].value_counts()
        short_labels = []
        for label in side_counts.index:
            side_char = "Л" if "Левое" in label else "П" if "Правое" in label else "?"
            if "ПМА" in label:
                short_labels.append(f"{side_char} ПМА")
            elif "СМА" in label:
                short_labels.append(f"{side_char} СМА")
            elif "ЗМА" in label:
                short_labels.append(f"{side_char} ЗМА")
            else:
                short_labels.append(f"{side_char}")
        axes[2].pie(side_counts, labels=short_labels, autopct='%1.1f%%',
                    colors=['#2196F3', '#00BCD4', '#4CAF50'])
    else:
        axes[2].text(0.5, 0.5, 'Инсультов нет', ha='center')
    axes[2].set_title("Локализация (полушарие и бассейн)", fontsize=12, fontweight='bold')

    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png', dpi=150)
    buf.seek(0)
    plt.close()
    return Image.open(buf)