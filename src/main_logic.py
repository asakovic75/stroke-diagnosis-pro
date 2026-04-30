import os
import time
import cv2
import numpy as np
import pandas as pd
from datetime import datetime
import pytz
import pydicom
import gradio as gr
from config import DB_DICOM_PATH, COLUMNS
from segmentation import load_selected_model, core_inference
from dicom_preprocessing import dicom_to_rgb, extract_meta
from diagnosis import get_hu_analysis, get_side_and_basin, calculate_area_percent, compute_confidence
from report import generate_report_universal, create_analytics
from database import save_history, get_history_dataframe, get_max_patient_id

def overlay_mask_on_image(img_res, mask, contour_color=(255, 0, 0), thickness=2):
    res_view = img_res.copy()
    if np.sum(mask) > 0:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(res_view, contours, -1, contour_color, thickness)
    return res_view

def predict_stroke(file_path, model_key):
    if not file_path or not load_selected_model(model_key):
        gr.Warning("Модель не выбрана или не загружена")
        return [None] * 6
    if isinstance(file_path, list):
        file_path = file_path[0] if file_path else None
    if not file_path or not os.path.exists(file_path):
        return [None] * 6

    filename = os.path.basename(file_path)
    try:
        ds = pydicom.dcmread(file_path)
        input_img = dicom_to_rgb(ds)
    except Exception:
        gr.Warning("Ошибка чтения DICOM файла")
        return [None] * 6

    input_img_resized = cv2.resize(input_img, (256, 256))
    start_time = time.time()
    img_res, mask, prob = core_inference(input_img_resized)
    speed_ms = round((time.time() - start_time) * 1000, 1)

    is_stroke = np.sum(mask) > 20
    confidence_val = compute_confidence(prob, mask)
    area_percent = calculate_area_percent(mask)
    hu_info = get_hu_analysis(input_img, mask, ds) if is_stroke else "Н/Д"
    raw_side, side_ru = get_side_and_basin(mask)
    if not is_stroke:
        side_ru = "Не выявлено"
    area_str = f"{area_percent:.2f}%"
    confidence = f"{confidence_val:.1f}%"
    status_ru = "ИНСУЛЬТ" if is_stroke else "НОРМА"

    warning_html = ""
    if is_stroke and confidence_val < 70:
        warning_html = '<div style="background:#FF9800; padding:10px; border-radius:10px; margin-top:15px; text-align:center; color:#000;">⚠️ НИЗКАЯ ДОСТОВЕРНОСТЬ! Требуется перепроверка врачом.</div>'

    res_view = overlay_mask_on_image(img_res, mask)

    p_id = get_max_patient_id() + 1

    now_gr = datetime.now(pytz.timezone('Europe/Minsk'))
    info = {
        'p_id': p_id, 'filename': filename, 'model': model_key.replace("📌 ", ""),
        'side_ru': side_ru, 'conf': confidence, 'area': area_str, 'hu': hu_info,
        'verdict_ru': status_ru, 'speed': speed_ms,
        'date': now_gr.strftime("%d.%m"), 'time': now_gr.strftime("%H:%M:%S")
    }
    meta = extract_meta(ds)

    pdf_path = generate_report_universal([{'orig_img': img_res, 'res_img': res_view, 'info': info, 'meta': meta}],
                                         is_batch=False)

    record = [str(p_id), filename, info['date'], info['time'], info['model'],
              "Инсульт" if is_stroke else "Норма", side_ru, hu_info, area_str, confidence, f"{speed_ms} мс"]
    save_history(record)

    df_history = get_history_dataframe()

    color = "#D32F2F" if is_stroke else "#2E7D32"
    stats_html = f"""
    <div style="text-align: center; padding: 15px; background: linear-gradient(135deg, {color} 0%, {color} 100%); border-radius: 15px; color: white;">
        <div style="font-size: 1.8em; font-weight: bold;">{status_ru}</div>
        <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px; margin-top: 20px;">
            <div style="background: rgba(255,255,255,0.2); padding: 10px; border-radius: 10px;">
                <div>🧠 Тип патологии</div><div style="font-weight:bold;">{hu_info}</div>
            </div>
            <div style="background: rgba(255,255,255,0.2); padding: 10px; border-radius: 10px;">
                <div>📍 Локализация</div><div style="font-weight:bold;">{side_ru}</div>
            </div>
            <div style="background: rgba(255,255,255,0.2); padding: 10px; border-radius: 10px;">
                <div>📐 Площадь поражения</div><div style="font-weight:bold;">{area_str}</div>
            </div>
            <div style="background: rgba(255,255,255,0.2); padding: 10px; border-radius: 10px;">
                <div>⚡ Скорость</div><div style="font-weight:bold;">{speed_ms} мс</div>
            </div>
            <div style="background: rgba(255,255,255,0.2); padding: 10px; border-radius: 10px;">
                <div>🎯 Уверенность</div><div style="font-weight:bold;">{confidence}</div>
            </div>
            <div style="background: rgba(255,255,255,0.2); padding: 10px; border-radius: 10px;">
                <div>🔧 Модель</div><div style="font-weight:bold;">{info['model']}</div>
            </div>
        </div>
        <div style="margin-top:15px;">🆔 Пациент ID: {p_id} | 📄 {filename}</div>
        {warning_html}
    </div>
    """
    return res_view, input_img_resized, stats_html, "", df_history, pdf_path

def process_batch(files, model_key):
    if not files or not load_selected_model(model_key):
        gr.Warning("Модель не выбрана или не загружена")
        return [None] * 6

    batch_results = []
    report_items = []
    speed_log = []
    total_start_time = time.time()
    low_confidence_warning = False

    for i, f in enumerate(files):
        file_start = time.time()
        filename = os.path.basename(f.name)
        try:
            ds = pydicom.dcmread(f.name)
            img = dicom_to_rgb(ds)
            img_256 = cv2.resize(img, (256, 256))
        except Exception as e:
            print(f"Ошибка чтения DICOM {filename}: {e}")
            continue

        img_res, mask, prob = core_inference(img_256)
        file_duration = round((time.time() - file_start) * 1000, 1)
        speed_log.append(file_duration)

        is_stroke = np.sum(mask) > 20
        confidence_val = compute_confidence(prob, mask)
        area_v = calculate_area_percent(mask)
        area_str = f"{area_v:.2f}%"
        hu = get_hu_analysis(img, mask, ds) if is_stroke else "Н/Д"
        raw_side, side = get_side_and_basin(mask)
        if not is_stroke:
            side = "Не выявлено"
        confidence = f"{confidence_val:.1f}%"

        if is_stroke and confidence_val < 70:
            low_confidence_warning = True

        res_view = overlay_mask_on_image(img_res, mask)

        now = datetime.now(pytz.timezone('Europe/Minsk'))
        meta = extract_meta(ds)
        info = {
            'p_id': f"B-{i+1}",
            'filename': filename,
            'model': model_key.replace("📌 ", ""),
            'verdict_ru': "ИНСУЛЬТ" if is_stroke else "НОРМА",
            'hu': hu,
            'side_ru': side,
            'area': area_str,
            'speed': file_duration,
            'conf': confidence,
            'date': now.strftime("%d.%m"),
            'time': now.strftime("%H:%M")
        }
        report_items.append({'orig_img': img_res, 'res_img': res_view, 'info': info, 'meta': meta})

        batch_results.append({
            "ID": i+1,
            "Снимок": filename,
            "Дата": info['date'],
            "Время": info['time'],
            "Модель": info['model'],
            "Вердикт": "Инсульт" if is_stroke else "Норма",
            "Полушарие": side,
            "Плотность (HU)": hu,
            "Площадь": area_str,
            "Достоверность": confidence,
            "Скорость": f"{file_duration} мс",
            "Площадь_Ч": area_v
        })

    if not batch_results:
        gr.Warning("Не удалось обработать ни одного DICOM файла")
        return [None] * 6

    total_duration = round((time.time() - total_start_time) * 1000, 1)
    avg_speed = round(np.mean(speed_log), 1) if speed_log else 0
    min_speed = min(speed_log) if speed_log else 0
    max_speed = max(speed_log) if speed_log else 0

    df = pd.DataFrame(batch_results)
    df_display = df.drop(columns=["Площадь_Ч"])
    df_for_csv = df.drop(columns=["Площадь_Ч"]).copy()
    df_for_csv.to_csv(DB_DICOM_PATH, index=False)

    pdf_p = generate_report_universal(report_items, "Batch_Diagnosis_Report.pdf", is_batch=True)

    df_ana = df.copy()
    df_ana['Площадь_Ч'] = df_ana['Площадь_Ч'].astype(float)
    analytics_img = create_analytics(df_ana)

    warning_global = ""
    if low_confidence_warning:
        warning_global = '<div style="background:#FF9800; padding:10px; border-radius:10px; margin-top:15px; text-align:center; color:#000;">⚠️ В одном или нескольких файлах низкая достоверность! Требуется перепроверка врачом.</div>'

    stats_html = f"""
    <div style="text-align: center; padding: 15px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; color: white;">
        <div style="font-size: 1.8em; font-weight: bold;">📊 РЕЗУЛЬТАТЫ </div>
        <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px; margin-top: 20px;">
            <div style="background: rgba(255,255,255,0.2); padding: 10px; border-radius: 10px;">
                <div>📁 Файлов</div><div style="font-size:2em; font-weight:bold;">{len(batch_results)}</div>
            </div>
            <div style="background: rgba(255,255,255,0.2); padding: 10px; border-radius: 10px;">
                <div>⚡ Средняя скорость</div><div style="font-size:2em; font-weight:bold;">{avg_speed} мс</div>
            </div>
            <div style="background: rgba(255,255,255,0.2); padding: 10px; border-radius: 10px;">
                <div>🚀 Макс. скорость</div><div style="font-size:1.5em; font-weight:bold;">{max_speed} мс</div>
            </div>
            <div style="background: rgba(255,255,255,0.2); padding: 10px; border-radius: 10px;">
                <div>🐢 Мин. скорость</div><div style="font-size:1.5em; font-weight:bold;">{min_speed} мс</div>
            </div>
        </div>
        <div style="margin-top:15px;">⏱️ Общее время: {total_duration} мс</div>
        {warning_global}
    </div>
    """
    return analytics_img, stats_html, None, df_display, pdf_p, df_display