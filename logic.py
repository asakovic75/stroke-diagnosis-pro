import os, cv2, torch, glob, numpy as np, time, pandas as pd
from datetime import datetime
from fpdf import FPDF
from fpdf.enums import XPos, YPos
import segmentation_models_pytorch as smp
import pydicom
import matplotlib.pyplot as plt
import io
from PIL import Image
from huggingface_hub import InferenceClient
import pytz

client = InferenceClient("meta-llama/Meta-Llama-3-8B-Instruct")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DB_PATH = "stroke_history.csv"
DB_DICOM_PATH = "stroke_history_dicom.csv"
FONT_PATH = "DejaVuSans.ttf"
COLUMNS = ["ID", "Снимок", "Дата", "Время", "Модель", "Вердикт", "Полушарие", "Площадь", "Достоверность", "Скорость"]

allowed = ["stroke_model.pth", "stroke_model_best.pth"]
model_paths = {f"📌 {name}": name for name in allowed if os.path.exists(name)}
if not model_paths:
    model_paths["❌ Модели не найдены"] = None

DICOM_DESC = {
    "SOPClassUID": "UID класса SOP (тип объекта)",
    "SliceThickness": "Толщина среза (мм)",
    "SamplesPerPixel": "Число цветовых каналов",
    "PhotometricInterpretation": "Фотометрическая интерпретация",
    "Rows": "Высота (пиксели)",
    "Columns": "Ширина (пиксели)",
    "PixelSpacing": "Размер пикселя (мм)",
    "BitsAllocated": "Выделено бит на пиксель",
    "BitsStored": "Используется бит на пиксель",
    "HighBit": "Старший бит",
    "PixelRepresentation": "Тип данных (0-un/1-sign)",
    "WindowCenter": "Центр окна (яркость)",
    "WindowWidth": "Ширина окна (контраст)",
    "RescaleIntercept": "Перехват шкалы (HU)",
    "RescaleSlope": "Наклон шкалы",
    "RescaleType": "Тип шкалы преобразования"
}

current_clinical_context = ""
current_batch_context = ""

def load_database():
    if os.path.exists(DB_PATH):
        try:
            df = pd.read_csv(DB_PATH)
            if len(df.columns) == len(COLUMNS):
                df['ID'] = pd.to_numeric(df['ID'])
                return df.values.tolist()
        except:
            pass
    return []

history_list = load_database()
model = smp.Unet(encoder_name="efficientnet-b4", encoder_weights=None, in_channels=3, classes=1).to(device)

def load_selected_model(model_key):
    if not model_key or "❌" in model_key: 
        return False
    try:
        model.load_state_dict(torch.load(model_paths[model_key], map_location=device))
        model.eval()
        return True
    except:
        return False

def clean_num(val):
    try:
        if isinstance(val, (list, pydicom.multival.MultiValue)):
            return ", ".join([f"{float(x):.3f}" for x in val])
        s = str(val).replace('[','').replace(']','').split(',')[0]
        return f"{float(s):.3f}".rstrip('0').rstrip('.')
    except:
        return str(val)

def ask_ai_assistant(question, context_type, data_summary):
    system_prompt = f"""
Ты — высококвалифицированный врач-радиолог и нейрохирург экспертного центра Stroke Diagnosis Pro.
ПРАВИЛА И КОМПЕТЕНЦИИ:
1. Твоя специализация: КТ-диагностика головного мозга, классификация инсультов и патологий.
2. Знания клиник: 
   - Гродно: Университетская клиника (БЛК 52, бывшая облбольница), БСМП (ул. Советских Пограничников 115), Городская больница №2 (Гагарина 5), Кардиоцентр.
   - Беларусь: РНПЦ Неврологии и Нейрохирургии (Минск), знаешь структуру помощи при ОНМК.
3. Анализ КТ: Ишемический инсульт — темная зона (гиподенсивная). Геморрагический — ярко-белая зона (гиперденсивная, скопление крови).
4. Контекстный анализ: Тебе переданы текущие данные: {data_summary}. Ты должен уметь сравнивать файлы, находить самые тяжелые случаи по % площади и отвечать на медицинские вопросы.
5. На бессмыслицу (ммазащщам) или не по теме отвечай: 'Ошибка: Некорректный запрос или тема не относится к медицинскому анализу КТ'.
"""
    full_prompt = f"{system_prompt}\n\nКОНТЕКСТ ({context_type}):\n{data_summary}\n\nВОПРОС: {question}"
    try:
        if len(question.strip()) < 3 or "ммаза" in question.lower():
            return "Ошибка: Некорректный запрос или тема не относится к медицинскому анализу КТ"
        response = client.chat_completion(messages=[{"role": "user", "content": full_prompt}], max_tokens=1000, temperature=0.4)
        return response.choices[0].message.content
    except:
        return "Ошибка связи с сервером ИИ. Проверьте соединение."

def generate_report_universal(results_list, output_name="Diagnosis_Report.pdf", is_batch=False):
    pdf = FPDF()
    has_font = os.path.exists(FONT_PATH)
    if has_font:
        try:
            pdf.add_font("DejaVu", "", FONT_PATH)
            pdf.add_font("DejaVu", "B", FONT_PATH)
        except: has_font = False

    disclaimer_text = "ВНИМАНИЕ: Данный отчет сформирован системой ИИ. Он носит справочный характер и не является диагнозом."

    for item in results_list:
        orig = item['orig_img']
        res = item['res_img']
        info = item['info']
        meta = item.get('meta', {})

        pdf.add_page()
        if has_font:
            pdf.set_font("DejaVu", "B", 20)
            pdf.cell(0, 15, "МЕДИЦИНСКИЙ ОТЧЕТ АНАЛИЗА КТ", align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.set_font("DejaVu", "", 10)
            pdf.cell(0, 8, f"ID Пациента: {info['p_id']} | Файл: {info['filename']}", align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.cell(0, 8, f"Дата: {info['date']} | Время: {info['time']}", align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.ln(5)

            pdf.set_font("DejaVu", "B", 14)
            pdf.cell(0, 10, "1. РЕЗУЛЬТАТЫ ОБСЛЕДОВАНИЯ:", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.set_font("DejaVu", "", 12)
            pdf.cell(0, 8, f"- Заключение: {info['verdict_ru']}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.cell(0, 8, f"- Полушарие: {info['side_ru']}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.cell(0, 8, f"- Площадь поражения: {info['area']}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.cell(0, 8, f"- Скорость анализа: {info['speed']} мс", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.cell(0, 8, f"- Уверенность системы: {info['conf']}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.cell(0, 8, f"- Модель: {info['model']}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            
            if meta and is_batch:
                pdf.ln(5)
                pdf.set_font("DejaVu", "B", 12)
                pdf.cell(0, 10, "2. ТЕХНИЧЕСКИЕ ПАРАМЕТРЫ (DICOM):", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
                pdf.set_font("DejaVu", "", 10)
                for tag, desc in DICOM_DESC.items():
                    val = meta.get(tag, "Н/Д")
                    pdf.cell(0, 6, f"• {desc}: {val}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        
        if not is_batch:
            cv2.imwrite("o_tmp.jpg", cv2.cvtColor(orig, cv2.COLOR_RGB2BGR))
            cv2.imwrite("r_tmp.jpg", cv2.cvtColor(res, cv2.COLOR_RGB2BGR))
            img_y = 155 if meta else 125
            pdf.image("o_tmp.jpg", x=15, y=img_y, w=85)
            pdf.image("r_tmp.jpg", x=110, y=img_y, w=85)

        pdf.set_y(-30)
        if has_font:
            pdf.set_font("DejaVu", "", 7)
            pdf.multi_cell(0, 4, disclaimer_text, align="C")
        else:
            pdf.set_font("Helvetica", "", 7)
            pdf.multi_cell(0, 4, "WARNING: AI Report. Disclaimer...", align="C")

    pdf.output(output_name)
    return output_name

def dicom_to_rgb(ds):
    img = ds.pixel_array.astype(float)
    intercept = ds.RescaleIntercept if 'RescaleIntercept' in ds else 0
    slope = ds.RescaleSlope if 'RescaleSlope' in ds else 1
    img = img * slope + intercept
    center = ds.WindowCenter[0] if isinstance(ds.WindowCenter, pydicom.multival.MultiValue) else ds.WindowCenter
    width = ds.WindowWidth[0] if isinstance(ds.WindowWidth, pydicom.multival.MultiValue) else ds.WindowWidth
    low, high = center - width // 2, center + width // 2
    img = np.clip(img, low, high)
    img = (img - low) / (high - low) * 255.0
    return cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_GRAY2RGB)

def create_analytics(df):
    plt.close('all')
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    verdict_counts = df['Вердикт'].value_counts()
    colors_v = ['#4CAF50' if x == 'Норма' else '#D32F2F' for x in verdict_counts.index]
    axes[0].pie(verdict_counts, labels=verdict_counts.index, autopct='%1.1f%%', colors=colors_v)
    axes[0].set_title("Статус: Норма / Инсульт", fontsize=12, fontweight='bold')
    areas = df[df['Вердикт'] == 'Инсульт']['Площадь_Ч'].astype(float)
    if not areas.empty:
        n, bins, patches = axes[1].hist(areas, bins=10, edgecolor='black')
        for i in range(len(patches)):
            bin_center = (bins[i] + bins[i+1]) / 2
            if bin_center < 0.5: patches[i].set_facecolor('#FFEB3B')
            elif bin_center < 2.0: patches[i].set_facecolor('#FB8C00')
            else: patches[i].set_facecolor('#D32F2F')
    else: axes[1].text(0.5, 0.5, 'Данных нет', ha='center')
    axes[1].set_title("Тяжесть: Площадь поражения (%)", fontsize=12, fontweight='bold')
    stroke_df = df[df['Вердикт'] == 'Инсульт']
    if not stroke_df.empty:
        side_counts = stroke_df['Полушарие'].value_counts()
        axes[2].pie(side_counts, labels=side_counts.index, autopct='%1.1f%%', colors=['#2196F3', '#00BCD4'])
    else: axes[2].text(0.5, 0.5, 'Инсультов нет', ha='center')
    axes[2].set_title("Локализация (Полушарие)", fontsize=12, fontweight='bold')
    buf = io.BytesIO()
    plt.tight_layout(); plt.savefig(buf, format='png'); buf.seek(0)
    return Image.open(buf)

def core_inference(img):
    img_res = cv2.resize(img, (256, 256))
    x = (img_res.astype(np.float32)/255.0 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    x_t = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).to(device).float()
    with torch.no_grad():
        prob = torch.sigmoid(model(x_t)).cpu().numpy()[0,0]
        mask = (prob > 0.5).astype(np.uint8)
    return img_res, mask, prob

def predict_stroke(file_path, model_key):
    global history_list, current_clinical_context
    if not file_path or not load_selected_model(model_key): return [None]*6
    filename = os.path.basename(file_path)
    input_img = cv2.cvtColor(cv2.imread(file_path), cv2.COLOR_BGR2RGB)
    start_time = time.time()
    img_res, mask, prob = core_inference(input_img)
    speed_ms = round((time.time() - start_time) * 1000, 1)
    is_s = 15 < np.sum(mask) < (256*256*0.7)
    side_ru = ("Левое" if np.sum(mask[:, :128]) > np.sum(mask[:, 128:]) else "Правое") if is_s else "Не выявлено"
    side_en = ("Left" if np.sum(mask[:, :128]) > np.sum(mask[:, 128:]) else "Right") if is_s else "None"
    confidence = f"{np.mean(prob[mask>0])*100:.1f}%" if is_s else "100.0%"
    area = f"{(np.sum(mask)/(256*256))*100:.2f}%"
    status_ru = "ОБНАРУЖЕН ИНСУЛЬТ" if is_s else "НОРМА"
    
    res_view = img_res.copy()
    if is_s:
        ov = res_view.copy(); ov[:] = [255, 120, 120]
        res_view[mask > 0] = cv2.addWeighted(res_view, 0.8, ov, 0.2, 0)[mask > 0]
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(res_view, contours, -1, (255, 0, 0), 2)
        
    p_id = int(max([row[0] for row in history_list])) + 1 if history_list else 1
    now_gr = datetime.now(pytz.timezone('Europe/Minsk'))
    info = {'p_id': p_id, 'filename': filename, 'model': model_key.replace("📌 ",""), 'side_ru': side_ru, 'side_en': side_en, 'conf': confidence, 'area': area, 'verdict_ru': status_ru, 'speed': speed_ms, 'date': now_gr.strftime("%d.%m"), 'time': now_gr.strftime("%H:%M:%S")}
    
    meta = {}
    if filename.lower().endswith('.dcm'):
        ds = pydicom.dcmread(file_path)
        meta = {tag: clean_num(getattr(ds, tag, "Н/Д")) for tag in DICOM_DESC.keys()}

    current_clinical_context = f"Снимок {filename}. Статус: {status_ru}. Площадь: {area}. Полушарие: {side_ru}. Уверенность: {confidence}."
    pdf_p = generate_report_universal([{'orig_img': img_res, 'res_img': res_view, 'info': info, 'meta': meta}], is_batch=False)
    
    history_list.insert(0, [p_id, filename, info['date'], info['time'], info['model'], "Инсульт" if is_s else "Норма", side_ru, area, confidence, f"{speed_ms} мс"])
    pd.DataFrame(history_list, columns=COLUMNS).to_csv(DB_PATH, index=False)
    
    color = "#D32F2F" if is_s else "#2E7D32"
    s_html = f'<div style="text-align: center; font-size: 2.2em; font-weight: bold; color: {color}; padding: 10px;">{status_ru}</div>'
    d_html = f'''<div style="text-align: center; padding: 15px; line-height: 1.8; background-color: transparent;"><div style="font-size: 1.2em;"><b>Модель:</b> {info['model']}</div><div style="font-size: 1.2em;"><b>Достоверность:</b> {confidence}</div><div style="font-size: 1.2em;"><b>Площадь поражения:</b> {area}</div><div style="font-size: 1.2em;"><b>Локализация:</b> {side_ru} полушарие</div><div style="color: #666; font-size: 0.9em; margin-top: 5px;">Пациент ID: {p_id} | Снимок: {filename} | Скорость: {speed_ms} мс</div></div>'''
    return res_view, img_res, s_html, d_html, pd.DataFrame(history_list, columns=COLUMNS), pdf_p

def process_batch(files, model_key):
    global current_batch_context
    if not files or not load_selected_model(model_key): return [None]*6
    batch_results, report_items, log_text, total_time, full_data_ai = [], [], "", 0, ""
    
    for i, f in enumerate(files):
        if not f.name.lower().endswith('.dcm'): continue
        ds = pydicom.dcmread(f.name)
        img = dicom_to_rgb(ds)
        start = time.time()
        img_res, mask, prob = core_inference(img)
        dur = round((time.time() - start) * 1000, 1)
        is_s = 15 < np.sum(mask) < (256*256*0.7)
        a_v = (np.sum(mask)/(256*256))*100
        side_ru = ("Левое" if np.sum(mask[:, :128]) > np.sum(mask[:, 128:]) else "Правое") if is_s else "Нет"
        side_en = ("Left" if np.sum(mask[:, :128]) > np.sum(mask[:, 128:]) else "Right") if is_s else "None"
        
        res_view = img_res.copy()
        if is_s:
            ov = res_view.copy(); ov[:] = [255, 120, 120]
            res_view[mask > 0] = cv2.addWeighted(res_view, 0.8, ov, 0.2, 0)[mask > 0]
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(res_view, contours, -1, (255, 0, 0), 2)

        meta_tags = {tag: clean_num(getattr(ds, tag, "Н/Д")) for tag in DICOM_DESC.keys()}
        now_gr = datetime.now(pytz.timezone('Europe/Minsk'))
        info = {'p_id': f"B-{i+1}", 'filename': os.path.basename(f.name), 'model': model_key.replace("📌 ",""), 'side_ru': side_ru, 'side_en': side_en, 'conf': f"{np.mean(prob[mask>0])*100:.1f}%" if is_s else "100%", 'area': f"{a_v:.2f}%", 'verdict_ru': "ОБНАРУЖЕН ИНСУЛЬТ" if is_s else "НОРМА", 'speed': dur, 'date': now_gr.strftime("%d.%m"), 'time': now_gr.strftime("%H:%M:%S")}
        
        report_items.append({'orig_img': img_res, 'res_img': res_view, 'info': info, 'meta': meta_tags})
        
        row = {
            "ID": i+1, 
            "Снимок": info['filename'], 
            "Дата": info['date'],
            "Время": info['time'],
            "Модель": info['model'],
            "Вердикт": "Инсульт" if is_s else "Норма", 
            "Полушарие": side_ru, 
            "Площадь": info['area'], 
            "Достоверность": info['conf'],
            "Скорость": f"{dur} мс", 
            "Площадь_Ч": a_v
        }
        batch_results.append(row); total_time += dur; log_text += f"{i+1}. {row['Снимок']} — {dur} мс<br>"
        full_data_ai += f"Файл {row['Снимок']}: {row['Вердикт']}, площадь {a_v:.2f}%. "

    df = pd.DataFrame(batch_results); df.to_csv(DB_DICOM_PATH, index=False)
    pdf_p = generate_report_universal(report_items, "Batch_Diagnosis_Report.pdf", is_batch=True)
    avg = round(total_time/len(batch_results), 1) if batch_results else 0
    current_batch_context = f"Пакет из {len(batch_results)} файлов. Средняя скорость {avg} мс. Данные: {full_data_ai}"
    s_html = f'<div style="text-align: center; font-size: 2.1em; font-weight: bold; color: #1976D2; padding: 10px;">ОБРАБОТАНО: {len(batch_results)} ФАЙЛОВ</div>'
    d_html = f'''<div style="text-align: center; padding: 15px; line-height: 1.8; background-color: transparent;"><div style="font-size: 1.1em;">{log_text}</div><hr style="opacity:0.2;"><div style="font-size: 1.2em;"><b>Средняя скорость:</b> {avg} мс/файл</div></div>'''
    
    return create_analytics(df), s_html, d_html, df.drop(columns=['Площадь_Ч']), pdf_p