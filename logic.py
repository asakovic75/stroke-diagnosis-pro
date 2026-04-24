import os, cv2, torch, glob, numpy as np, time, pandas as pd
from datetime import datetime
from fpdf import FPDF
from fpdf.enums import XPos, YPos
import segmentation_models_pytorch as smp
import pydicom
import matplotlib.pyplot as plt
import io
from PIL import Image
import pytz

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DB_PATH = "stroke_history.csv"
DB_DICOM_PATH = "stroke_history_dicom.csv"
FONT_PATH = "DejaVuSans.ttf"
COLUMNS = ["ID", "Снимок", "Дата", "Время", "Модель", "Вердикт", "Полушарие", "Плотность (HU)", "Площадь", "Достоверность", "Скорость"]

def get_hu_analysis(image_orig, mask_256, ds=None):
    try:
        mask_orig = cv2.resize(mask_256, (image_orig.shape[1], image_orig.shape[0]), interpolation=cv2.INTER_NEAREST)
        
        if ds is not None:
            raw_pixels = ds.pixel_array.astype(float)
            slope = getattr(ds, 'RescaleSlope', 1)
            intercept = getattr(ds, 'RescaleIntercept', 0)
            hu_data = raw_pixels * slope + intercept
            stroke_pixels = hu_data[mask_orig > 0]
        else:
            gray = cv2.cvtColor(image_orig, cv2.COLOR_RGB2GRAY)
            stroke_pixels = (gray[mask_orig > 0].astype(float) / 255.0) * 80 + 0
            
        if len(stroke_pixels) > 0:
            avg_val = np.mean(stroke_pixels)
            label = "Кровь/Геморрагия" if avg_val > 50 else "Ишемия/Отек"
            return f"{avg_val:.1f} HU ({label})"
    except Exception as e:
        print(f"Ошибка анализа плотности: {e}")
    return "Н/Д"

def get_artery_basin(mask_256, side_ru):
    if side_ru == "Не выявлено": return side_ru
    
    y_coords, x_coords = np.where(mask_256 > 0)
    if len(y_coords) == 0: return side_ru
    
    mean_y = np.mean(y_coords)
    
    if mean_y < 80:
        basin = "бассейн ПМА (передняя)"
    elif mean_y > 180:
        basin = "бассейн ЗМА (задняя)"
    else:
        basin = "бассейн СМА (средняя)"
        
    return f"{side_ru} полушарие, {basin}"
    
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

def generate_report_universal(results_list, output_name="Diagnosis_Report.pdf", is_batch=False):
    pdf = FPDF()
    has_font = os.path.exists(FONT_PATH)
    if has_font:
        try:
            pdf.add_font("DejaVu", "", FONT_PATH)
            pdf.add_font("DejaVu", "B", FONT_PATH)
        except: 
            has_font = False

    disclaimer_text = "ВНИМАНИЕ: Данный отчет сформирован системой ИИ. Он носит справочный характер и не является диагнозом."

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
            pdf.cell(0, 8, f"- Заключение: {info['verdict_ru']}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.cell(0, 8, f"- Плотность очага: {info.get('hu', 'Н/Д')}", new_x=XPos.LMARGIN, new_y=YPos.NEXT) 
            pdf.cell(0, 8, f"- Локализация: {info['side_ru']}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
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
            os.remove("o_tmp.jpg")
            os.remove("r_tmp.jpg")

        pdf.set_y(-30)
        if has_font:
            pdf.set_font("DejaVu", "", 7)
            pdf.multi_cell(0, 4, disclaimer_text, align="C")
        else:
            pdf.set_font("Helvetica", "", 7)
            pdf.multi_cell(0, 4, "WARNING: AI Report. This is for informational purposes only.", align="C")

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
            if bin_center < 0.5:
                patches[i].set_facecolor('#FFEB3B')
            elif bin_center < 2.0:
                patches[i].set_facecolor('#FB8C00')
            else:
                patches[i].set_facecolor('#D32F2F')
        axes[1].set_xlabel("Площадь поражения (%)")
    else: 
        axes[1].text(0.5, 0.5, 'Данных нет', ha='center')
    axes[1].set_title("Тяжесть: Площадь поражения (%)", fontsize=12, fontweight='bold')
    
    stroke_df = df[df['Вердикт'] == 'Инсульт']
    if not stroke_df.empty:
        side_counts = stroke_df['Полушарие'].value_counts()
        axes[2].pie(side_counts, labels=side_counts.index, autopct='%1.1f%%', colors=['#2196F3', '#00BCD4'])
    else: 
        axes[2].text(0.5, 0.5, 'Инсультов нет', ha='center')
    axes[2].set_title("Локализация (Полушарие)", fontsize=12, fontweight='bold')
    
    avg_speed = df['Скорость'].str.replace(' мс', '').astype(float).mean()
    
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png', dpi=150)
    buf.seek(0)
    plt.close()
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
    global history_list
    if not file_path or not load_selected_model(model_key): return [None]*6
    
    filename = os.path.basename(file_path)
    is_dicom = filename.lower().endswith('.dcm')
    
    ds = pydicom.dcmread(file_path) if is_dicom else None
    input_img = dicom_to_rgb(ds) if is_dicom else cv2.cvtColor(cv2.imread(file_path), cv2.COLOR_BGR2RGB)
    
    start_time = time.time()
    img_res, mask, prob = core_inference(input_img)
    speed_ms = round((time.time() - start_time) * 1000, 1)
    
    is_s = np.sum(mask) > 20
    
    hu_info = get_hu_analysis(input_img, mask, ds) if is_s else "Н/Д"
    raw_side = "Левое" if np.sum(mask[:, :128]) > np.sum(mask[:, 128:]) else "Правое"
    side_ru = get_artery_basin(mask, raw_side) if is_s else "Не выявлено"

    area = f"{(np.sum(mask)/(256*256))*100:.2f}%"
    confidence = f"{np.mean(prob[mask>0])*100:.1f}%" if is_s else "100.0%"
    status_ru = "ОБНАРУЖЕН ИНСУЛЬТ" if is_s else "НОРМА"
    
    res_view = img_res.copy()
    if is_s:
        ov = res_view.copy()
        ov[:] = [255, 120, 120]
        res_view[mask > 0] = cv2.addWeighted(res_view, 0.8, ov, 0.2, 0)[mask > 0]
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(res_view, contours, -1, (255, 0, 0), 2)
        
    p_id = int(max([row[0] for row in history_list])) + 1 if history_list else 1
    now_gr = datetime.now(pytz.timezone('Europe/Minsk'))
    
    info = {
        'p_id': p_id, 'filename': filename, 'model': model_key.replace("📌 ",""), 
        'side_ru': side_ru, 'conf': confidence, 'area': area, 'hu': hu_info,
        'verdict_ru': status_ru, 'speed': speed_ms, 
        'date': now_gr.strftime("%d.%m"), 'time': now_gr.strftime("%H:%M:%S")
    }
    
    meta = {}
    if is_dicom:
        meta = {tag: clean_num(getattr(ds, tag, "Н/Д")) for tag in DICOM_DESC.keys()}
    
    pdf_p = generate_report_universal([{'orig_img': img_res, 'res_img': res_view, 'info': info, 'meta': meta}], is_batch=False)
    
    history_list.insert(0, [p_id, filename, info['date'], info['time'], info['model'], 
                            "Инсульт" if is_s else "Норма", side_ru, hu_info, area, confidence, f"{speed_ms} мс"])
    
    pd.DataFrame(history_list, columns=COLUMNS).to_csv(DB_PATH, index=False)
    
    color = "#D32F2F" if is_s else "#2E7D32"
    
    stats_html = f"""
    <div style="text-align: center; padding: 15px; background: linear-gradient(135deg, {color if is_s else '#2E7D32'} 0%, {color if is_s else '#1B5E20'} 100%); border-radius: 15px; color: white;">
        <div style="font-size: 1.8em; font-weight: bold;">{status_ru}</div>
        <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px; margin-top: 20px;">
            <div style="background: rgba(255,255,255,0.2); padding: 10px; border-radius: 10px;">
                <div style="font-size: 0.9em;">🧠 Тип патологии</div>
                <div style="font-size: 1.2em; font-weight: bold;">{hu_info}</div>
            </div>
            <div style="background: rgba(255,255,255,0.2); padding: 10px; border-radius: 10px;">
                <div style="font-size: 0.9em;">📍 Локализация</div>
                <div style="font-size: 1.2em; font-weight: bold;">{side_ru}</div>
            </div>
            <div style="background: rgba(255,255,255,0.2); padding: 10px; border-radius: 10px;">
                <div style="font-size: 0.9em;">📐 Площадь поражения</div>
                <div style="font-size: 1.2em; font-weight: bold;">{area}</div>
            </div>
            <div style="background: rgba(255,255,255,0.2); padding: 10px; border-radius: 10px;">
                <div style="font-size: 0.9em;">⚡ Скорость анализа</div>
                <div style="font-size: 1.2em; font-weight: bold;">{speed_ms} мс</div>
            </div>
            <div style="background: rgba(255,255,255,0.2); padding: 10px; border-radius: 10px;">
                <div style="font-size: 0.9em;">🎯 Уверенность</div>
                <div style="font-size: 1.2em; font-weight: bold;">{confidence}</div>
            </div>
            <div style="background: rgba(255,255,255,0.2); padding: 10px; border-radius: 10px;">
                <div style="font-size: 0.9em;">🔧 Модель</div>
                <div style="font-size: 1.0em; font-weight: bold;">{info['model']}</div>
            </div>
        </div>
        <div style="margin-top: 15px; font-size: 0.85em;">
            🆔 Пациент ID: {p_id} | 📄 {filename}
        </div>
    </div>
    """
    
    d_html = ""
    
    return res_view, img_res, stats_html, d_html, pd.DataFrame(history_list, columns=COLUMNS), pdf_p
    
def process_batch(files, model_key):
    if not files or not load_selected_model(model_key): 
        return [None] * 6
    
    batch_results = []
    report_items = []
    speed_log = []
    total_start_time = time.time()
    
    for i, f in enumerate(files):
        if not f.name.lower().endswith('.dcm'): 
            continue
            
        file_start_time = time.time()
        
        ds = pydicom.dcmread(f.name)
        img = dicom_to_rgb(ds)
        
        img_res, mask, prob = core_inference(img)
        
        file_duration = round((time.time() - file_start_time) * 1000, 1)
        speed_log.append(file_duration)
        
        is_s = np.sum(mask) > 20
        area_v = (np.sum(mask)/(256*256))*100
        area_str = f"{area_v:.2f}%"
        
        hu = get_hu_analysis(img, mask, ds) if is_s else "Н/Д"
        side = get_artery_basin(mask, ("Левое" if np.sum(mask[:, :128]) > np.sum(mask[:, 128:]) else "Правое")) if is_s else "Не выявлено"
        confidence = f"{np.mean(prob[mask>0])*100:.1f}%" if is_s else "100%"
        
        res_view = img_res.copy()
        if is_s:
            ov = res_view.copy()
            ov[:] = [255, 120, 120]
            res_view[mask > 0] = cv2.addWeighted(res_view, 0.8, ov, 0.2, 0)[mask > 0]
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(res_view, contours, -1, (255, 0, 0), 2)

        now = datetime.now(pytz.timezone('Europe/Minsk'))
        
        meta = {tag: clean_num(getattr(ds, tag, "Н/Д")) for tag in DICOM_DESC.keys()}
        
        info = {
            'p_id': f"B-{i+1}", 
            'filename': os.path.basename(f.name), 
            'model': model_key.replace("📌 ",""), 
            'verdict_ru': "Инсульт" if is_s else "Норма",
            'hu': hu, 
            'side_ru': side, 
            'area': area_str, 
            'speed': file_duration,
            'conf': confidence,
            'date': now.strftime("%d.%m"), 
            'time': now.strftime("%H:%M")
        }
        
        report_items.append({'orig_img': img_res, 'res_img': res_view, 'info': info, 'meta': meta})
        
        batch_results.append([
            i+1, 
            info['filename'], 
            info['date'], 
            info['time'], 
            info['model'], 
            info['verdict_ru'], 
            side, 
            hu, 
            area_str, 
            confidence, 
            f"{file_duration} мс",
            area_v
        ])
    
    total_duration = round((time.time() - total_start_time) * 1000, 1)
    avg_speed = round(np.mean(speed_log), 1) if speed_log else 0
    min_speed = min(speed_log) if speed_log else 0
    max_speed = max(speed_log) if speed_log else 0
    
    df_columns = COLUMNS + ["Площадь_Ч"]
    df = pd.DataFrame(batch_results, columns=df_columns)
    df = df.drop(columns=['Площадь_Ч'])
    df.to_csv(DB_DICOM_PATH, index=False)
    
    pdf_p = generate_report_universal(report_items, "Batch_Diagnosis_Report.pdf", is_batch=True)
    
    df_ana = pd.DataFrame(batch_results, columns=df_columns)
    df_ana['Площадь_Ч'] = df_ana['Площадь_Ч'].astype(float)
    
    stats_html = f"""
    <div style="text-align: center; padding: 15px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; color: white;">
        <div style="font-size: 1.8em; font-weight: bold;">📊 РЕЗУЛЬТАТЫ </div>
        <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px; margin-top: 20px;">
            <div style="background: rgba(255,255,255,0.2); padding: 10px; border-radius: 10px;">
                <div style="font-size: 0.9em;">📁 Файлов</div>
                <div style="font-size: 2em; font-weight: bold;">{len(batch_results)}</div>
            </div>
            <div style="background: rgba(255,255,255,0.2); padding: 10px; border-radius: 10px;">
                <div style="font-size: 0.9em;">⚡ Средняя скорость</div>
                <div style="font-size: 2em; font-weight: bold;">{avg_speed} мс</div>
            </div>
            <div style="background: rgba(255,255,255,0.2); padding: 10px; border-radius: 10px;">
                <div style="font-size: 0.9em;">🚀 Макс. скорость</div>
                <div style="font-size: 1.5em; font-weight: bold;">{max_speed} мс</div>
            </div>
            <div style="background: rgba(255,255,255,0.2); padding: 10px; border-radius: 10px;">
                <div style="font-size: 0.9em;">🐢 Мин. скорость</div>
                <div style="font-size: 1.5em; font-weight: bold;">{min_speed} мс</div>
            </div>
        </div>
        <div style="margin-top: 15px; font-size: 0.85em;">
            ⏱️ Общее время: {total_duration} мс
        </div>
    </div>
    """
    
    return create_analytics(df_ana), stats_html, None, df, pdf_p
