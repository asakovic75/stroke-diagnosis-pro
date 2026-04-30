import gradio as gr
import pandas as pd
from main_logic import predict_stroke, process_batch
from config import model_paths, DB_PATH, DB_DICOM_PATH, COLUMNS
from database import get_history_dataframe

css = """
.gradio-container { max-width: 1100px !important; margin: 0 auto !important; }
#header { text-align: center !important; }
footer { display: none !important; }
.compact-df { margin-top: 10px !important; }
"""

with gr.Blocks(fill_width=True) as demo:
    gr.Markdown("<div id='header'><h1>🧠 Диагностика инсульта по КТ</h1><h3>Интеллектуальная система анализа медицинских изображений</h3></div>")
    
    with gr.Tabs():
        with gr.Tab("🏥 Клинический режим"):
            with gr.Column():
                model_selector = gr.Dropdown(choices=list(model_paths.keys()), value=list(model_paths.keys())[0], label="🔧 ВЫБЕРИТЕ НЕЙРОСЕТЕВУЮ МОДЕЛЬ")
                input_f = gr.File(label="📸 ЗАГРУЗИТЕ DICOM ФАЙЛ", file_types=[".dcm"])
                with gr.Row():
                    btn = gr.Button("🔍 ЗАПУСТИТЬ АНАЛИЗ", variant="primary", size="lg")
                    clr = gr.ClearButton(value="🗑 ОЧИСТИТЬ ЭКРАН", size="lg")
                status_out, details_out = gr.HTML(), gr.HTML()
                with gr.Row():
                    o_res = gr.Image(label="🎯 Результат сегментации", height=300, width=300)
                    o_orig = gr.Image(label="📷 Исходный снимок", height=300, width=300)
                pdf_file = gr.File(label="📄 МЕДИЦИНСКИЙ ОТЧЕТ (PDF)")
                
                initial_df = get_history_dataframe()
                history_table = gr.Dataframe(value=initial_df, interactive=True, elem_classes="compact-df")
                with gr.Row():
                    save_csv_btn = gr.Button("💾 СОХРАНИТЬ (CSV)", variant="primary", size="lg")
                    download_csv_btn = gr.DownloadButton("📥 СКАЧАТЬ (CSV)", size="lg")
                    
            btn.click(predict_stroke, [input_f, model_selector], [o_res, o_orig, status_out, details_out, history_table, pdf_file])
            clr.add([input_f, o_res, o_orig, status_out, details_out, history_table, pdf_file])

        with gr.Tab("🚀 Массовый поток"):
            with gr.Column():
                bm_sel = gr.Dropdown(choices=list(model_paths.keys()), value=list(model_paths.keys())[0], label="🔧 ВЫБЕРИТЕ НЕЙРОСЕТЕВУЮ МОДЕЛЬ")
                binp = gr.File(label="📸 ЗАГРУЗИТЕ ПАКЕТ DICOM ФАЙЛОВ", file_count="multiple", file_types=[".dcm"])
                with gr.Row():
                    bbtn = gr.Button("🔍 ЗАПУСТИТЬ АНАЛИЗ", variant="primary", size="lg")
                    bclr = gr.ClearButton(value="🗑 ОЧИСТИТЬ ЭКРАН", size="lg")
                bst_out, bdet_out = gr.HTML(), gr.HTML()
                gr.Markdown("### 📊 СТАТИСТИКА ПОТОКА")
                bres = gr.Image(show_label=False)
                b_pdf_file = gr.File(label="📄 МЕДИЦИНСКИЙ ОТЧЕТ ПО ПАКЕТУ (PDF)")
                
                bhist = gr.Dataframe(interactive=True, elem_classes="compact-df")
                state_full_df = gr.State()
                with gr.Row():
                    btn_priority = gr.Button("🎯 ПРИОРИТЕТ", variant="secondary", size="lg")
                    btn_reset = gr.Button("↺ СБРОС", variant="secondary", size="lg")
                    bdl_b = gr.DownloadButton("📥 СКАЧАТЬ CSV", size="lg")
                    
            bbtn.click(process_batch, [binp, bm_sel], [bres, bst_out, bdet_out, bhist, b_pdf_file, state_full_df])
            bclr.add([binp, bres, bst_out, bdet_out, bhist, b_pdf_file])
            
            def priority_filter(df):
                if not isinstance(df, pd.DataFrame) or df.empty:
                    return pd.DataFrame()
                if 'Инсульт' not in df['Вердикт'].values:
                    return pd.DataFrame(columns=df.columns)
                filtered = df[df['Вердикт'] == 'Инсульт'].copy()
                filtered = filtered.sort_values(by='Площадь', key=lambda x: x.str.rstrip('%').astype(float), ascending=False)
                return filtered
            
            btn_priority.click(priority_filter, [state_full_df], [bhist])
            btn_reset.click(lambda df: df, [state_full_df], [bhist])
            bdl_b.click(lambda: DB_DICOM_PATH, None, bdl_b)

    save_csv_btn.click(lambda: gr.Info("Сохранено!"), None, None)
    download_csv_btn.click(lambda: DB_PATH, None, download_csv_btn)

if __name__ == "__main__":
    demo.launch(ssr_mode=False, theme=gr.themes.Soft(), css=css)