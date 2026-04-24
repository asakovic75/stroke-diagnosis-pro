import gradio as gr
import pandas as pd
import logic

css = """
.gradio-container { max-width: 1100px !important; margin: 0 auto !important; }
#header { text-align: center !important; }
footer { display: none !important; }
.compact-df { margin-top: -10px !important; }
"""

with gr.Blocks(fill_width=True, css=css) as demo:
    gr.Markdown("<div id='header'><h1>🧠 Диагностика инсульта по КТ</h1><h3>Интеллектуальная система анализа медицинских изображений</h3></div>")
    
    with gr.Tabs():
        with gr.Tab("🏥 Клинический режим"):
            with gr.Column():
                model_selector = gr.Dropdown(choices=list(logic.model_paths.keys()), value=list(logic.model_paths.keys())[0], label="🔧 ВЫБЕРИТЕ НЕЙРОСЕТЕВУЮ МОДЕЛЬ")
                input_f = gr.Image(label="📸 ЗАГРУЗИТЕ СНИМОК КТ", type="filepath")
                with gr.Row():
                    btn = gr.Button("🔍 ЗАПУСТИТЬ АНАЛИЗ", variant="primary", size="lg")
                    clr = gr.ClearButton(value="🗑 ОЧИСТИТЬ ЭКРАН", size="lg")
                status_out, details_out = gr.HTML(), gr.HTML()
                with gr.Row():
                    o_res, o_orig = gr.Image(label="🎯 Результат сегментации"), gr.Image(label="📷 Исходный снимок")
                pdf_file = gr.File(label="📄 МЕДИЦИНСКИЙ ОТЧЕТ (PDF)")
                
                gr.HTML("<br>")
                history_table = gr.Dataframe(value=pd.DataFrame(logic.history_list, columns=logic.COLUMNS), interactive=False, elem_classes="compact-df")
                with gr.Row():
                    save_csv_btn = gr.Button("💾 СОХРАНИТЬ (CSV)", variant="primary", size="lg")
                    download_csv_btn = gr.DownloadButton("📥 СКАЧАТЬ (CSV)", size="lg")
                    
            btn.click(logic.predict_stroke, [input_f, model_selector], [o_res, o_orig, status_out, details_out, history_table, pdf_file])
            clr.add([input_f, o_res, o_orig, status_out, details_out, history_table, pdf_file])

        with gr.Tab("🚀 Массовый поток"):
            with gr.Column():
                bm_sel = gr.Dropdown(choices=list(logic.model_paths.keys()), value=list(logic.model_paths.keys())[0], label="🔧 ВЫБЕРИТЕ НЕЙРОСЕТЕВУЮ МОДЕЛЬ")
                binp = gr.File(label="📸 ЗАГРУЗИТЕ ПАКЕТ DICOM ФАЙЛОВ", file_count="multiple")
                with gr.Row():
                    bbtn = gr.Button("🔍 ЗАПУСТИТЬ АНАЛИЗ", variant="primary", size="lg")
                    bclr = gr.ClearButton(value="🗑 ОЧИСТИТЬ ЭКРАН", size="lg")
                bst_out, bdet_out = gr.HTML(), gr.HTML()
                gr.Markdown("### 📊 СТАТИСТИКА ПОТОКА")
                bres = gr.Image(show_label=False)
                b_pdf_file = gr.File(label="📄 МЕДИЦИНСКИЙ ОТЧЕТ ПО ПАКЕТУ (PDF)")
                
                gr.HTML("<br>") 
                bhist = gr.Dataframe(interactive=False, elem_classes="compact-df")
                bdl_b = gr.DownloadButton("📥 СКАЧАТЬ (CSV)", size="lg")
                    
            bbtn.click(logic.process_batch, [binp, bm_sel], [bres, bst_out, bdet_out, bhist, b_pdf_file])
            bclr.add([binp, bres, bst_out, bdet_out, bhist, b_pdf_file])

    save_csv_btn.click(lambda: gr.Info("Сохранено!"), None, None)
    download_csv_btn.click(lambda: logic.DB_PATH, None, download_csv_btn)
    bdl_b.click(lambda: logic.DB_DICOM_PATH, None, bdl_b)

if __name__ == "__main__":
    demo.launch(ssr_mode=False, theme=gr.themes.Soft())
