from fpdf import FPDF
import os


def generate_pdf_from_charts(chart_paths, output_path="outputs/eda_report.pdf"):
    pdf = FPDF(orientation='P', unit='mm', format='A4')

    for chart in chart_paths:
        pdf.add_page()
        pdf.set_font("Arial", size=14)
        title = os.path.basename(chart).replace('_', ' ').replace('.png', '').title()
        pdf.cell(200, 10, txt=title, ln=True, align='C')

        # Resize image for A4 (max width = 180mm)
        pdf.image(chart, x=15, y=30, w=180)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    pdf.output(output_path)
    return output_path
