from flask import Flask, render_template, request, send_file
import os
import pandas as pd
from modules.eda_pipeline import auto_eda_pipeline
from modules.model_pipeline import train_best_model

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    try:
        file = request.files['file']
        task_type = request.form.get('task_type')
        target_col = request.form.get('target_col')

        if file and task_type and target_col:
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)

            # Load CSV safely
            try:
                df = pd.read_csv(filepath, encoding='utf-8')
            except:
                df = pd.read_csv(filepath, encoding='latin1')

            # Clean column names and target input
            df.columns = df.columns.str.strip()
            target_col = target_col.strip()

            # Debug print
            print("🔍 User-selected target column:", target_col)
            print("✅ Dataset columns:", df.columns.tolist())

            # Validate column presence
            if target_col not in df.columns:
                return f"❌ Error: Target column '{target_col}' not found in uploaded file."

            # 💡 Clean common dirty formats in 'Price' column
            if target_col.lower() == 'price':
                df[target_col] = df[target_col].astype(str).str.replace(',', '')
                df[target_col] = df[target_col].replace({'Ask For Price': None, 'ask for price': None})
                df[target_col] = pd.to_numeric(df[target_col], errors='coerce')

            # Run EDA and Modeling
            clean_df, eda_summary = auto_eda_pipeline(df, task_type=task_type, target_col=target_col)
            clean_path = os.path.join(OUTPUT_FOLDER, "cleaned_data.csv")
            clean_df.to_csv(clean_path, index=False)

            best_model, report = train_best_model(clean_df, task_type=task_type)

            return render_template("result.html", report=report, clean_path=clean_path)

        return "❌ Missing required fields: file, task type, or target column."

    except Exception as e:
        return f"❌ Error during processing: {str(e)}"

@app.route('/download')
def download():
    return send_file(os.path.join(OUTPUT_FOLDER, 'cleaned_data.csv'), as_attachment=True)

@app.route('/download_pdf')
def download_pdf():
    return send_file("outputs/eda_report.pdf", as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
