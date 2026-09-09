import os
import glob
import numpy as np
import pandas as pd
import xgboost as xgb
from flask import Flask, render_template, request, jsonify
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__, template_folder='template')

model = None
target_encoder = None
feature_columns = []
categorical_encoders = {}
baseline_energy = 0.0

precision_specs = {
    'fp32': {'accuracy': 99.25, 'latency': 0.20, 'energy': 1.00},
    'fp16': {'accuracy': 99.00, 'latency': 0.50, 'energy': 0.50},
    'int8': {'accuracy': 98.10, 'latency': 0.80, 'energy': 0.30},
    'int4': {'accuracy': 97.59, 'latency': 1.00, 'energy': 0.18}
}

def train_optimizer_model():
    global model, target_encoder, feature_columns, categorical_encoders, baseline_energy
    
    # Locate dataset
    dataset_path = 'AI Platform Performance Dataset.xlsx'
    if not os.path.exists(dataset_path):
        csvs = glob.glob('*.xlsx') + glob.glob('*.csv')
        if csvs:
            dataset_path = csvs[0]
        else:
            raise FileNotFoundError("Dataset file not found!")

    df = pd.read_excel(dataset_path) if dataset_path.endswith('.xlsx') else pd.read_csv(dataset_path)
    df.columns = df.columns.str.strip()
    df_clean = df.dropna().copy()

    # Drop datetime columns
    datetime_cols = df_clean.select_dtypes(include=['datetime64', 'datetime']).columns.tolist()
    df_clean = df_clean.drop(columns=datetime_cols)

    lat_col = next((c for c in df_clean.columns if 'inference' in c.lower() or 'latency' in c.lower() or 'time' in c.lower()), 'Inference Time')
    eng_col = next((c for c in df_clean.columns if 'energy' in c.lower() or 'power' in c.lower()), 'Energy Consumption')

    baseline_energy = float(df_clean[eng_col].mean())

    # 1. BALANCED MULTI-CLASS ASSIGNMENT (Spread across fp32, fp16, int8, int4)
    def select_constrained_quantization(row):
        base_lat = float(row[lat_col])
        base_eng = float(row[eng_col])
        
        # Thresholds tailored to dataset values
        if base_eng <= 20.0 and base_lat <= 8.0:
            return 'fp32'
        elif base_eng <= 40.0 or base_lat <= 15.0:
            return 'fp16'
        elif base_eng <= 65.0 or base_lat <= 25.0:
            return 'int8'
        else:
            return 'int4'

    df_clean['Optimal_Quantization'] = df_clean.apply(select_constrained_quantization, axis=1)

    # 2. FIT TARGET ENCODER
    target_encoder = LabelEncoder()
    df_clean['Target'] = target_encoder.fit_transform(df_clean['Optimal_Quantization'])

    # 3. ENCODE FEATURE COLUMNS
    cat_cols = df_clean.select_dtypes(include=['object', 'category']).columns.tolist()
    if 'Optimal_Quantization' in cat_cols:
        cat_cols.remove('Optimal_Quantization')

    for col in cat_cols:
        le = LabelEncoder()
        df_clean[col] = le.fit_transform(df_clean[col].astype(str))
        categorical_encoders[col] = le

    feature_columns = [c for c in df_clean.columns if c not in ['Optimal_Quantization', 'Target']]
    X = df_clean[feature_columns]
    y = df_clean['Target']

    # 4. TRAIN MULTI-CLASS XGBOOST
    num_classes = len(np.unique(y))
    model = xgb.XGBClassifier(
        objective='multi:softprob' if num_classes > 2 else 'binary:logistic',
        num_class=num_classes if num_classes > 2 else None,
        n_estimators=100,
        learning_rate=0.05,
        max_depth=5,
        random_state=42
    )
    model.fit(X, y)
    print("XGBoost Datacenter Optimizer Model Trained Successfully.")

train_optimizer_model()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json or {}
    
    # Collect incoming request values
    req_latency = float(data.get('latency', 15.0))
    req_energy = float(data.get('energy', 35.0))
    req_memory = float(data.get('memory', 8.0))
    req_accuracy = float(data.get('accuracy', 85.0))

    input_dict = {}
    for col in feature_columns:
        col_lower = col.lower()
        if 'inference' in col_lower or 'latency' in col_lower or 'time' in col_lower:
            input_dict[col] = req_latency
        elif 'energy' in col_lower or 'power' in col_lower:
            input_dict[col] = req_energy
        elif 'memory' in col_lower:
            input_dict[col] = req_memory
        elif 'accuracy' in col_lower:
            input_dict[col] = req_accuracy
        else:
            input_dict[col] = 0.0

    input_df = pd.DataFrame([input_dict], columns=feature_columns)
    
    # Predict precision
    pred_class = int(model.predict(input_df)[0])
    predicted_precision = target_encoder.inverse_transform([pred_class])[0]

    # Energy calculations
    unoptimized_kwh = req_energy
    multiplier = precision_specs.get(predicted_precision, {}).get('energy', 0.5)
    optimized_kwh = unoptimized_kwh * multiplier
    energy_saved_kwh = unoptimized_kwh - optimized_kwh
    savings_pct = ((unoptimized_kwh - optimized_kwh) / max(unoptimized_kwh, 1e-5)) * 100

    return jsonify({
        'precision': predicted_precision,
        'unoptimized_energy': round(unoptimized_kwh, 3),
        'optimized_energy': round(optimized_kwh, 3),
        'saved_energy': round(energy_saved_kwh, 3),
        'savings_percentage': round(savings_pct, 1),
        'accuracy': precision_specs.get(predicted_precision, {}).get('accuracy', 98.1),
        'relative_latency': precision_specs.get(predicted_precision, {}).get('latency', 0.8)
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)