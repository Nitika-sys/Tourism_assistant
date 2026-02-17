import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder

def run_modeling_and_evaluation():
    print("--- Starting Model Training & Evaluation ---")
    
    # 1. LOAD DATA
    if not os.path.exists('master_tourism_data.csv'):
        print("❌ Error: master_tourism_data.csv not found. Run data_pipeline.py first.")
        return
        
    df = pd.read_csv('master_tourism_data.csv')
    
    # --- DATA TYPE CONVERSION ---
    # Ensure IDs are numeric for the ML models
    id_cols = ['ContinentId', 'CountryId', 'RegionId', 'CityId', 'VisitMonth', 'AttractionTypeId', 'UserId', 'AttractionId']
    for col in id_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)

    # 2. FILTER FOR REAL VISITS
    real_visits = df[df['Rating'] > 0].copy()
    
    if real_visits.empty:
        print("❌ Error: No ratings found in your data. Check your Transaction.xlsx file.")
        return

    # 3. PREPROCESSING
    le_vmode = LabelEncoder()
    real_visits['VisitMode_Encoded'] = le_vmode.fit_transform(real_visits['VisitMode'].astype(str))
    
    features = ['ContinentId', 'CountryId', 'RegionId', 'CityId', 'VisitMonth', 'AttractionTypeId']
    X = real_visits[features]
    
    # --- TASK 1: REGRESSION (Rating) ---
    y_reg = real_visits['Rating']
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X, y_reg, test_size=0.2, random_state=42)
    reg_model = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_train_r, y_train_r)
    print(f"✅ Regression Model Trained (R2: {r2_score(y_test_r, reg_model.predict(X_test_r)):.2f})")

    # --- TASK 2: CLASSIFICATION (VisitMode) ---
    y_clf = real_visits['VisitMode_Encoded']
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X, y_clf, test_size=0.2, random_state=42)
    clf_model = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train_c, y_train_c)
    print(f"✅ Classification Model Trained (Acc: {accuracy_score(y_test_c, clf_model.predict(X_test_c)):.2f})")

    # --- TASK 3: RECOMMENDATION (User-Item Matrix) ---
    user_item_matrix = real_visits.pivot_table(index='UserId', columns='AttractionId', values='Rating').fillna(0)
    
    print(f"✅ Recommendation Matrix Created (Users: {len(user_item_matrix)})")

    # 4. SAVE ALL FILES
    print("\nSaving model files...")
    files_to_save = {
        'reg_model.pkl': reg_model,
        'clf_model.pkl': clf_model,
        'le_vmode.pkl': le_vmode,
        'user_item_matrix.pkl': user_item_matrix
    }

    for filename, obj in files_to_save.items():
        with open(filename, 'wb') as f:
            pickle.dump(obj, f)
        print(f"✔️  Saved: {filename}")

    print("\n--- ALL MODELS READY! Run 'streamlit run app.py' now. ---")

if __name__ == "__main__":
    run_modeling_and_evaluation()
