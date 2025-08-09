# train.py
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib

# Baca dataset
df = pd.read_csv("data.csv")  # ganti nama file sesuai dataset kamu

# Pisahkan fitur dan target
X = df.drop("Profit", axis=1)  # ganti "Profit" dengan nama kolom target
y = df["Profit"]

# Tentukan fitur kategori & numerik
categorical_features = X.select_dtypes(include=['object']).columns.tolist()
numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
        ('num', StandardScaler(), numerical_features)
    ]
)

# Pipeline model
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', RandomForestClassifier(random_state=42))
])

# Latih model
pipeline.fit(X, y)

# Simpan model
joblib.dump(pipeline, "pipeline_rfnew.pkl")
print("✅ Model berhasil dilatih dan disimpan ke pipeline_rfnew.pkl")
