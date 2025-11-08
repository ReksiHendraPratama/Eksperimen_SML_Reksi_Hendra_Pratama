import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# --- Konfigurasi Path & URL ---
# 1. Memuat data langsung dari URL (Mengabaikan folder 'winequality_raw' lokal)
RAW_DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"

# 2. Path output LOKAL (di dalam folder 'preprocessing')
PROCESSED_DATA_DIR = os.path.join('preprocessing', 'winequality_preprocessing')

def preprocess_wine_data(raw_data_url: str, output_dir: str) -> None:
    """
    Mengotomatiskan langkah dari Notebook (Kriteria Skilled).
    Strategi: Baca dari URL, Simpan ke Lokal (preprocessing/winequality_preprocessing/)
    """
    
    print(f"Memulai proses preprocessing otomatis...")
    print(f"Memuat dataset dari URL: {raw_data_url}")
    
    try:
        # 1. Memuat data dari URL
        df = pd.read_csv(raw_data_url, sep=';')
    except Exception as e:
        print(f"\nERROR: Gagal memuat data dari URL. Pastikan koneksi internet aktif.")
        print(f"Detail Error: {e}")
        return

    # 2. Menghapus data duplikat
    initial_rows = len(df)
    df.drop_duplicates(inplace=True)
    print(f"Menghapus {initial_rows - len(df)} baris data duplikat.")

    # 3. Binerisasi variabel target
    df['is_good'] = np.where(df['quality'] >= 7, 1, 0)
    df.drop('quality', axis=1, inplace=True)
    
    # 4. Memisahkan X (fitur) dan y (target)
    X = df.drop('is_good', axis=1)
    y = df['is_good']
    feature_columns = X.columns
    
    # 5. Menerapkan StandardScaler
    print("Menerapkan StandardScaler pada fitur...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=feature_columns)
    
    # 6. Menggabungkan kembali
    processed_df = pd.concat([X_scaled_df.reset_index(drop=True), y.reset_index(drop=True)], axis=1)
    
    # 7. Membagi data (Train/Test Split)
    print("Membagi data menjadi 80% Train dan 20% Test...")
    df_train, df_test = train_test_split(
        processed_df, test_size=0.2, random_state=42, stratify=processed_df['is_good']
    )
    
    # 8. Menyimpan data bersih (di dalam folder preprocessing)
    os.makedirs(output_dir, exist_ok=True)
    train_path = os.path.join(output_dir, 'train_processed.csv')
    test_path = os.path.join(output_dir, 'test_processed.csv')
    
    df_train.to_csv(train_path, index=False)
    df_test.to_csv(test_path, index=False)
    
    print("-" * 50)
    print("Preprocessing otomatis selesai! (Kriteria 1 Skilled)")
    print(f"Data Training disimpan di: {train_path}")
    print(f"Data Testing disimpan di: {test_path}")
    print("-" * 50)

if __name__ == "__main__":
    preprocess_wine_data(RAW_DATA_URL, PROCESSED_DATA_DIR)