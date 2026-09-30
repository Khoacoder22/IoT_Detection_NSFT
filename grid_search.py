import os
import glob
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import QuantileTransformer, StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef

# 1. Hàm Kernel L0.5
def l05_exponential_kernel(X, Y=None, gamma=0.1):
    X_arr = np.asarray(X, dtype=np.float64)
    Y_arr = X_arr if Y is None else np.asarray(Y, dtype=np.float64)
    
    n_samples_X, n_features = X_arr.shape
    n_samples_Y = Y_arr.shape[0]
    
    dist_l05 = np.zeros((n_samples_X, n_samples_Y), dtype=np.float64)
    for j in range(n_features):
        diff_j = np.abs(X_arr[:, j:j+1] - Y_arr[:, j:j+1].T)
        dist_l05 += np.sqrt(diff_j)
        
    alpha = float(gamma) if gamma is not None else 0.1
    return np.exp(-alpha * dist_l05)

# 2. Định nghĩa Lưới tham số muốn tìm kiếm (Grid Range)
PARAM_GRID = {
    'gamma': [0.01, 0.05, 0.1, 0.2, 0.5, 1.0],      # Các giá trị Gamma thử nghiệm
    'scaler': ['quantile', 'standard', 'minmax']     # Các phương pháp Scaler
}

def get_scaler(scaler_name):
    if scaler_name == 'quantile':
        return QuantileTransformer(output_distribution='normal', random_state=42)
    elif scaler_name == 'standard':
        return StandardScaler()
    elif scaler_name == 'minmax':
        return MinMaxScaler()

print("="*60)
print("     BẮT ĐẦU QUÁ TRÌNH TÌM THAM SỐ TỐI ƯU (GRID SEARCH)    ")
print("="*60)

# 3. Lấy danh sách tất cả các file CSV dataset trong thư mục results/runs
dataset_files = glob.glob(r"results/**/*.csv", recursive=True)
# Hoặc khai báo trực tiếp danh sách các tập dữ liệu của bạn:
datasets = [
    "5G_NIDD_1000", "BoT_IoT_1000", "CIC_IoT2023_1000", 
    "Edge_IIoTset_1000", "IoTID20_2000", "N_BaIoT_1000", 
    "ToN_IoT_1000", "UNSW_NB15_1000"
]

best_results = []

# Mẫu vòng lặp tìm tham số tối ưu cho từng dataset
for ds_name in datasets:
    print(f"\n[+] Đang tối ưu tham số cho Dataset: {ds_name}...")
    
    best_f1 = -1.0
    best_config = {}
    
    # Giả định thử nghiệm tất cả các kết hợp trong PARAM_GRID
    for scaler_name in PARAM_GRID['scaler']:
        for gamma in PARAM_GRID['gamma']:
            
            # --- TẠI ĐÂY LÀ NƠI CHẠY EVALUATE MODEL CỦA BẠN ---
            # Ví dụ: Giả lập đánh giá F1/ACC từ việc chạy model
            # Bạn thay thế bằng hàm evaluate_model_of_your_project(ds_name, gamma, scaler_name)
            
            # (Đoạn này minh họa cách lưu config tốt nhất)
            current_f1 = np.random.uniform(0.85, 0.99) # Thay bằng điểm F1 thật của bạn
            current_acc = current_f1 * 100
            current_mcc = current_f1 - 0.02
            
            if current_f1 > best_f1:
                best_f1 = current_f1
                best_config = {
                    'Dataset': ds_name,
                    'Best_Kernel': 'l05_exponential',
                    'Best_Gamma': gamma,
                    'Best_Scaler': scaler_name,
                    'Best_F1_Macro': round(current_f1, 4),
                    'Best_ACC': round(current_acc, 2),
                    'Best_MCC': round(current_mcc, 4)
                }
    
    best_results.append(best_config)
    print(f"    --> Tham số tốt nhất: Gamma={best_config['Best_Gamma']}, Scaler={best_config['Best_Scaler']} (F1={best_config['Best_F1_Macro']})")

# 4. Xuất kết quả tham số tốt nhất ra file CSV
df_best = pd.DataFrame(best_results)
df_best.to_csv("best_parameters_per_dataset.csv", index=False)

print("\n" + "="*60)
print("ĐÃ HOÀN THÀNH! Kết quả được lưu tại: best_parameters_per_dataset.csv")
print("="*60)
print(df_best.to_string(index=False))