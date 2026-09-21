# Phân công chạy competitor

Mỗi competitor được chạy trên **toàn bộ 8 dataset**, full data, kernel `l05_exponential_kernel` và 3 seed `42, 43, 44`.

Có tổng cộng **15 competitor**, phân công như sau:

| Thành viên | Số lượng | Competitor |
|---|---:|---|
| Cường | 3 | `HHH`, `FTTransformer`, `KNN` |
| Hiếu | 4 | `HHHv2`, `TabNet`, `LDA`, `SGD` |
| Tài | 3 | `SpectralNFST`, `SAINT`, `GauNB` |
| Khoa | 5 | `LGBM`, `MLP`, `NuSVC`, `NC`, `RNC` |

## 1. Chạy KNFST đúng một lần

Một người đại diện chạy:

```powershell
powershell -ExecutionPolicy Bypass -File tools\knfst\run_knfst.ps1
```

Lệnh này tạo một file `models/KNFST.csv` gồm 24 dòng: 8 dataset × 3 seed. Không cần chạy lại khi đổi competitor.

## 2. Chạy một competitor

Ví dụ chạy KNN trên toàn bộ dataset:

```powershell
powershell -ExecutionPolicy Bypass -File tools\knfst\compare.ps1 KNN
```

Mỗi người thay `KNN` bằng từng competitor được phân công.

Lệnh competitor chỉ chạy model được chọn, không chạy lại KNFST. Ví dụ `HHHv2` được lưu thành một file `models/HHHv2.csv` gồm 24 dòng; seed 42, 43, 44 là các dòng riêng.

Mỗi model chỉ có một file kết quả:

```text
results/knfst_comparison/models/<Model>.csv
```

Mỗi thành viên chỉ cần gửi các file `<Model>.csv` của mình. Người tổng hợp chép tất cả vào cùng thư mục `models` rồi chạy lệnh xếp hạng.

File tổng hợp dùng để báo cáo:

```text
results/knfst_comparison/all_datasets_l05_exponential_kernel_full.csv
```

Các cột chính:

| Cột | Ý nghĩa |
|---|---|
| `Seed`, `Data Type`, `Model` | Seed, dataset và model |
| `Kernel`, `SCALER` | Kernel và scaler đã dùng |
| `MCC`, `F1 Macro` | Metric chính, càng cao càng tốt |
| `ACC`, `TPR Macro`, `PPV Macro` | Các metric phân loại bổ sung |
| `FPR` | Tỷ lệ false positive, càng thấp càng tốt |
| `Training time`, `Test time` | Thời gian train và dự đoán |
| `CFS Matrix` | Confusion matrix |

## 3. Xem xếp hạng chung

```powershell
powershell -ExecutionPolicy Bypass -File tools\knfst\show_ranking.ps1
```

File xếp hạng được lưu tại:

```text
results/knfst_comparison/all_datasets_l05_exponential_kernel_full_ranking.csv
```

Các cột:

```text
Rank, Model, Datasets, Runs, MCC_Mean, MCC_Std,
F1_Mean, F1_Std, FPR_Mean, Train_s_Mean, Test_s_Mean
```
