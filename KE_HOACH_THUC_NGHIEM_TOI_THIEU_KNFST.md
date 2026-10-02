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

## 4. Phân công chạy Spectral NFST theo gamma trên full data

Thí nghiệm này giữ nguyên model `SpectralNFST` và chỉ chạy grid:

- kernel cố định: `l05_exponential_kernel`;
- 6 gamma: `heuristic`, `0.001`, `0.01`, `0.1`, `1`, `10`;
- 2 cấu hình feature: `-1`, `0`;
- chỉ dùng một seed cố định: `42`;
- tổng cộng 12 cấu hình cho mỗi dataset;
- luôn dùng full data, không giới hạn 100 mẫu/lớp.

Lệnh terminal mẫu duy nhất:

```powershell
python code/tune_spectral_nfst.py --dataset BoT_IoT --limit 1000
```

Mỗi người thay tên dataset và limit theo bảng phân công:

| Thành viên | Dataset 1 | Limit | Dataset 2 | Limit |
|---|---|---:|---|---:|
| Tài | `Edge_IIoTset` | 1000 | `BoT_IoT` | 1000 |
| Hiếu | `ToN_IoT` | 1000 | `CIC_IoT2023` | 1000 |
| Cường | `IoTID20` | 2000 | `N_BaIoT` | 1000 |
| Khoa | `5G_NIDD` | 1000 | `UNSW_NB15` | 1000 |

Mỗi người chạy lần lượt hai dataset được giao, không chạy hai job full-data song
song trên cùng máy. Script mặc định full data và seed 42, không cần truyền thêm
`--samples-per-class` hoặc `--seed`.

Kết quả được lưu riêng theo dataset tại:

```text
results/spectral_nfst/gamma_<Dataset>_<Limit>_full_seed42.csv
```

Nếu tiến trình bị dừng, chạy lại đúng lệnh cũ. Script sẽ bỏ qua các cặp
`feature × gamma` đã có trong CSV và tiếp tục phần còn thiếu.

Nếu một cấu hình gặp lỗi số học như `SVD did not converge`, script ghi lỗi vào
file `gamma_<Dataset>_<Limit>_full_seed42_errors.csv` rồi tiếp tục cấu hình
kế tiếp; model `SpectralNFST` không bị sửa.

## 5. Chọn Q tốt nhất và so sánh với competitor

Codebase đã có đủ kết quả để chọn riêng `gamma`, `feature` và `Q` tốt nhất cho
cả 8 dataset. Tất cả cấu hình dùng `l05_exponential_kernel` và
`QuantileTransformer`. Không chạy lại grid gamma hoặc grid Q. Cấu hình được tái
sử dụng:

| Dataset | Gamma | Feature | Q |
|---|---:|---:|---:|
| `BoT_IoT_1000` | 0.01 | -1 | 1 |
| `CIC_IoT2023_1000` | 0.1 | -1 | 1 |
| `ToN_IoT_1000` | 0.001 | 0 | 2 |
| `UNSW_NB15_1000` | 0.1 | 0 | 2 |
| `IoTID20_2000` | 0.01 | -1 | 2 |
| `N_BaIoT_1000` | 0.01 | -1 | 1 |
| `Edge_IIoTset_1000` | 0.1 | 0 | 5 |
| `5G_NIDD_1000` | 0.01 | -1 | 1 |

Chỉ cần một lệnh:

```powershell
python code/compare_spectral_nfst_competitors.py
```

Script tự đọc các CSV tuning cũ. Ba dataset có Q=2 tái sử dụng trực tiếp metric
full-data từ vòng gamma; năm dataset còn lại chỉ chạy một cấu hình kết hợp cuối
cùng. Kết quả final được cache nên chạy lại lệnh sẽ không train lại. Phần so
sánh chỉ lấy seed 42, dùng đúng 8 biến thể dữ liệu giống competitor, và bỏ qua
file `SpectralNFST.csv` cũ đang lỗi conflict. Kết quả chính:

```text
results/knfst_comparison/models/SpectralNFST_Tuned.csv
results/knfst_comparison/seed42_spectral_tuned_all_models.csv
results/knfst_comparison/seed42_spectral_tuned_ranking.csv
```
