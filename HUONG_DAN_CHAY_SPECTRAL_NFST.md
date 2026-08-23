# Hướng dẫn chạy Spectral NFST cho người mới

Tài liệu này dành cho thành viên chỉ cần chạy thí nghiệm, không cần biết Python.
Các lệnh dưới đây dùng cho Windows PowerShell.

## A. Chỉ cần nhớ 4 điều

1. Luôn đứng ở thư mục dự án có các thư mục `code`, `data`, `results`.
2. Tên dataset là `CIC_IoT2023`, không phải `CIC_IoT2023_1000`.
   Số `1000` được nhập riêng bằng `-Limit 1000`.
3. Không mở file kết quả CSV bằng Excel khi chương trình đang chạy.
4. Nếu máy hoặc chương trình dừng giữa chừng, chạy lại đúng lệnh grid cũ. Script
   sẽ bỏ qua kết quả đã có và tiếp tục các cấu hình còn thiếu.

## B. Bước 1 — mở đúng thư mục

Mở PowerShell và chạy:

```powershell
cd C:\Users\Admin\Downloads\CPAI-main\CPAI-main
```

Kiểm tra Python:

```powershell
python --version
```

Nếu chưa cài thư viện, chạy một lần:

```powershell
python -m pip install pandas numpy scipy scikit-learn pytest
```

## C. Bước 2 — chạy thử đúng một cấu hình

Copy nguyên lệnh này:

```powershell
powershell -ExecutionPolicy Bypass -File tools/spectral_nfst/run_one_classification.ps1
```

Lệnh mặc định chạy BoT-IoT, RBF, QuantileTransformer, Q=2 và 100 mẫu/lớp.

Ví dụ tự chọn CIC-IoT2023, Laplacian, StandardScaler, Q=5:

```powershell
powershell -ExecutionPolicy Bypass -File tools/spectral_nfst/run_one_classification.ps1 -Dataset CIC_IoT2023 -Limit 1000 -Kernel laplacian -Scaler StandardScaler -Q 5 -SamplesPerClass 100 -Seed 42
```

Muốn chạy toàn bộ mẫu, dùng `-SamplesPerClass 0`. Chỉ làm việc này sau khi cấu
hình nhỏ đã chạy thành công vì ma trận của Spectral NFST dùng rất nhiều RAM.

## D. Bước 3 — tìm tham số bằng grid

Chạy grid mặc định gồm 27 cấu hình:

- kernel: RBF, Laplacian, Linear;
- scaler: QuantileTransformer, StandardScaler, RobustScaler;
- Q: 1, 2, 3;
- 100 mẫu mỗi lớp.

BoT-IoT:

```powershell
powershell -ExecutionPolicy Bypass -File tools/spectral_nfst/run_grid_classification.ps1 -Dataset BoT_IoT -Limit 1000
```

CIC-IoT2023:

```powershell
powershell -ExecutionPolicy Bypass -File tools/spectral_nfst/run_grid_classification.ps1 -Dataset CIC_IoT2023 -Limit 1000
```

Thử thêm Q=5 và tăng lên 250 mẫu/lớp:

```powershell
powershell -ExecutionPolicy Bypass -File tools/spectral_nfst/run_grid_classification.ps1 -Dataset CIC_IoT2023 -Limit 1000 -Components 1,2,3,5 -SamplesPerClass 250
```

Chọn riêng kernel và scaler:

```powershell
powershell -ExecutionPolicy Bypass -File tools/spectral_nfst/run_grid_classification.ps1 -Dataset CIC_IoT2023 -Limit 1000 -Kernels rbf,laplacian -Scalers QuantileTransformer,StandardScaler -Components 1,2,3,5 -SamplesPerClass 100
```

Tên output mặc định chứa dataset, sample và seed. Ví dụ:

```text
results/spectral_nfst/grid_CIC_IoT2023_1000_sample100_seed42.csv
```

## E. Tiếp tục grid cũ bị lỗi Excel hoặc bị dừng

Trước tiên đóng file CSV trong Excel. Sau đó chạy lại cùng lệnh và chỉ rõ file
kết quả cũ:

```powershell
powershell -ExecutionPolicy Bypass -File tools/spectral_nfst/run_grid_classification.ps1 -Dataset CIC_IoT2023 -Limit 1000 -Components 1,2,3,5 -SamplesPerClass 1000 -Output results/spectral_nfst/grid_CIC_IoT2023_1000.csv
```

Script đọc CSV trước khi chạy:

- dòng `[SKIP]` nghĩa là cấu hình đã có kết quả;
- dòng `[RUN]` nghĩa là cấu hình đang được chạy;
- cấu hình bị `PermissionError` hoặc `KeyboardInterrupt` không có trong CSV nên
  sẽ tự chạy lại;
- nếu CSV vẫn bị Excel khóa, script dừng trước khi train để không mất thời gian.

## F. Xem 10 kết quả tốt nhất

Không cần tự viết `Import-Csv`, `Sort-Object` hoặc ký hiệu `$_`. Chạy:

```powershell
powershell -ExecutionPolicy Bypass -File tools/spectral_nfst/show_top_results.ps1 -InputFile results/spectral_nfst/grid_CIC_IoT2023_1000.csv
```

Muốn xem top 20, thêm `-Top 20`:

```powershell
powershell -ExecutionPolicy Bypass -File tools/spectral_nfst/show_top_results.ps1 -InputFile results/spectral_nfst/grid_CIC_IoT2023_1000.csv -Top 20
```

Ý nghĩa cột:

| Cột | Cách đọc |
|---|---|
| MCC | Metric chính để xếp hạng, càng gần 1 càng tốt |
| F1 Macro | F1 của từng lớp rồi lấy trung bình, càng cao càng tốt |
| ACC | Accuracy tổng hợp của codebase; không nên dùng một mình để chọn model |
| Q | Số component/manifold mục tiêu trong mỗi lớp |
| Train(s) | Thời gian train tính bằng giây |

Nếu MCC bằng nhau, ưu tiên F1 Macro cao hơn. Nếu MCC và F1 vẫn bằng nhau, ưu
tiên Q nhỏ hơn vì mô hình đơn giản hơn.

## G. Quy trình team nên dùng

1. Chạy grid với 100 mẫu/lớp để kiểm tra nhanh.
2. Chọn 3–5 cấu hình có MCC và F1 Macro tốt nhất.
3. Chạy lại các cấu hình tốt với 250 hoặc 1000 mẫu/lớp.
4. Chạy cấu hình cuối với seed 42, 43 và 44.
5. Không chọn tham số chỉ dựa vào ACC.
6. Không mở output CSV bằng Excel trong lúc chạy; muốn xem thì copy CSV sang
   một file `preview.csv` rồi mở bản copy.

## H. Các dataset hợp lệ

```text
BoT_IoT
CIC_IoT2023
ToN_IoT
UNSW_NB15
IoTID20
N_BaIoT
Edge_IIoTset
5G_NIDD
```

Chỉ `ToN_IoT` và `IoTID20` có `-Limit 2000`. Các dataset khác dùng
`-Limit 1000`.

## I. Lỗi thường gặp

### `PermissionError: [Errno 13] Permission denied`

File output đang mở bằng Excel. Đóng file rồi chạy lại cùng lệnh grid.

### `KeyboardInterrupt`

Chương trình bị dừng thủ công, thường do nhấn Ctrl+C. Chạy lại cùng lệnh; script
grid sẽ tiếp tục cấu hình chưa hoàn thành.

### `invalid choice: CIC_IoT2023_1000`

Dùng `-Dataset CIC_IoT2023 -Limit 1000`.

### Máy chậm hoặc hết RAM

Giảm `-SamplesPerClass` xuống 100 hoặc 250. Không chạy nhiều tiến trình Spectral
NFST full-data song song.

