# Giải Thích Missing Values & Outliers
## Phần 2, Mục 2.2.2 và 2.2.3

---

## 1. Tổng Quan Missing Values

Dataset Laptop Price April 2024 có **8/10 cột** chứa giá trị thiếu.
Bốn cột vượt ngưỡng 5% có ý nghĩa xử lý là `GPU` (18.44%),
`GPU_Type` (18.16%), `Resolution` (16.60%) và `RAM` (6.30%).
Bảng dưới tóm tắt toàn bộ tình trạng missing:

| Cột | Missing | % | Cơ chế | Phương pháp xử lý |
|---|---|---|---|---|
| GPU | 734 | 18.44% | MAR | Điền `"Unknown"` |
| GPU_Type | 723 | 18.16% | MAR | Điền `"Unknown"` |
| Resolution | 661 | 16.60% | MAR | Điền `"Unknown"` |
| RAM | 251 | 6.30% | MAR | Median (sau clean) |
| Screen_Size | 196 | 4.92% | MAR | Median (sau clean) |
| Processor | 101 | 2.54% | MAR | Điền `"Unknown"` |
| Brand | 32 | 0.80% | MCAR | Mode |
| Price | 1 | 0.03% | MCAR | Xóa dòng |

---

## 2. Phân Loại Cơ Chế Missing — Giải Thích Chi Tiết

### 2.1 GPU và GPU_Type — MAR (Missing At Random)

Hai cột `GPU` và `GPU_Type` có tỉ lệ missing cao nhất trong dataset
(18.44% và 18.16%), và chúng missing **đồng thời** trên phần lớn các dòng.
Điều này không phải ngẫu nhiên hoàn toàn: laptop sử dụng GPU tích hợp
trực tiếp vào CPU (ví dụ Intel Core i5 thế hệ 11 trở lên với Intel Iris Xe)
thường không được người bán liệt kê thông tin GPU riêng, vì GPU không
phải là card rời độc lập.

Xác suất thiếu của `GPU` phụ thuộc vào giá trị `Processor` và `Condition`
— đây là các biến đã quan sát được — do đó phân loại cơ chế là **MAR**.

**Quyết định xử lý**: Điền giá trị `"Unknown"` thay vì xóa dòng.
Lý do: xóa 18% dữ liệu sẽ làm mất 734 quan sát, ảnh hưởng nghiêm trọng
đến kích thước tập huấn luyện. Giá trị `"Unknown"` sau khi one-hot encoding
sẽ trở thành một category riêng, cho phép model học được pattern từ
nhóm laptop không có thông tin GPU.

### 2.2 Resolution — MAR (Missing At Random)

Tỉ lệ missing 16.60%. Qua phân tích EDA, nhận thấy laptop `Condition`
là refurbished hoặc used có tỉ lệ thiếu `Resolution` cao hơn đáng kể so
với laptop `New` hay `Open box`. Điều này phản ánh thực tế: người bán
laptop cũ thường không điền đầy đủ thông số kỹ thuật chi tiết.

Xác suất thiếu có tương quan với `Condition` (biến quan sát được)
→ phân loại **MAR**.

**Quyết định xử lý**: Điền `"Unknown"` — tương tự cách xử lý GPU.

### 2.3 RAM — MAR (Missing At Random)

Tỉ lệ missing 6.30% (251 dòng). `RAM` là thông số cơ bản nên ít bị bỏ
sót hơn GPU. Tuy nhiên, ngoài missing thật sự, cột này còn chứa **51 dòng
có giá trị bẩn** dạng string như `"16GB"`, `"8GB,"`, `"16gb"`, `"Up"`,
`"upto"` — sau khi clean, các giá trị không hợp lệ này sẽ trở thành NaN,
nâng tỉ lệ missing thực tế lên cao hơn.

Xác suất thiếu có tương quan với chất lượng listing của người bán
→ phân loại **MAR**.

**Quyết định xử lý**:
1. Clean trước: dùng regex `^(\d+)` để trích xuất phần số đầu chuỗi,
   loại giá trị ngoài range [1, 512] GB.
2. Impute bằng **median** (robust hơn mean khi có outlier).

### 2.4 Screen_Size — MAR (Missing At Random)

Tỉ lệ missing 4.92% (196 dòng) — dưới ngưỡng 5% nhưng vẫn cần xử lý.
Tương tự RAM, cột này chứa giá trị bẩn như `"14\""`, `"Does Not Apply"`,
`"N\A"` — sau clean, tỉ lệ missing thực tế tăng lên.

**Quyết định xử lý**:
1. Clean trước: bỏ dấu `"`, ép float, lọc range [6, 25] inch.
2. Impute bằng **median**.

### 2.5 Brand và Processor — MCAR (Missing Completely At Random)

Hai cột có tỉ lệ missing rất thấp (0.80% và 2.54%). Không có pattern rõ
ràng nào liên kết việc thiếu dữ liệu với các biến khác, nhiều khả năng
do lỗi nhập liệu ngẫu nhiên → phân loại **MCAR**.

**Quyết định xử lý**: `Brand` → impute bằng **mode** (Dell chiếm 38%).
`Processor` → điền `"Unknown"`.

### 2.6 Price — MCAR (Missing Completely At Random)

Chỉ 1 dòng missing (0.03%). Đây là biến target — không thể impute.

**Quyết định xử lý**: **Xóa dòng** đó trước train/test split.

---

## 3. Phân Tích Outliers

### 3.1 Price = 0 — Lỗi dữ liệu nghiêm trọng

Phát hiện **5 dòng** có `Price = 0.0`. Kiểm tra thủ công cho thấy
đây là các model laptop cao cấp như Dell XPS 15, Razer Blade 16,
Dell Latitude Rugged — giá thị trường thực tế từ 800–2500 USD.
Giá trị 0 rõ ràng là **lỗi nhập liệu**, không phản ánh thực tế.

**Quyết định**: Xóa 5 dòng này **trước** train/test split.

**Lý do**: Giữ lại sẽ làm model học sai pattern (Price = 0 với
cấu hình cao cấp), dẫn đến hệ số hồi quy bị nhiễu.

### 3.2 RAM ≥ 32 GB — Outlier hợp lệ về kỹ thuật

IQR method xác định ngưỡng upper = 28 GB, có khoảng 400 dòng
vượt ngưỡng (RAM = 32, 64, 128 GB). Tuy nhiên, đây là cấu hình
**workstation và gaming laptop** hoàn toàn phổ biến trên thị trường
năm 2024, không phải lỗi dữ liệu.

**Quyết định**: Giữ nguyên.

**Lý do**: Xóa sẽ loại bỏ một phân khúc thị trường thật sự có
ảnh hưởng đến giá. IQR với dữ liệu lệch phân phối thường cho
ngưỡng quá chặt, không phù hợp để dùng làm tiêu chí loại bỏ.

### 3.3 Screen_Size ngoài range [6, 25] — Xử lý bằng filter

Một số giá trị Screen_Size như `39.6` (thực ra là cm, không phải inch)
và `0` (desktop không có màn hình) xuất hiện sau khi clean chuỗi.
Các giá trị này không phải laptop thật sự.

**Quyết định**: Gán NaN cho giá trị ngoài range [6, 25], sau đó
impute bằng median trong pipeline.

### 3.4 Condition = "--" — Giá trị không hợp lệ

1 dòng có `Condition = "--"` — không thuộc bất kỳ category nào có nghĩa.

**Quyết định**: Xóa dòng này trước split.

---

## 4. Tóm Tắt Thứ Tự Xử Lý

```
Bước 1 — Xóa dòng bất hợp lệ (trước split):
    Price = 0          →  xóa 5 dòng
    Price là NaN       →  xóa 1 dòng
    Condition = "--"   →  xóa 1 dòng
    Còn lại: ~3974 dòng

Bước 2 — Clean data dirty (trước pipeline, áp dụng toàn bộ):
    Screen_Size: bỏ dấu ", ép float, filter between(6, 25)
    RAM        : regex ^(\d+), ép float, filter between(1, 512)
    GPU_Type   : gom nhóm về Integrated / Dedicated / Unknown
    Brand      : .str.title() để thống nhất chữ hoa/thường

Bước 3 — Train/Test Split (80/20, random_state=42)

Bước 4 — DataPipeline (chỉ fit trên train, transform cả train + test):
    Screen_Size, RAM   →  median imputation  →  StandardScaler
    GPU, GPU_Type,
    Resolution,
    Processor          →  constant "Unknown"  →  One-Hot Encoding
    Brand              →  mode imputation     →  One-Hot Encoding
    Condition          →  Ordinal Encoding
                           (New=5, Open box=4, Excellent=3,
                            Very Good=2, Good=1, Used=0)
```