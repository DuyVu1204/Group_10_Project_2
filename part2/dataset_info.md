# Dataset Info

## Thông tin chung

| Mục | Chi tiết |
|---|---|
| **Tên dataset** | Laptop Price Dataset April 2024 |
| **Nguồn** | https://www.kaggle.com/datasets/keremabdullahoglu/laptop-price-dataset-april-2024 |
| **File gốc** | `part2/data/raw/dataset.csv` |
| **Số dòng** | 3981 |
| **Số cột** | 10 |
| **Target** | `Price` (giá bán laptop, đơn vị USD) |
| **Loại bài toán** | Regression (hồi quy) |

---

## Mô tả từng cột

| Cột | Kiểu gốc | Ý nghĩa | Ví dụ giá trị |
|---|---|---|---|
| `Brand` | categorical | Hãng sản xuất | Dell, HP, Lenovo, ASUS, Acer |
| `Product_Description` | text | Mô tả sản phẩm tự do | "Lenovo ThinkPad L15 16GB RAM 512GB SSD" |
| `Screen_Size` | string (dirty) | Kích thước màn hình (inch) | 15.6, 14, `14"`, `Does Not Apply` |
| `RAM` | string (dirty) | Dung lượng RAM (GB) | 16, 8, `16GB`, `16gb`, `Up` |
| `Processor` | categorical | Tên CPU | Intel Core i5-1135G7, AMD Ryzen 5 |
| `GPU` | categorical | Tên card đồ họa | Intel Iris Xe Graphics, NVIDIA RTX 3050 |
| `GPU_Type` | categorical | Loại GPU | Integrated/On-Board, Dedicated Graphics |
| `Resolution` | categorical | Độ phân giải màn hình | 1920 x 1080, 2560 x 1600 |
| `Condition` | categorical | Tình trạng máy | New, Open box, Excellent-Refurbished, Good-Refurbished |
| `Price` | float | **TARGET** — Giá bán (USD) | min=0, mean=521.7, max=999.99 |

### Cột bỏ qua
- `Product_Description` — text tự do, không đưa thẳng vào model

---

## Phân loại features

### Numeric features (sau khi clean)
- `Screen_Size` — inch, range hợp lệ: 6–25
- `RAM` — GB, range hợp lệ: 1–512

### Categorical features
- `Brand` — 20+ hãng (Dell chiếm 38%, HP 19%, Lenovo 19%)
- `Processor` — tên CPU đầy đủ (cần extract thế hệ / hãng)
- `GPU` — tên GPU đầy đủ
- `GPU_Type` — 2 nhóm chính: Integrated / Dedicated (có nhiều giá trị bẩn cần gom nhóm)
- `Resolution` — ~15 giá trị khác nhau
- `Condition` — 6 giá trị: New, Open box, Excellent/Very Good/Good-Refurbished, Used

---

## Missing Values

| Cột | Missing Count | Missing % | Flag ≥ 5% |
|---|---|---|---|
| `GPU` | 734 | 18.44% | ✅ |
| `GPU_Type` | 723 | 18.16% | ✅ |
| `Resolution` | 661 | 16.60% | ✅ |
| `RAM` | 251 | 6.30% | ✅ |
| `Screen_Size` | 196 | 4.92% | — |
| `Processor` | 101 | 2.54% | — |
| `Brand` | 32 | 0.80% | — |
| `Price` | 1 | 0.03% | — |
| `Product_Description` | 1 | 0.03% | — |
| `Condition` | 0 | 0.00% | — |

### Chiến lược imputation (chỉ fit trên train, transform cả train + test)

```python
IMPUTE_STRATEGY = {
    "Screen_Size" : "median",
    "RAM"         : "median",
    "GPU"         : "constant",       # fill_value = "Unknown"
    "GPU_Type"    : "constant",       # fill_value = "Unknown"
    "Resolution"  : "constant",       # fill_value = "Unknown"
    "Processor"   : "constant",       # fill_value = "Unknown"
    "Brand"       : "most_frequent",
}
# Price NaN (1 dòng) → xóa dòng TRƯỚC split
```

---

## Data Dirty cần clean TRƯỚC pipeline

| Cột | Vấn đề | Cách xử lý |
|---|---|---|
| `RAM` | `"16GB"`, `"16gb"`, `"8GB,"`, `"Up"`, `"upto"` | `str.extract(r'^(\d+)')` → ép float → filter `between(1, 512)` |
| `Screen_Size` | `"14\""`, `"15.6\""`, `"Does Not Apply"`, `"N\A"` | Bỏ `"`, ép float → filter `between(6, 25)` |
| `GPU_Type` | Nhiều giá trị bẩn: `"Intel Iris Xe"`, `"Integrated"`, `"Not available"` | Gom về 3 nhóm: `"Integrated"`, `"Dedicated"`, `"Unknown"` |
| `Brand` | `"DELL"` và `"Dell"` tồn tại song song | `.str.title()` để chuẩn hóa chữ hoa/thường |

---

## Outlier cần xử lý TRƯỚC split

| Vấn đề | Số dòng | Quyết định | Lý do |
|---|---|---|---|
| `Price = 0` | 5 dòng | **Xóa** | Lỗi dữ liệu rõ ràng (laptop cao cấp giá 0?) |
| `Price` là NaN | 1 dòng | **Xóa** | Không có target |
| `Condition = "--"` | 1 dòng | **Xóa** | Không thuộc category hợp lệ |
| RAM ≥ 32GB | ~400 dòng | **Giữ** | Workstation/gaming hợp lệ |
| Screen_Size ngoài range | — | Đã xử lý bằng filter `between(6,25)` | — |

---

## Config cho DataPipeline 

```python
TARGET            = "Price"
DROP_COLUMNS      = ["Product_Description"]
NUMERIC_FEATURES  = ["Screen_Size", "RAM"]
CATEGORICAL_FEATURES = [
    "Brand", "Processor", "GPU", "GPU_Type", "Resolution", "Condition"
]

# Encoding
# - Brand, GPU, GPU_Type, Resolution, Processor → One-Hot (drop='first')
# - Condition → Ordinal (New=5, Open box=4, Excellent=3, Very Good=2, Good=1, Used=0)

# Scaling
# - Screen_Size, RAM → StandardScaler (z-score)

# Target transform
# - Kiểm tra histogram Price → nếu lệch phải thì log1p transform

RANDOM_STATE = 42
TEST_SIZE    = 0.2
```

---

## Thống kê mô tả Target (Price)

| Chỉ số | Giá trị |
|---|---|
| Count | 3980 |
| Mean | $521.74 |
| Std | $241.28 |
| Min | $0.00 ← lỗi |
| Q1 | $324.56 |
| Median | $498.85 |
| Q3 | $700.00 |
| Max | $999.99 |

> **Nhận xét**: Price phân phối lệch nhẹ sang trái (median < mean). Cần kiểm tra histogram để quyết định có cần log-transform không.