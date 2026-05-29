# Đồ Án 2: Data Fitting và OLS

Repository code chính nằm trong thư mục `Group_10_Project_2`.

Nội dung dự án gồm:
- Part 1: Lý thuyết và cài đặt OLS, Ridge, Lasso, K-fold CV, residual analysis.
- Part 2: Pipeline dữ liệu thực tế, so sánh mô hình, metrics, test.

## 1. Cấu trúc thư mục

```text
Project_2/
	README.md
	Group_10_Project_2/
		requirements.txt
		part1/
			part1_notebook.ipynb
			ols_implementation.py
			ridge_lasso.py
			cross_validation.py
			residual_analysis.py
		part2/
			part2_notebook.ipynb
			run_all.py
			data_pipeline.py
			model_comparison.py
			metrics.py
			tests/
```

## 2. Yêu cầu môi trường

- Python 3.10+ (khuyến nghị 3.11 hoặc 3.12)
- pip mới

## 3. Cài đặt nhanh

Chạy từ thư mục `Project_2`:

```powershell
cd Group_10_Project_2
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 4. Chạy Part 1

### 4.1 Mở notebook Part 1

```powershell
cd part1
jupyter notebook part1_notebook.ipynb
```

Trong notebook, chạy theo thứ tự từ trên xuống dưới.

### 4.2 Chạy kiểm tra từng module Part 1 (không cần notebook)

```powershell
cd part1
python ols_implementation.py
python ridge_lasso.py
python cross_validation.py
python residual_analysis.py
```

## 5. Chạy Part 2

### 5.1 Chạy toàn bộ pipeline

```powershell
cd part2
python run_all.py
```

### 5.2 Mở notebook Part 2

```powershell
cd part2
jupyter notebook part2_notebook.ipynb
```

### 5.3 Chạy unit tests Part 2

```powershell
cd part2
pytest -v
```

Hoặc chạy từng file test:

```powershell
python tests/test_data_pipeline.py
python tests/test_models.py
python tests/test_metrics.py
```

## 6. Checklist trước khi nộp/bảo vệ

- Cài đủ thư viện từ `requirements.txt`.
- Chạy được `part1_notebook.ipynb` từ đầu đến cuối.
- Chạy được `python part1/ridge_lasso.py`, `python part1/cross_validation.py`, `python part1/residual_analysis.py`.
- Chạy được `python part2/run_all.py`.
- `pytest -v` trong `part2` pass.

## 7. Lỗi thường gặp và cách xử lý

1. Lỗi `ModuleNotFoundError` khi chạy notebook:
- Đảm bảo kernel notebook đang dùng đúng môi trường `.venv`.
- Chạy lại cell import đường dẫn (`sys.path`) trước khi chạy các cell demo.

2. Lỗi `NameError` khi chạy một cell lẻ:
- Chạy lại từ đầu notebook (Restart Kernel and Run All).
- Hoặc đảm bảo cell hiện tại đã import đầy đủ hàm cần dùng.

3. Lỗi khi vẽ biểu đồ trong test:
- Đây là bình thường nếu cell test gọi `residual_plots`.
- Có thể `plt.close('all')` để dọn figure sau test.
