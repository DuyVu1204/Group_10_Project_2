import numpy as np


def kfold_cv(X: np.ndarray, y: np.ndarray, k: int, fit_intercept: bool = True, random_state: int = None) -> float:
    """
    Thực hiện k-fold cross-validation cho mô hình hồi quy tuyến tính OLS.

    Theo định nghĩa k-fold CV (công thức đồ án, mục 1.4.4):
        CV(k) = (1/k) * Σ_{i=1}^k MSE_i
    với MSE_i là sai số bình phương trung bình trên fold thứ i:
        MSE_i = (1 / |val_i|) * Σ_{j ∈ val_i} (y_j - ŷ_j)^2

    Nghiệm OLS trên mỗi fold train (Normal Equations):
        β̂ = (X_train^T X_train)^{-1} X_train^T y_train
    Giải bằng np.linalg.solve thay vì pinv/lstsq để đúng tinh thần cài đặt từ đầu.

    Tham số:
        X            : Ma trận đặc trưng, numpy.ndarray, shape (n_samples, n_features).
        y            : Vector mục tiêu, numpy.ndarray, shape (n_samples,).
        k            : Số lượng fold, số nguyên >= 2 và <= n_samples.
        fit_intercept: Nếu True, tự động thêm cột 1 vào ma trận thiết kế.
        random_state : Seed để cố định việc xáo trộn dữ liệu (đảm bảo tái lập kết quả).

    Trả về:
        CV score — Mean MSE trung bình trên tất cả các fold (float).
    """
    # ==== Validate input (fail-fast) ====
    if not isinstance(X, np.ndarray):
        raise ValueError("X phải là numpy.ndarray.")
    if not isinstance(y, np.ndarray):
        raise ValueError("y phải là numpy.ndarray.")
    if X.ndim != 2:
        raise ValueError("X phải là ma trận 2 chiều (ndim == 2).")
    if y.ndim != 1:
        raise ValueError("y phải là vector 1 chiều (ndim == 1).")
    if not isinstance(k, int) or k < 2:
        raise ValueError("Số fold k phải là số nguyên >= 2.")

    n = len(y)
    if n == 0 or X.shape[0] != n:
        raise ValueError("X và y phải cùng số mẫu, n > 0.")
    if k > n:
        raise ValueError("Số fold k không được lớn hơn số mẫu.")

    # ==== Xáo trộn chỉ số với seed cố định (reproducible) ====
    rng = np.random.default_rng(random_state)
    indices = np.arange(n)
    rng.shuffle(indices)

    # Chia đều kích thước các fold; các fold đầu nhận thêm 1 mẫu nếu n % k != 0
    fold_sizes = np.full(k, n // k)
    fold_sizes[:n % k] += 1

    current  = 0
    mse_list = []

    for fold_size in fold_sizes:
        start, stop = current, current + fold_size
        val_idx   = indices[start:stop]
        train_idx = np.concatenate([indices[:start], indices[stop:]])

        X_train, y_train = X[train_idx], y[train_idx]
        X_val,   y_val   = X[val_idx],   y[val_idx]

        # Thêm cột hệ số chặn (intercept) nếu cần
        if fit_intercept:
            X_train_ = np.hstack([np.ones((X_train.shape[0], 1)), X_train])
            X_val_   = np.hstack([np.ones((X_val.shape[0],   1)), X_val])
        else:
            X_train_ = X_train
            X_val_   = X_val

        # Nghiệm OLS từ Normal Equations: β̂ = (XᵀX)⁻¹ Xᵀy
        # Dùng np.linalg.solve(XᵀX, Xᵀy) — không dùng pinv/lstsq
        XtX      = X_train_.T @ X_train_
        Xty      = X_train_.T @ y_train
        beta_hat = np.linalg.solve(XtX, Xty)

        y_pred = X_val_ @ beta_hat
        mse    = np.mean((y_val - y_pred) ** 2)
        mse_list.append(mse)

        current = stop

    return float(np.mean(mse_list))


# ================== UNIT TESTS ==================

def test_kfold_cv_simple():
    """Test 1: Dữ liệu tuyến tính hoàn hảo không nhiễu → MSE phải xấp xỉ 0."""
    X = np.array([[0], [1], [2], [3], [4], [5]], dtype=float)
    y = 2.0 * X.flatten() + 1.0
    mse = kfold_cv(X, y, k=3, fit_intercept=True, random_state=123)
    assert np.isclose(mse, 0.0, atol=1e-10), (
        f"MSE phải bằng 0 với dữ liệu không nhiễu, thực tế: {mse}"
    )


def test_kfold_cv_known_result():
    """Test 2: Dữ liệu nhiễu nhỏ → MSE phải nhỏ hơn ngưỡng kỳ vọng."""
    rng = np.random.default_rng(42)
    X = rng.normal(size=(20, 2))
    beta_true = np.array([1.0, -2.0, 0.5])
    X_ = np.hstack([np.ones((20, 1)), X])
    y = X_ @ beta_true + rng.normal(scale=0.1, size=20)
    mse = kfold_cv(X, y, k=4, fit_intercept=True, random_state=42)
    assert mse < 0.02, (
        f"MSE phải < 0.02 với dữ liệu nhiễu nhỏ, thực tế: {mse}"
    )


def test_kfold_cv_k_greater_than_n():
    """Test 3: k > n phải raise ValueError."""
    X = np.array([[1], [2], [3]], dtype=float)
    y = np.array([2.0, 4.0, 6.0])
    try:
        kfold_cv(X, y, k=5, fit_intercept=True, random_state=1)
        assert False, "Hàm không ném ra ngoại lệ khi k > n"
    except ValueError as e:
        assert "Số fold k không được lớn hơn số mẫu" in str(e)


def test_kfold_cv_with_outlier():
    """Test 4: Outlier lớn (+100) → MSE phải lớn hơn 1.0."""
    rng = np.random.default_rng(99)
    X = rng.normal(size=(30, 2))
    beta_true = np.array([1.0, 2.0, -1.0])
    X_ = np.hstack([np.ones((30, 1)), X])
    y = X_ @ beta_true + rng.normal(scale=0.1, size=30)
    y[0] += 100.0
    mse = kfold_cv(X, y, k=3, fit_intercept=True, random_state=99)
    assert mse > 1.0, f"MSE phải > 1.0 khi có outlier +100, thực tế: {mse}"


def test_kfold_cv_reproducibility():
    """Test 5: Cùng random_state → kết quả phải hoàn toàn giống nhau (reproducible)."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 3))
    beta_true = np.array([1.0, -1.0, 2.0, 0.5])
    X_ = np.hstack([np.ones((50, 1)), X])
    y = X_ @ beta_true + rng.normal(scale=0.2, size=50)
    mse1 = kfold_cv(X, y, k=5, fit_intercept=True, random_state=42)
    mse2 = kfold_cv(X, y, k=5, fit_intercept=True, random_state=42)
    assert np.isclose(mse1, mse2), (
        f"Kết quả phải giống nhau với cùng random_state, thực tế: {mse1} vs {mse2}"
    )


if __name__ == "__main__":
    print("Đang chạy kiểm thử cho file cross_validation.py...")
    test_kfold_cv_simple()
    print(" -> test_kfold_cv_simple: PASSED")
    test_kfold_cv_known_result()
    print(" -> test_kfold_cv_known_result: PASSED")
    test_kfold_cv_k_greater_than_n()
    print(" -> test_kfold_cv_k_greater_than_n: PASSED")
    test_kfold_cv_with_outlier()
    print(" -> test_kfold_cv_with_outlier: PASSED")
    test_kfold_cv_reproducibility()
    print(" -> test_kfold_cv_reproducibility: PASSED")
    print("\nAll tests passed!")

    # Demo với dữ liệu giả lập (seed cố định — reproducible)
    rng_demo = np.random.default_rng(42)
    X_demo = rng_demo.normal(size=(100, 3))
    beta_demo = np.array([2.0, -1.0, 0.5, 3.0])
    X_demo_ = np.hstack([np.ones((100, 1)), X_demo])
    y_demo = X_demo_ @ beta_demo + rng_demo.normal(scale=0.2, size=100)
    cv_score = kfold_cv(X_demo, y_demo, k=5, fit_intercept=True, random_state=42)
    print(f"\n[Demo] Mean CV MSE trên dữ liệu giả lập (k=5): {cv_score:.6f}")