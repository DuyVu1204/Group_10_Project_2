"""
data_pipeline.py -- Task 2: DataPipeline, Preprocessing, VIF, Feature Selection.

Tu cai dat: Imputer, StandardScaler, OneHotEncoder, VIF.
Khong dung sklearn cho cac buoc core.
Chi dung: pandas, numpy.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from Group_10_Project_2.part2.config import RANDOM_STATE, TEST_SIZE


# ==============================================================================
#                          IMPUTERS (tu cai dat)
# ==============================================================================

class SimpleImputer:
    """
    Imputer tu cai dat.
    strategy: "median", "mean", "most_frequent"
    """

    def __init__(self, strategy: str = "median"):
        self.strategy = strategy
        self.fill_values_: dict[str, float | str] = {}

    def fit(self, X: pd.DataFrame, columns: list[str]) -> "SimpleImputer":
        for col in columns:
            if self.strategy == "median":
                self.fill_values_[col] = X[col].median()
            elif self.strategy == "mean":
                self.fill_values_[col] = X[col].mean()
            elif self.strategy == "most_frequent":
                self.fill_values_[col] = X[col].mode().iloc[0] if len(X[col].mode()) > 0 else "Unknown"
            else:
                raise ValueError(f"Unknown strategy: {self.strategy}")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col, val in self.fill_values_.items():
            if col in X.columns:
                X[col] = X[col].fillna(val)
        return X


# ==============================================================================
#                      STANDARD SCALER (tu cai dat)
# ==============================================================================

class StandardScaler:
    """
    Z-score scaling: (x - mean) / std
    """

    def __init__(self):
        self.mean_: dict[str, float] = {}
        self.std_: dict[str, float] = {}

    def fit(self, X: pd.DataFrame, columns: list[str]) -> "StandardScaler":
        for col in columns:
            self.mean_[col] = X[col].mean()
            self.std_[col] = X[col].std()
            if self.std_[col] == 0:
                self.std_[col] = 1.0  # tranh chia 0
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col in self.mean_:
            if col in X.columns:
                X[col] = (X[col] - self.mean_[col]) / self.std_[col]
        return X


# ==============================================================================
#                    ONE-HOT ENCODER (tu cai dat)
# ==============================================================================

class OneHotEncoder:
    """
    One-hot encoding tu cai dat.
    handle_unknown="ignore": unknown categories -> all zeros.
    drop_first=True: bo cot dau de tranh multicollinearity.
    max_categories: chi giu lai top categories, cac loai hiem se gom thanh 'Other' hoac bi drop neu drop_first.
    """

    def __init__(self, handle_unknown: str = "ignore", drop_first: bool = True, max_categories: int = None):
        self.handle_unknown = handle_unknown
        self.drop_first = drop_first
        self.max_categories = max_categories
        self.categories_: dict[str, list] = {}
        self.frequent_cats_: dict[str, set] = {}

    def fit(self, X: pd.DataFrame, columns: list[str]) -> "OneHotEncoder":
        for col in columns:
            if self.max_categories is not None:
                val_counts = X[col].dropna().value_counts()
                top_cats = val_counts.nlargest(self.max_categories - 1).index.tolist()
                self.frequent_cats_[col] = set(top_cats)
                
                # Transform internally just to find all resulting categories
                temp_col = X[col].apply(lambda x: x if x in self.frequent_cats_[col] else "Other")
                cats = sorted(temp_col.dropna().unique().tolist())
            else:
                cats = sorted(X[col].dropna().unique().tolist())
                self.frequent_cats_[col] = set(cats)

            self.categories_[col] = cats
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        new_cols = {}
        drop_cols = []

        for col, cats in self.categories_.items():
            if col not in X.columns:
                continue
                
            # Gom nhom neu co max_categories
            if self.max_categories is not None:
                mapped_col = X[col].apply(lambda x: x if pd.isna(x) else (x if x in self.frequent_cats_[col] else "Other"))
            else:
                mapped_col = X[col]
                
            use_cats = cats[1:] if self.drop_first else cats
            for cat in use_cats:
                col_name = f"{col}_{cat}"
                new_cols[col_name] = (mapped_col == cat).astype(float)
            drop_cols.append(col)

        X = X.drop(columns=drop_cols, errors="ignore")
        new_df = pd.DataFrame(new_cols, index=X.index)
        X = pd.concat([X, new_df], axis=1)
        return X


# ==============================================================================
#                         DATA PIPELINE
# ==============================================================================

class DataPipeline:
    """
    Pipeline: Imputation -> Encoding -> Scaling.
    Fit chi tren train. Transform tren train va test.
    Khong data leakage.
    """

    def __init__(
        self,
        numeric_features: list[str],
        categorical_features: list[str],
        numeric_strategy: str = "median",
        categorical_strategy: str = "most_frequent",
        scale_numeric: bool = True,
        max_categories: int = 20,
    ):
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
        self.scale_numeric = scale_numeric

        self.num_imputer = SimpleImputer(strategy=numeric_strategy)
        self.cat_imputer = SimpleImputer(strategy=categorical_strategy)
        self.scaler = StandardScaler() if scale_numeric else None
        self.encoder = OneHotEncoder(handle_unknown="ignore", drop_first=True, max_categories=max_categories)

        self._is_fitted = False
        self.feature_names_: list[str] = []

    def fit(self, X_train: pd.DataFrame) -> "DataPipeline":
        """Fit tren train only."""
        # Fit imputers
        if self.numeric_features:
            self.num_imputer.fit(X_train, self.numeric_features)
        if self.categorical_features:
            self.cat_imputer.fit(X_train, self.categorical_features)

        # Apply imputation de fit encoder va scaler
        X_temp = self.num_imputer.transform(X_train)
        X_temp = self.cat_imputer.transform(X_temp)

        # Fit encoder
        if self.categorical_features:
            self.encoder.fit(X_temp, self.categorical_features)

        # Apply encoding de fit scaler
        X_temp = self.encoder.transform(X_temp)

        # Fit scaler tren tat ca numeric columns (bao gom one-hot)
        if self.scaler is not None and self.numeric_features:
            self.scaler.fit(X_temp, self.numeric_features)

        self.feature_names_ = list(X_temp.columns)
        self._is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform (train hoac test)."""
        if not self._is_fitted:
            raise ValueError("Pipeline chua duoc fit! Goi fit() truoc.")

        X_out = X.copy()

        # 1. Impute
        X_out = self.num_imputer.transform(X_out)
        X_out = self.cat_imputer.transform(X_out)

        # 2. Encode
        X_out = self.encoder.transform(X_out)

        # 3. Scale numeric
        if self.scaler is not None:
            X_out = self.scaler.transform(X_out)

        # Dam bao cung columns voi luc fit
        for col in self.feature_names_:
            if col not in X_out.columns:
                X_out[col] = 0.0
        X_out = X_out[self.feature_names_]

        return X_out

    def fit_transform(self, X_train: pd.DataFrame) -> pd.DataFrame:
        self.fit(X_train)
        return self.transform(X_train)

    def _get_feature_names(self) -> list[str]:
        return self.feature_names_


# ==============================================================================
#                      HELPER FUNCTIONS
# ==============================================================================

def split_features_target(df: pd.DataFrame, target_column: str):
    """Tach X va y."""
    y = df[target_column].copy()
    X = df.drop(columns=[target_column]).copy()
    return X, y


def train_test_split_data(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE):
    """
    Train/test split tu cai dat (khong dung sklearn).
    Shuffle va split theo ti le.
    """
    n = len(X)
    indices = np.arange(n)
    rng = np.random.RandomState(random_state)
    rng.shuffle(indices)

    n_test = int(n * test_size)
    test_idx = indices[:n_test]
    train_idx = indices[n_test:]

    if isinstance(X, pd.DataFrame):
        X_train = X.iloc[train_idx].reset_index(drop=True)
        X_test = X.iloc[test_idx].reset_index(drop=True)
    else:
        X_train = X[train_idx]
        X_test = X[test_idx]

    if isinstance(y, pd.Series):
        y_train = y.iloc[train_idx].reset_index(drop=True)
        y_test = y.iloc[test_idx].reset_index(drop=True)
    else:
        y_train = y[train_idx]
        y_test = y[test_idx]

    return X_train, X_test, y_train, y_test


def clean_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Lam sach du lieu tho tu raw_ebay.csv:
    - Chuan hoa RAM thanh so (GB).
    - Chuan hoa Screen_Size thanh float.
    - Chuan hoa Brand (DELL -> Dell).
    - Loai dong khong co Price hoac Price <= 0.
    - Chuan hoa GPU_Type thanh 2 nhom chinh.
    """
    df = df.copy()

    # --- Price ---
    if "Price" in df.columns:
        df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
        df = df.dropna(subset=["Price"])
        df = df[df["Price"] > 0].reset_index(drop=True)

    # --- RAM ---
    if 'RAM' in df.columns:
        df['RAM'] = df['RAM'].astype(str).str.extract(r'^(\d+)', expand=False)
        df['RAM'] = pd.to_numeric(df['RAM'], errors='coerce')
        df = df[(df['RAM'].isna()) | (df['RAM'].between(1, 512))]

    # --- Screen_Size ---
    if 'Screen_Size' in df.columns:
        df['Screen_Size'] = df['Screen_Size'].astype(str).str.replace('"', '', regex=False)
        df['Screen_Size'] = pd.to_numeric(df['Screen_Size'], errors='coerce')
        df = df[(df['Screen_Size'].isna()) | (df['Screen_Size'].between(6, 25))]

    # --- Brand ---
    if 'Brand' in df.columns:
        df['Brand'] = df['Brand'].str.title()

    # --- GPU_Type: nhom lai ---
    if 'GPU_Type' in df.columns:
        gpu_type_map = {
            'Integrated/On-Board Graphics': 'Integrated',
            'Integrated': 'Integrated',
            'Dedicated Graphics': 'Dedicated',
            'Dedicated': 'Dedicated'
        }
        df['GPU_Type'] = df['GPU_Type'].map(gpu_type_map).fillna('Unknown')

    # --- Condition ---
    if 'Condition' in df.columns:
        df = df[df['Condition'] != '--']

    return df.reset_index(drop=True)


def prepare_data(
    df: pd.DataFrame,
    target_column: str,
    numeric_features: list[str],
    categorical_features: list[str],
    test_size: float = 0.2,
    random_state: int = 42
):
    """
    Full pipeline: clean -> split -> fit pipeline -> transform.

    Returns:
        X_train_processed: pd.DataFrame
        X_test_processed: pd.DataFrame
        y_train: pd.Series
        y_test: pd.Series
        feature_names: list[str]
        pipeline: DataPipeline
    """
    # 1. Clean
    df_clean = clean_raw_data(df)

    # 2. Chi giu cac cot can thiet
    keep_cols = numeric_features + categorical_features + [target_column]
    df_clean = df_clean[keep_cols].copy()

    # 3. Split X / y
    X, y = split_features_target(df_clean, target_column)

    # 4. Train/test split TRUOC preprocessing
    X_train, X_test, y_train, y_test = train_test_split_data(X, y, test_size=test_size, random_state=random_state)

    # 5. Fit pipeline tren train, transform ca hai
    pipeline = DataPipeline(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        numeric_strategy="median",
        categorical_strategy="most_frequent",
        scale_numeric=True,
        max_categories=20,
    )
    X_train_processed = pipeline.fit_transform(X_train)
    X_test_processed = pipeline.transform(X_test)

    feature_names = pipeline._get_feature_names()

    return X_train_processed, X_test_processed, y_train, y_test, feature_names, pipeline


# ==============================================================================
#                               VIF
# ==============================================================================

def calculate_vif(X: pd.DataFrame) -> pd.DataFrame:
    """
    Tinh Variance Inflation Factor tu cai dat.
    VIF_j = 1 / (1 - R^2_j)
    trong do R^2_j la R^2 khi hoi quy X_j len cac cot con lai.
    """
    X_arr = np.asarray(X, dtype=float)
    n, p = X_arr.shape
    vif_data = []

    for j in range(p):
        # y_j = X[:, j], X_others = cac cot con lai
        y_j = X_arr[:, j]
        X_others = np.delete(X_arr, j, axis=1)

        # Them intercept
        X_b = np.column_stack([np.ones(n), X_others])

        # Dung pseudo-inverse thay vi solve/lstsq de tranh loi
        try:
            pinv = np.linalg.pinv(X_b.T @ X_b)
            beta = pinv @ (X_b.T @ y_j)
        except Exception:
            # Fallback rut gon
            beta = np.zeros(X_b.shape[1])

        y_pred = X_b @ beta
        ss_res = np.sum((y_j - y_pred) ** 2)
        ss_tot = np.sum((y_j - np.mean(y_j)) ** 2)

        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        vif = 1.0 / (1.0 - r2) if r2 < 1.0 else np.inf

        vif_data.append({
            "Feature": X.columns[j] if isinstance(X, pd.DataFrame) else f"X{j}",
            "VIF": vif,
        })

    return pd.DataFrame(vif_data).sort_values("VIF", ascending=False)


def select_features_by_vif(X: pd.DataFrame, threshold: float = 10.0) -> list[str]:
    """
    Lap loai bo feature co VIF cao nhat cho den khi tat ca VIF < threshold.
    Tra ve danh sach selected features.
    """
    X_current = X.copy()
    dropped = []

    while True:
        if X_current.shape[1] <= 1:
            break
        vif_df = calculate_vif(X_current)
        max_vif = vif_df["VIF"].max()
        if max_vif < threshold:
            break
        worst = vif_df.loc[vif_df["VIF"].idxmax(), "Feature"]
        dropped.append(worst)
        X_current = X_current.drop(columns=[worst])

    # Get final VIF
    final_vif = calculate_vif(X_current) if X_current.shape[1] > 0 else pd.DataFrame()
    return list(X_current.columns), final_vif


# ==============================================================================
#                            MAIN TEST
# ==============================================================================

if __name__ == "__main__":
    import os

    # Load raw data
    csv_path = os.path.join(os.path.dirname(__file__), "..", "raw_ebay.csv")
    df = pd.read_csv(csv_path)
    print(f"Raw data: {df.shape}")

    from Group_10_Project_2.part2.config import TARGET_COLUMN, NUMERIC_FEATURES, CATEGORICAL_FEATURES

    X_train, X_test, y_train, y_test, feature_names, pipeline = prepare_data(
        df, TARGET_COLUMN, NUMERIC_FEATURES, CATEGORICAL_FEATURES
    )

    print(f"\nX_train: {X_train.shape}")
    print(f"X_test:  {X_test.shape}")
    print(f"y_train: {y_train.shape}")
    print(f"y_test:  {y_test.shape}")
    print(f"Features ({len(feature_names)}): {feature_names}")
    print(f"\nMissing in X_train: {X_train.isnull().sum().sum()}")
    print(f"Missing in X_test:  {X_test.isnull().sum().sum()}")

    # VIF
    print("\n-- VIF --")
    vif_df = calculate_vif(X_train)
    print(vif_df.to_string(index=False))

    # Select features
    selected = select_features_by_vif(X_train, threshold=10.0)
    print(f"\nSelected features (VIF < 10): {selected}")
