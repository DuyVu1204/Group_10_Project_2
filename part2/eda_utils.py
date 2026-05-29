import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", palette="muted")

# 0. CLEAN NUMERIC STRINGS  
def clean_numeric_strings(df: pd.DataFrame) -> pd.DataFrame:
    df_clean = df.copy()

    if "RAM" in df_clean.columns:
        df_clean["RAM"] = (
            df_clean["RAM"]
            .astype(str)
            .str.extract(r"^(\d+)")[0]          
        )
        df_clean["RAM"] = pd.to_numeric(df_clean["RAM"], errors="coerce")
        df_clean.loc[~df_clean["RAM"].between(1, 512), "RAM"] = np.nan

    if "Screen_Size" in df_clean.columns:
        df_clean["Screen_Size"] = (
            df_clean["Screen_Size"]
            .astype(str)
            .str.replace(r'"', "", regex=True)   
            .str.strip()
        )
        df_clean["Screen_Size"] = pd.to_numeric(
            df_clean["Screen_Size"], errors="coerce"
        )
        invalid_mask = ~df_clean["Screen_Size"].between(6, 25)
        df_clean.loc[invalid_mask, "Screen_Size"] = np.nan

    return df_clean

# 1. THÔNG TIN CƠ BẢN
def basic_info(df: pd.DataFrame) -> None:

    print("=" * 60)
    print(f"Shape : {df.shape[0]} dòng  ×  {df.shape[1]} cột")
    print("=" * 60)

    dtype_df = pd.DataFrame({
        "Column"  : df.columns,
        "Dtype"   : df.dtypes.values,
        "Non-Null": df.notnull().sum().values,
        "Null"    : df.isnull().sum().values,
    })
    print("\n── Kiểu dữ liệu và null count ──")
    print(dtype_df.to_string(index=False))

    print("\n── 5 dòng đầu ──")
    print(df.head().to_string())

# 2. THỐNG KÊ MÔ TẢ
def descriptive_stats(df: pd.DataFrame) -> pd.DataFrame:

    num_cols = df.select_dtypes(include="number").columns.tolist()
    if not num_cols:
        print("Không có cột numeric.")
        return pd.DataFrame()

    stats = df[num_cols].agg(["mean", "median", "std", "min", "max"])
    stats.loc["Q1"] = df[num_cols].quantile(0.25)
    stats.loc["Q3"] = df[num_cols].quantile(0.75)

    print("── Thống kê mô tả (numeric) ──")
    print(stats.round(3).to_string())
    return stats

# 3. KIỂM TRA TRÙNG LẮP
def duplicate_count(df: pd.DataFrame) -> int:

    n = df.duplicated().sum()
    print(f"Số dòng trùng lắp : {n} / {len(df)}  ({n / len(df) * 100:.2f}%)")
    if n > 0:
        print("  → Cần xem xét loại bỏ trước khi train.")
    else:
        print("  → Không có dòng trùng lắp.")
    return int(n)


# 4. MISSING VALUES
def missing_summary(df: pd.DataFrame) -> pd.DataFrame:

    total   = df.isnull().sum()
    percent = (total / len(df) * 100).round(2)
    summary = pd.DataFrame({
        "Missing Count"  : total,
        "Missing Ratio %": percent,
        "Flag (>= 5%)"   : percent >= 5.0,
    }).sort_values("Missing Ratio %", ascending=False)
    summary = summary[summary["Missing Count"] > 0]

    print("── Bảng Missing Values ──")
    print(summary.to_string())
    print(f"\nTổng cột có missing : {len(summary)}")
    print(f"Cột missing >= 5%   : {int(summary['Flag (>= 5%)'].sum())}")
    return summary


def plot_missing_values(df: pd.DataFrame) -> None:

    missing = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
    missing = missing[missing > 0]

    if missing.empty:
        print("Không có missing values.")
        return

    colors = ["#e74c3c" if v >= 5 else "#f39c12" for v in missing.values]

    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(missing.index, missing.values, color=colors, edgecolor="white", width=0.6)

    for bar, val in zip(bars, missing.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            f"{val:.1f}%",
            ha="center", va="bottom", fontsize=9,
        )

    ax.axhline(5, color="red", linestyle="--", linewidth=1.2, label="Ngưỡng 5%")
    ax.set_title("Tỉ lệ Missing Values theo từng cột", fontsize=14, fontweight="bold")
    ax.set_xlabel("Cột", fontsize=11)
    ax.set_ylabel("Missing (%)", fontsize=11)
    ax.legend(fontsize=10)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.show()


# 5. HISTOGRAM
def plot_histograms(df: pd.DataFrame, columns: list) -> None:

    from scipy.stats import gaussian_kde

    n     = len(columns)
    ncols = 2
    nrows = (n + 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4 * nrows))
    axes = np.array(axes).flatten()

    for i, col in enumerate(columns):
        data = df[col].dropna()
        if data.empty:
            axes[i].set_title(f"{col} — không có dữ liệu")
            continue

        axes[i].hist(
            data, bins=40, color="#3498db", edgecolor="white",
            density=True, alpha=0.7, label="Histogram",
        )
        # KDE
        if data.std() > 0:
            kde      = gaussian_kde(data)
            x_range  = np.linspace(data.min(), data.max(), 300)
            axes[i].plot(x_range, kde(x_range), color="#e74c3c", linewidth=2, label="KDE")

        axes[i].set_title(f"Phân phối: {col}", fontsize=11, fontweight="bold")
        axes[i].set_xlabel(col, fontsize=10)
        axes[i].set_ylabel("Mật độ", fontsize=10)
        axes[i].legend(fontsize=9)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Histogram + KDE của các biến numeric", fontsize=14,
                 fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.show()


# 6. BOXPLOT
def plot_boxplots(df: pd.DataFrame, columns: list) -> None:

    n    = len(columns)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, col in zip(axes, columns):
        data = df[col].dropna()
        if data.empty:
            ax.set_title(f"{col} — không có dữ liệu")
            continue

        ax.boxplot(
            data, patch_artist=True,
            boxprops     = dict(facecolor="#3498db", color="#2c3e50"),
            medianprops  = dict(color="#e74c3c", linewidth=2),
            whiskerprops = dict(color="#2c3e50"),
            capprops     = dict(color="#2c3e50"),
            flierprops   = dict(marker="o", color="#e74c3c", alpha=0.4, markersize=4),
        )
        mean_val = data.mean()
        ax.scatter([1], [mean_val], color="#f39c12", zorder=5,
                   s=60, label=f"Mean: {mean_val:.2f}")

        ax.set_title(f"Boxplot: {col}", fontsize=11, fontweight="bold")
        ax.set_xlabel(col, fontsize=10)
        ax.set_ylabel("Giá trị", fontsize=10)
        ax.set_xticks([])
        ax.legend(fontsize=9)

    plt.suptitle("Boxplot của các biến numeric", fontsize=14,
                 fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.show()


# 7. CORRELATION HEATMAP
def plot_correlation_heatmap(df: pd.DataFrame) -> None:

    numeric_df = df.select_dtypes(include="number")
    if numeric_df.empty:
        print("Không có cột numeric để vẽ heatmap.")
        return

    corr = numeric_df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))   

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f",
        cmap="coolwarm", center=0, vmin=-1, vmax=1,
        linewidths=0.5, ax=ax, annot_kws={"size": 11},
    )
    ax.set_title("Ma trận tương quan Pearson (numeric features)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.show()

    target = "Price"
    if target in corr.columns:
        print(f"\n── Tương quan với {target} ──")
        corr_target = (
            corr[target].drop(target)
            .sort_values(key=abs, ascending=False)
        )
        print(corr_target.round(3).to_string())


# 8. PHÁT HIỆN OUTLIERS — IQR
def detect_outliers_iqr(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    records = []
    for col in columns:
        data = pd.to_numeric(df[col], errors="coerce").dropna()
        if data.empty:
            continue

        q1    = data.quantile(0.25)
        q3    = data.quantile(0.75)
        iqr   = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        n_out = int(((data < lower) | (data > upper)).sum())

        records.append({
            "Column"         : col,
            "Q1"             : round(q1, 3),
            "Q3"             : round(q3, 3),
            "IQR"            : round(iqr, 3),
            "Lower Bound"    : round(lower, 3),
            "Upper Bound"    : round(upper, 3),
            "Outlier Count"  : n_out,
            "Outlier Ratio %": round(n_out / len(data) * 100, 2),
        })

    result = pd.DataFrame(records).set_index("Column")
    print("── Bảng Outliers (IQR method) ──")
    print(result.to_string())
    return result

# 9. PHÂN PHỐI TARGET THEO CATEGORICAL 
def plot_target_by_category(
    df: pd.DataFrame,
    cat_columns: list,
    target: str = "Price",
) -> None:

    for col in cat_columns:
        top_vals = df[col].value_counts().head(10).index
        subset   = df[df[col].isin(top_vals)].copy()
        order    = (
            subset.groupby(col)[target]
            .median()
            .sort_values(ascending=False)
            .index
        )

        fig, ax = plt.subplots(figsize=(12, 4))
        sns.boxplot(data=subset, x=col, y=target, order=order,
                    hue=col, palette="muted", legend=False, ax=ax)
        ax.set_title(f"{target} theo {col} (top 10 giá trị phổ biến)",
                     fontsize=12, fontweight="bold")
        ax.set_xlabel(col, fontsize=10)
        ax.set_ylabel(f"{target} (USD)", fontsize=10)
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.show()