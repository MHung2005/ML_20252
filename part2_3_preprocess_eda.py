"""
PHẦN 2 & 3: TIỀN XỬ LÝ DỮ LIỆU + KHÁM PHÁ DỮ LIỆU (EDA)
"""

from __future__ import annotations

import os
import sys
import warnings

import pandas as pd
import numpy as np

try:
    from sklearn.preprocessing import LabelEncoder, MinMaxScaler
except ImportError as exc:
    raise ImportError(
        "Missing dependency: scikit-learn. Install it with: "
        "python -m pip install scikit-learn"
    ) from exc

warnings.filterwarnings("ignore")


def configure_output_encoding() -> None:
    """Make Vietnamese console output work on Windows terminals."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


configure_output_encoding()

PLOT_AVAILABLE = True
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    import seaborn as sns
except Exception as exc:
    PLOT_AVAILABLE = False
    plt = None
    mticker = None
    sns = None
    print(f"[WARN] Tắt EDA biểu đồ vì không thể import matplotlib/seaborn: {exc}")

os.makedirs("outputs", exist_ok=True)

# ════════════════════════════════════════════════════════════════════════════
# PHẦN 2 – TIỀN XỬ LÝ
# ════════════════════════════════════════════════════════════════════════════

def load_raw(path: str = "data/raw_wines.csv") -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    print(f"[Load] {len(df)} dòng, {df.shape[1]} cột")
    print(df.dtypes, "\n")
    return df


# ── 2.1  Làm sạch dữ liệu ──────────────────────────────────────────────────

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    print("=== 2.1 LÀM SẠCH DỮ LIỆU ===")

    if "price_vnd" not in df.columns:
        raise ValueError("raw data must contain the target column: price_vnd")

    defaults = {
        "vintage": np.nan,
        "brand": "Unknown",
        "rating_score": np.nan,
        "alcohol_content": np.nan,
        "volume": np.nan,
        "wine_type": "Unknown",
        "grape_variety": "Unknown",
        "origin_country": "Unknown",
        "region": "Unknown",
    }
    for col, default in defaults.items():
        if col not in df.columns:
            df[col] = default

    # Loại bỏ dòng thiếu biến mục tiêu
    before = len(df)
    df = df.dropna(subset=["price_vnd"]).copy()
    print(f"  Xóa {before - len(df)} dòng thiếu price_vnd  → còn {len(df)}")

    # Loại bỏ giá âm hoặc bằng 0
    df["price_vnd"] = pd.to_numeric(df["price_vnd"], errors="coerce")
    df = df[df["price_vnd"] > 0]
    if df.empty:
        raise ValueError("No valid rows remain after filtering price_vnd > 0")

    # Gán giá trị thiếu cho vintage bằng trung bình theo brand
    df["brand"] = df["brand"].fillna("Unknown").astype(str)
    df["vintage"] = pd.to_numeric(df["vintage"], errors="coerce")
    df["vintage"] = df.groupby("brand")["vintage"].transform(
        lambda x: x.fillna(x.mean())
    )
    vintage_mean = df["vintage"].mean()
    df["vintage"] = df["vintage"].fillna(vintage_mean if pd.notna(vintage_mean) else 2018)
    df["vintage"] = df["vintage"].round().astype(int)

    # Gán thiếu rating_score = trung bình theo brand, rồi toàn bộ
    df["rating_score"] = pd.to_numeric(df["rating_score"], errors="coerce")
    df["rating_score"] = df.groupby("brand")["rating_score"].transform(
        lambda x: x.fillna(x.mean())
    )
    rating_mean = df["rating_score"].mean()
    df["rating_score"] = df["rating_score"].fillna(rating_mean if pd.notna(rating_mean) else 4.0)

    # Xử lý alcohol_content thiếu
    df["alcohol_content"] = pd.to_numeric(df["alcohol_content"], errors="coerce")
    # Nếu nhập dạng % (> 1), chuẩn về tỷ lệ
    mask_pct = df["alcohol_content"] > 1
    df.loc[mask_pct, "alcohol_content"] = df.loc[mask_pct, "alcohol_content"] / 100
    alcohol_mean = df["alcohol_content"].mean()
    df["alcohol_content"] = df["alcohol_content"].fillna(
        alcohol_mean if pd.notna(alcohol_mean) else 0.13
    )

    # volume mặc định 750ml nếu thiếu
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
    df["volume"] = df["volume"].fillna(750.0)

    # Phát hiện và xóa outlier giá (IQR x3)
    Q1, Q3 = df["price_vnd"].quantile(0.25), df["price_vnd"].quantile(0.75)
    IQR = Q3 - Q1
    before = len(df)
    df = df[(df["price_vnd"] >= Q1 - 3 * IQR) & (df["price_vnd"] <= Q3 + 3 * IQR)]
    print(f"  Xóa {before - len(df)} outlier giá (IQR×3)  → còn {len(df)}")

    print(f"  Thiếu sau clean:\n{df.isnull().sum()[df.isnull().sum()>0]}\n")
    return df


# ── 2.2  Tích hợp dữ liệu ──────────────────────────────────────────────────

def integrate_data(df: pd.DataFrame) -> pd.DataFrame:
    print("=== 2.2 TÍCH HỢP / ĐỒNG NHẤT ĐƠN VỊ ===")

    # alcohol_content đã là tỷ lệ 0–1, volume đã là ml
    # Chuẩn hóa chuỗi
    for col in ["wine_type", "grape_variety", "origin_country", "region", "brand"]:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown").str.strip().str.title()

    print(f"  wine_type duy nhất: {df['wine_type'].unique()}")
    return df


# ── 2.3  Chuyển đổi dữ liệu ────────────────────────────────────────────────

def transform_data(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, LabelEncoder], MinMaxScaler]:
    print("\n=== 2.3 CHUYỂN ĐỔI DỮ LIỆU ===")

    # Rời rạc hóa alcohol_content → bins
    df["alcohol_bin"] = pd.cut(
        df["alcohol_content"] * 100,
        bins=[0, 12, 13, 14, 100],
        labels=["Nhẹ(<12%)", "Trung bình(12-13%)", "Cao(13-14%)", "Rất cao(>14%)"],
        include_lowest=True,
    )
    print(f"  alcohol_bin:\n{df['alcohol_bin'].value_counts()}")

    # Mã hóa biến phân loại → Label Encoding
    encoders: dict[str, LabelEncoder] = {}
    cat_cols = ["wine_type", "grape_variety", "origin_country", "region", "brand"]
    for col in cat_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col + "_enc"] = le.fit_transform(df[col].astype(str))
            encoders[col] = le

    # Chuẩn hóa MinMax các thuộc tính số (để dùng cho k-NN)
    scale_cols = ["vintage", "alcohol_content", "volume", "rating_score"]
    scaler = MinMaxScaler()
    df_scaled = df.copy()
    df_scaled[scale_cols] = scaler.fit_transform(df[scale_cols])

    print(f"  Chuẩn hóa xong: {scale_cols}")
    return df, df_scaled, encoders, scaler


# ── 2.4  Giảm chiều / Feature Selection ────────────────────────────────────

FEATURE_COLS = [
    "wine_type_enc", "grape_variety_enc", "origin_country_enc",
    "region_enc", "brand_enc", "vintage", "alcohol_content", "volume", "rating_score",
]
TARGET_COL = "price_vnd"


def select_features(df: pd.DataFrame) -> pd.DataFrame:
    print("\n=== 2.4 GIẢM CHIỀU (FEATURE SELECTION) ===")
    keep = [c for c in FEATURE_COLS if c in df.columns] + [TARGET_COL]
    df_final = df[keep].dropna()
    print(f"  Giữ lại {len(keep)-1} feature + target  ({len(df_final)} mẫu)")
    return df_final


# ════════════════════════════════════════════════════════════════════════════
# PHẦN 3 – EDA
# ════════════════════════════════════════════════════════════════════════════

PALETTE = "#7F77DD"

def eda(df_raw: pd.DataFrame, df_clean: pd.DataFrame) -> None:
    print("\n=== PHẦN 3: EDA ===")

    if not PLOT_AVAILABLE:
        print("  Bỏ qua phần vẽ biểu đồ; chỉ xuất thống kê mô tả.")
        summary = df_clean[[c for c in ["price_vnd", "vintage", "alcohol_content", "volume", "rating_score"] if c in df_clean.columns]].describe().round(3)
        summary.to_csv("outputs/eda_summary.csv", encoding="utf-8-sig")
        print("  ✓ Thống kê mô tả lưu tại outputs/eda_summary.csv")
        print("\n── Thống kê mô tả ──")
        print(summary)
        return

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("Khám Phá Dữ Liệu – Wine Price Dataset", fontsize=14, fontweight="bold")

    # 3.1 Phân phối giá
    ax = axes[0, 0]
    sns.histplot(df_clean["price_vnd"] / 1e6, bins=30, kde=True, color=PALETTE, ax=ax)
    ax.set_title("Phân phối giá bán (VNĐ)")
    ax.set_xlabel("Giá (triệu VNĐ)")
    ax.set_ylabel("Số lượng")
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.1fM"))

    # 3.2 Log-scale giá
    ax = axes[0, 1]
    log_prices = np.log1p(df_clean["price_vnd"])
    sns.histplot(log_prices, bins=30, kde=True, color="#1D9E75", ax=ax)
    ax.set_title("Phân phối log(price_vnd)")
    ax.set_xlabel("log(1 + price_vnd)")
    ax.set_ylabel("")

    # 3.3 Giá theo wine_type
    ax = axes[0, 2]
    order = df_clean.groupby("wine_type")["price_vnd"].median().sort_values(ascending=False).index
    sns.boxplot(data=df_clean, x="wine_type", y="price_vnd",
                order=order, palette="Set2", ax=ax)
    ax.set_title("Giá theo loại rượu")
    ax.set_xlabel("")
    ax.set_ylabel("Giá (VNĐ)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v/1e6:.1f}M"))
    ax.tick_params(axis="x", rotation=20)

    # 3.4 Top 10 brand theo giá trung bình
    ax = axes[1, 0]
    top_brands = (df_clean.groupby("brand")["price_vnd"]
                  .mean().sort_values(ascending=False).head(10))
    sns.barplot(x=top_brands.values / 1e6, y=top_brands.index, color=PALETTE, ax=ax)
    ax.set_title("Top 10 thương hiệu (giá trung bình)")
    ax.set_xlabel("Triệu VNĐ")
    ax.set_ylabel("")

    # 3.5 Scatter: alcohol vs price
    ax = axes[1, 1]
    scatter_data = df_clean[["alcohol_content", "price_vnd"]].dropna()
    ax.scatter(scatter_data["alcohol_content"] * 100,
               scatter_data["price_vnd"] / 1e6,
               alpha=0.4, s=20, color="#D85A30")
    ax.set_title("Nồng độ cồn vs Giá bán")
    ax.set_xlabel("Nồng độ cồn (%)")
    ax.set_ylabel("Giá (triệu VNĐ)")

    # 3.6 Correlation heatmap
    ax = axes[1, 2]
    num_cols = ["price_vnd", "vintage", "alcohol_content", "volume", "rating_score"]
    corr_df = df_clean[[c for c in num_cols if c in df_clean.columns]].corr()
    sns.heatmap(
        corr_df, annot=True, fmt=".2f", cmap="coolwarm",
        center=0, linewidths=0.5, ax=ax, annot_kws={"size": 9},
    )
    ax.set_title("Ma trận tương quan")

    plt.tight_layout()
    plt.savefig("outputs/eda_report.png", dpi=130, bbox_inches="tight")
    plt.close()
    print("  ✓ Biểu đồ EDA lưu tại outputs/eda_report.png")

    # In thống kê mô tả
    print("\n── Thống kê mô tả ──")
    print(df_clean[["price_vnd", "vintage", "alcohol_content", "volume", "rating_score"]]
          .describe().round(3))


# ════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════

def preprocess_and_eda(raw_csv: str = "data/raw_wines.csv"):
    df_raw    = load_raw(raw_csv)
    df_clean  = clean_data(df_raw.copy())
    df_clean  = integrate_data(df_clean)
    df_raw_for_eda = df_clean.copy()   # trước khi encode để EDA còn chữ
    df_clean, df_scaled, encoders, scaler = transform_data(df_clean)
    df_final  = select_features(df_scaled)

    eda(df_raw, df_raw_for_eda)

    # Lưu kết quả
    os.makedirs("data", exist_ok=True)
    df_clean.to_csv("data/cleaned_wines.csv", index=False, encoding="utf-8-sig")
    df_final.to_csv("data/processed_wines.csv", index=False, encoding="utf-8-sig")
    print("\n✓ Lưu: data/cleaned_wines.csv  &  data/processed_wines.csv")
    return df_final, encoders, scaler


if __name__ == "__main__":
    preprocess_and_eda()
