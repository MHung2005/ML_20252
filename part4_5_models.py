"""
PHẦN 4 & 5: MÔ HÌNH HỌC MÁY + ĐÁNH GIÁ & LỰA CHỌN MÔ HÌNH
Gồm: OLS, Ridge, LASSO, k-NN Regression
Đánh giá: Hold-out + k-fold Cross-Validation, chỉ số MAE
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import os, warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib

os.makedirs("outputs", exist_ok=True)
os.makedirs("models", exist_ok=True)

RANDOM_STATE = 42
PALETTE = {
    "OLS": "#7F77DD",
    "Ridge": "#1D9E75",
    "LASSO": "#D85A30",
    "k-NN": "#BA7517",
}

# ════════════════════════════════════════════════════════════════════════════
# PHẦN 4 – XÂY DỰNG MÔ HÌNH
# ════════════════════════════════════════════════════════════════════════════

def load_processed(path: str = "data/processed_wines.csv") -> tuple:
    df = pd.read_csv(path, encoding="utf-8-sig")
    feature_cols = [c for c in df.columns if c != "price_vnd"]
    X = df[feature_cols].values
    y = df["price_vnd"].values
    print(f"[Load processed] X: {X.shape}, y: {y.shape}")
    return X, y, feature_cols


# ── 4.1 Chia tập Train / Validation / Test ──────────────────────────────────

def split_data(X, y):
    """60% Train | 20% Validation | 20% Test"""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.25, random_state=RANDOM_STATE
    )
    print(f"  Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    return X_train, X_val, X_test, y_train, y_val, y_test


# ── 4.2 Tìm siêu tham số tốt nhất trên tập Validation ─────────────────────

def tune_ridge(X_train, y_train, X_val, y_val) -> float:
    lambdas = [0.001, 0.01, 0.1, 1, 10, 100, 500, 1000]
    best_lambda, best_mae = None, float("inf")
    maes = []
    for lam in lambdas:
        model = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=lam))])
        model.fit(X_train, y_train)
        mae = mean_absolute_error(y_val, model.predict(X_val))
        maes.append(mae)
        if mae < best_mae:
            best_mae, best_lambda = mae, lam
    print(f"  Ridge  best λ={best_lambda}  val_MAE={best_mae:,.0f}")
    return best_lambda


def tune_lasso(X_train, y_train, X_val, y_val) -> float:
    lambdas = [1, 10, 100, 500, 1000, 5000, 10000]
    best_lambda, best_mae = None, float("inf")
    for lam in lambdas:
        model = Pipeline([("scaler", StandardScaler()), ("lasso", Lasso(alpha=lam, max_iter=10000))])
        model.fit(X_train, y_train)
        mae = mean_absolute_error(y_val, model.predict(X_val))
        if mae < best_mae:
            best_mae, best_lambda = mae, lam
    print(f"  LASSO  best λ={best_lambda}  val_MAE={best_mae:,.0f}")
    return best_lambda


def tune_knn(X_train, y_train, X_val, y_val) -> int:
    ks = list(range(1, 21))
    best_k, best_mae = None, float("inf")
    maes = []
    for k in ks:
        model = Pipeline([("scaler", StandardScaler()), ("knn", KNeighborsRegressor(n_neighbors=k))])
        model.fit(X_train, y_train)
        mae = mean_absolute_error(y_val, model.predict(X_val))
        maes.append(mae)
        if mae < best_mae:
            best_mae, best_k = mae, k

    # Vẽ biểu đồ k vs MAE
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(ks, [m / 1e6 for m in maes], marker="o", color=PALETTE["k-NN"])
    ax.axvline(best_k, color="red", linestyle="--", label=f"best k={best_k}")
    ax.set_title("k-NN: Số láng giềng k vs Validation MAE")
    ax.set_xlabel("k")
    ax.set_ylabel("MAE (triệu VNĐ)")
    ax.legend()
    plt.tight_layout()
    plt.savefig("outputs/knn_tuning.png", dpi=120, bbox_inches="tight")
    plt.close()

    print(f"  k-NN   best k={best_k}  val_MAE={best_mae:,.0f}")
    return best_k


# ── 4.3 Xây dựng các mô hình cuối ──────────────────────────────────────────

def build_models(best_ridge_lam, best_lasso_lam, best_k) -> dict:
    return {
        "OLS":   Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())]),
        "Ridge": Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=best_ridge_lam))]),
        "LASSO": Pipeline([("scaler", StandardScaler()), ("model", Lasso(alpha=best_lasso_lam, max_iter=10000))]),
        "k-NN":  Pipeline([("scaler", StandardScaler()), ("model", KNeighborsRegressor(n_neighbors=best_k))]),
    }


# ════════════════════════════════════════════════════════════════════════════
# PHẦN 5 – ĐÁNH GIÁ & LỰA CHỌN
# ════════════════════════════════════════════════════════════════════════════

def evaluate_with_cv(models: dict, X_train, y_train, k_fold: int = 5) -> pd.DataFrame:
    """k-fold Cross-Validation trên tập train."""
    print(f"\n=== 5.1 {k_fold}-fold Cross-Validation (Train) ===")
    kf = KFold(n_splits=k_fold, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    for name, pipe in models.items():
        neg_mae = cross_val_score(pipe, X_train, y_train,
                                  cv=kf, scoring="neg_mean_absolute_error")
        mae_scores = -neg_mae
        rows.append({
            "Model": name,
            "CV_MAE_mean":  mae_scores.mean(),
            "CV_MAE_std":   mae_scores.std(),
        })
        print(f"  {name:6s}  MAE = {mae_scores.mean():>12,.0f} ± {mae_scores.std():>10,.0f}")
    return pd.DataFrame(rows)


def evaluate_on_test(models: dict, X_train, y_train, X_test, y_test) -> pd.DataFrame:
    """Huấn luyện trên toàn bộ Train rồi đánh giá trên Test."""
    print("\n=== 5.2 Đánh giá cuối cùng trên Test ===")
    rows = []
    for name, pipe in models.items():
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        mae  = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2   = r2_score(y_test, y_pred)
        rows.append({"Model": name, "MAE": mae, "RMSE": rmse, "R2": r2})
        print(f"  {name:6s}  MAE={mae:>12,.0f}  RMSE={rmse:>12,.0f}  R²={r2:.4f}")

        # Lưu model
        joblib.dump(pipe, f"models/{name.lower().replace('-','_')}_model.pkl")
    return pd.DataFrame(rows)


def plot_results(cv_df: pd.DataFrame, test_df: pd.DataFrame,
                 models: dict, X_train, y_train, X_test, y_test) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    fig.suptitle("Kết Quả Đánh Giá Mô Hình", fontsize=13, fontweight="bold")

    colors = [PALETTE.get(m, "#888") for m in test_df["Model"]]

    # ── Biểu đồ 1: MAE trên Test ──
    ax = axes[0]
    bars = ax.bar(test_df["Model"], test_df["MAE"] / 1e6, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_title("MAE trên tập Test")
    ax.set_ylabel("MAE (triệu VNĐ)")
    for bar, val in zip(bars, test_df["MAE"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{val/1e6:.2f}M", ha="center", va="bottom", fontsize=9)

    # ── Biểu đồ 2: R² trên Test ──
    ax = axes[1]
    bars = ax.bar(test_df["Model"], test_df["R2"], color=colors, edgecolor="white", linewidth=0.5)
    ax.set_title("R² trên tập Test")
    ax.set_ylabel("R²")
    ax.set_ylim(0, 1.1)
    for bar, val in zip(bars, test_df["R2"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    # ── Biểu đồ 3: Actual vs Predicted (mô hình tốt nhất) ──
    best_model_name = test_df.loc[test_df["MAE"].idxmin(), "Model"]
    best_pipe = models[best_model_name]
    best_pipe.fit(X_train, y_train)
    y_pred_best = best_pipe.predict(X_test)

    ax = axes[2]
    ax.scatter(y_test / 1e6, y_pred_best / 1e6,
               alpha=0.5, s=20, color=PALETTE.get(best_model_name, "#888"))
    mn = min(y_test.min(), y_pred_best.min()) / 1e6
    mx = max(y_test.max(), y_pred_best.max()) / 1e6
    ax.plot([mn, mx], [mn, mx], "r--", linewidth=1.5, label="Lý tưởng")
    ax.set_title(f"Thực tế vs Dự đoán – {best_model_name}")
    ax.set_xlabel("Giá thực (triệu VNĐ)")
    ax.set_ylabel("Giá dự đoán (triệu VNĐ)")
    ax.legend()

    plt.tight_layout()
    plt.savefig("outputs/model_comparison.png", dpi=130, bbox_inches="tight")
    plt.close()
    print("✓ Biểu đồ so sánh lưu tại outputs/model_comparison.png")


def plot_lasso_coefs(models: dict, feature_cols: list, X_train, y_train) -> None:
    """Trực quan hóa hệ số LASSO để thấy feature selection."""
    lasso_pipe = models["LASSO"]
    lasso_pipe.fit(X_train, y_train)
    coefs = lasso_pipe.named_steps["model"].coef_
    coef_df = pd.DataFrame({"feature": feature_cols, "coef": coefs})
    coef_df = coef_df[coef_df["coef"] != 0].sort_values("coef", key=abs, ascending=False)

    fig, ax = plt.subplots(figsize=(8, max(4, len(coef_df) * 0.4)))
    colors = ["#1D9E75" if c > 0 else "#D85A30" for c in coef_df["coef"]]
    ax.barh(coef_df["feature"], coef_df["coef"], color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title("LASSO – Hệ số thuộc tính (không bằng 0)")
    ax.set_xlabel("Hệ số")
    plt.tight_layout()
    plt.savefig("outputs/lasso_coefs.png", dpi=120, bbox_inches="tight")
    plt.close()
    print(f"✓ LASSO giữ lại {len(coef_df)}/{len(feature_cols)} thuộc tính")
    print("  " + "  ".join(coef_df["feature"].tolist()))


def select_best_model(test_df: pd.DataFrame) -> str:
    best = test_df.loc[test_df["MAE"].idxmin()]
    print(f"\n🏆 Mô hình tốt nhất (MAE thấp nhất): {best['Model']}")
    print(f"   MAE  = {best['MAE']:>12,.0f} VNĐ")
    print(f"   RMSE = {best['RMSE']:>12,.0f} VNĐ")
    print(f"   R²   = {best['R2']:.4f}")
    return best["Model"]


# ════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════

def train_and_evaluate(processed_csv: str = "data/processed_wines.csv"):
    X, y, feature_cols = load_processed(processed_csv)

    print("\n=== PHẦN 4: CHIA DỮ LIỆU ===")
    X_train_full = np.concatenate
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    X_train_full = np.vstack([X_train, X_val])
    y_train_full = np.concatenate([y_train, y_val])

    print("\n=== PHẦN 4: TINH CHỈNH SIÊU THAM SỐ (VALIDATION) ===")
    best_ridge_lam = tune_ridge(X_train, y_train, X_val, y_val)
    best_lasso_lam = tune_lasso(X_train, y_train, X_val, y_val)
    best_k         = tune_knn(X_train, y_train, X_val, y_val)

    models = build_models(best_ridge_lam, best_lasso_lam, best_k)

    cv_df   = evaluate_with_cv(models, X_train_full, y_train_full, k_fold=5)
    test_df = evaluate_on_test(models, X_train_full, y_train_full, X_test, y_test)

    plot_results(cv_df, test_df, models, X_train_full, y_train_full, X_test, y_test)
    plot_lasso_coefs(models, feature_cols, X_train_full, y_train_full)

    best_name = select_best_model(test_df)

    # Lưu bảng kết quả
    summary = test_df.copy()
    summary["MAE_M"] = (summary["MAE"] / 1e6).round(3)
    summary["RMSE_M"] = (summary["RMSE"] / 1e6).round(3)
    summary[["Model", "MAE_M", "RMSE_M", "R2"]].to_csv(
        "outputs/model_summary.csv", index=False
    )
    print("\n✓ Bảng kết quả lưu tại outputs/model_summary.csv")
    return best_name, models, test_df


if __name__ == "__main__":
    train_and_evaluate()
