"""
main.py – Chạy toàn bộ pipeline Wine Price Prediction
Usage:
    python main.py          # dùng dữ liệu mẫu (sample)
    python main.py crawl    # crawl thật từ winecellar.vn
"""
import sys, os, time

os.chdir(os.path.dirname(os.path.abspath(__file__)))

print("=" * 60)
print("  WINE PRICE PREDICTION – TOÀN BỘ PIPELINE")
print("=" * 60)

# ── PHẦN 1: Thu thập dữ liệu ─────────────────────────────────────────────
print("\n[PHẦN 1] Thu thập dữ liệu...")
from part1_crawl import crawl_all, generate_sample_data

mode = sys.argv[1] if len(sys.argv) > 1 else "sample"
if mode == "crawl":
    df_raw = crawl_all(max_pages_per_cat=3, max_products=300)
else:
    df_raw = generate_sample_data(n=200)

print(f"  → {len(df_raw)} sản phẩm đã thu thập\n")

# ── PHẦN 2 & 3: Tiền xử lý + EDA ─────────────────────────────────────────
print("[PHẦN 2 & 3] Tiền xử lý & EDA...")
from part2_3_preprocess_eda import preprocess_and_eda

df_final, encoders, scaler = preprocess_and_eda("data/raw_wines.csv")
print(f"  → Tập dữ liệu sau xử lý: {df_final.shape}\n")

# ── PHẦN 4 & 5: Mô hình + Đánh giá ───────────────────────────────────────
print("[PHẦN 4 & 5] Huấn luyện & đánh giá mô hình...")
from part4_5_models import train_and_evaluate

best_model, models, results_df = train_and_evaluate("data/processed_wines.csv")

# ── Tổng kết ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  KẾT QUẢ CUỐI CÙNG")
print("=" * 60)
print(results_df[["Model", "MAE", "RMSE", "R2"]].to_string(index=False))
print(f"\n🏆 Mô hình được chọn: {best_model}")
print("\nCác file đầu ra:")
for f in sorted(os.listdir("outputs")):
    print(f"  outputs/{f}")
for f in sorted(os.listdir("models")):
    print(f"  models/{f}")
