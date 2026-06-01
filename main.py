"""
main.py - Run the full Wine Price Prediction pipeline.

Usage:
    python main.py          # use deterministic sample data
    python main.py sample   # same as default
    python main.py crawl    # crawl live data from winecellar.vn
"""

from __future__ import annotations

import os
import sys


def configure_output_encoding() -> None:
    """Make Vietnamese console output work on Windows terminals."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


def main() -> None:
    configure_output_encoding()
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    mode = sys.argv[1].lower() if len(sys.argv) > 1 else "sample"
    if mode not in {"sample", "crawl"}:
        raise SystemExit("Usage: python main.py [sample|crawl]")

    print("=" * 60)
    print("  WINE PRICE PREDICTION - TOÀN BỘ PIPELINE")
    print("=" * 60)

    print("\n[PHẦN 1] Thu thập dữ liệu...")
    from part1_crawl import crawl_all, generate_sample_data

    if mode == "crawl":
        df_raw = crawl_all(max_pages_per_cat=3, max_products=300)
    else:
        df_raw = generate_sample_data(n=200)

    print(f"  -> {len(df_raw)} sản phẩm đã thu thập\n")

    print("[PHẦN 2 & 3] Tiền xử lý & EDA...")
    from part2_3_preprocess_eda import preprocess_and_eda

    df_final, encoders, scaler = preprocess_and_eda("data/raw_wines.csv")
    print(f"  -> Tập dữ liệu sau xử lý: {df_final.shape}\n")

    print("[PHẦN 4 & 5] Huấn luyện & đánh giá mô hình...")
    from part4_5_models import train_and_evaluate

    best_model, models, results_df = train_and_evaluate("data/processed_wines.csv")

    print("\n" + "=" * 60)
    print("  KẾT QUẢ CUỐI CÙNG")
    print("=" * 60)
    print(results_df[["Model", "MAE", "RMSE", "R2"]].to_string(index=False))
    print(f"\nMô hình được chọn: {best_model}")
    print("\nCác file đầu ra:")
    for f in sorted(os.listdir("outputs")):
        print(f"  outputs/{f}")
    for f in sorted(os.listdir("models")):
        print(f"  models/{f}")


if __name__ == "__main__":
    main()
