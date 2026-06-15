"""
Wine Price Prediction API — FastAPI Backend
===========================================
• POST /predict   → nhận form data, tiền xử lý, trả về giá dự đoán
• GET  /health    → kiểm tra server + model sẵn sàng
• GET  /meta      → trả về danh sách giống nho, quốc gia… để FE dùng
"""

import re
import datetime
import numpy as np
import pandas as pd
import joblib
import pickle
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional

# ─── Khởi tạo app ────────────────────────────────────────────────────────────

app = FastAPI(
    title="Vitis Analytics API",
    description="Wine price prediction using KNN model trained on winecellar.vn data",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # Đổi thành domain FE cụ thể khi production
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Load model ──────────────────────────────────────────────────────────────

MODEL_PATH = Path(__file__).parent / "model.pkl"
model_pipeline = None

def load_model():
    global model_pipeline
    if not MODEL_PATH.exists():
        print(f"[WARN] {MODEL_PATH} không tồn tại — API /predict sẽ trả lỗi 503.")
        return
    try:
        # Hỗ trợ cả joblib lẫn pickle
        try:
            model_pipeline = joblib.load(MODEL_PATH)
        except Exception:
            with open(MODEL_PATH, "rb") as f:
                model_pipeline = pickle.load(f)
        print(f"[OK] Đã load model từ {MODEL_PATH}")
    except Exception as e:
        print(f"[ERROR] Không load được model: {e}")

load_model()

# ─── Schema ──────────────────────────────────────────────────────────────────

class WineInput(BaseModel):
    ten:           Optional[str]   = Field(None,   description="Tên rượu (để trích xuất năm sản xuất)")
    giong_nho:     Optional[str]   = Field("Blend", description="Giống nho")
    nha_san_xuat:  Optional[str]   = Field(None,   description="Nhà sản xuất")
    quoc_gia:      Optional[str]   = Field("France", description="Quốc gia")
    nong_do:       Optional[float] = Field(13.5,   description="Độ cồn (%)")
    dung_tich:     Optional[str]   = Field("750ml", description="Dung tích (e.g. '750ml', '1500ml')")
    loai_ruou:     Optional[str]   = Field("Rượu Vang Đỏ", description="Loại rượu (danh mục web)")

class PredictResponse(BaseModel):
    gia_du_doan:   float   = Field(..., description="Giá dự đoán (VNĐ)")
    gia_thap:      float   = Field(..., description="Giá thấp ước tính (VNĐ)")
    gia_cao:       float   = Field(..., description="Giá cao ước tính (VNĐ)")
    do_tin_cay:    float   = Field(..., description="Điểm tin cậy giả định (0-100)")
    features_used: dict    = Field(..., description="Các đặc trưng đã dùng để predict")

# ─── Hàm tiền xử lý (mirror hoàn toàn notebook Preprocessing) ────────────────

APPELLATIONS = [
    r"Châteauneuf-Du-Pape", r"Côtes Du Rhône", r"Bourgogne", r"Mercurey",
    r"Bordeaux", r"Champagne", r"Alsace", r"Provence", r"Languedoc",
    r"Primitivo", r"Negroamaro", r"Barolo", r"Chianti", r"Amarone",
    r"Brunello", r"Prosecco", r"Rioja", r"Ribera Del Duero", r"Cava",
    r"Pinot Noir", r"Cabernet Sauvignon", r"Sauvignon Blanc",
    r"Chardonnay", r"Merlot", r"Syrah", r"Grenache",
]
APPELLATION_PATTERN = re.compile(
    "|".join(APPELLATIONS), flags=re.IGNORECASE
)

LOAI_RUOU_MAP = {
    "Rượu Vang Đỏ":        "red",
    "Rượu Vang Trắng":     "white",
    "Rượu Vang Sủi":       "sparkling",
    "Champagne":            "champagne",
    "Rượu Vang Hồng":      "rose",
    "Rượu Vang Ngọt":      "sweet",
    "Rượu Vang Cường Hóa": "fortified",
    "Rượu Vang Không Cồn": "non_alcohol",
    "Rượu Vang Organic":   "organic",
    # FE values
    "red": "red", "white": "white", "sparkling": "sparkling",
    "rose": "rose", "sweet": "sweet", "fortified": "fortified",
}

def extract_year(name: Optional[str]) -> Optional[int]:
    if not name:
        return None
    match = re.search(r'\b(19|20)\d{2}\b', str(name))
    return int(match.group()) if match else None

def extract_rank(name: Optional[str]) -> str:
    if not name:
        return "Unknown"
    m = APPELLATION_PATTERN.search(str(name))
    return m.group() if m else "Unknown"

def parse_dung_tich(val: Optional[str]) -> float:
    """Chuyển chuỗi dung tích về ml — giống hàm trong notebook."""
    if not val:
        return 750.0
    v = str(val).split(",")[0].strip().lower()
    if "ml" in v:
        try:
            return float(v.replace("ml", "").strip())
        except ValueError:
            pass
    elif "l" in v:
        try:
            return float(v.replace("l", "").strip()) * 1000
        except ValueError:
            pass
    # fallback: lấy số đầu tiên trong chuỗi
    m = re.search(r"[\d.]+", v)
    if m:
        num = float(m.group())
        return num * 1000 if num < 10 else num
    return 750.0


def build_feature_row(inp: WineInput) -> pd.DataFrame:
    """
    Chuyển WineInput → DataFrame 1 dòng với đúng các cột mà model.pkl
    đã được train — mirror pipeline Preprocessing + Model_and_Evaluate.
    """
    current_year = datetime.datetime.now().year

    # ── Năm sản xuất → tuổi ──
    nam = extract_year(inp.ten)
    year_is_null = 1 if nam is not None else 0
    tuoi = (current_year - nam) if nam else np.nan
    tuoi = tuoi if tuoi and tuoi > 0 else np.nan

    # ── Nong độ ──
    nong_do_num = inp.nong_do if inp.nong_do is not None else 13.5

    # ── Dung tích ──
    dung_tich_ml = parse_dung_tich(inp.dung_tich)

    # ── Hạng rượu (hang_ruou) ──
    hang_ruou = extract_rank(inp.ten)

    # ── Loại rượu (dạng chuỗi gốc tiếng Việt hoặc en) ──
    loai_ruou = inp.loai_ruou or "Rượu Vang Đỏ"

    # ── Quốc gia ──
    quoc_gia_raw = inp.quoc_gia or "France"

    row = {
        # Numeric (được StandardScaler xử lý trong pipeline)
        "nong_do_num":  nong_do_num,
        "dung_tich_ml": dung_tich_ml,
        "tuoi":         tuoi if not np.isnan(tuoi) else 0.0,
        "Year_is_Null": year_is_null,

        # Categorical — TargetEncoder (được xử lý trong pipeline)
        "giong_nho":    inp.giong_nho or "Blend",
        "nha_san_xuat": inp.nha_san_xuat or "Unknown",

        # OHE columns đã được tạo sẵn ở Preprocessing
        # Pipeline sẽ passthrough; ta cần truyền giá trị gốc nếu pipeline
        # có OneHotEncoder, hoặc truyền dummy nếu đã OHE từ trước.
        # → Ta truyền dạng raw string; nếu model expect OHE thì dùng
        #   get_dummies bên dưới.
        "hang_ruou":    hang_ruou,
        "loai_ruou":    loai_ruou,
        "quoc_gia":     quoc_gia_raw,
    }

    return pd.DataFrame([row])


def build_ohe_row(inp: WineInput) -> pd.DataFrame:
    """
    Fallback: nếu model được train trên dataframe đã OHE (dạng processed_wine_data.csv),
    ta phải tái tạo đúng schema OHE thay vì dùng pipeline encoder.
    Hàm này tạo row với tất cả cột OHE đặt 0, rồi bật cột phù hợp.
    """
    current_year = datetime.datetime.now().year
    nam = extract_year(inp.ten)
    tuoi = float(current_year - nam) if nam else 0.0
    nong_do_num = inp.nong_do if inp.nong_do is not None else 13.5
    dung_tich_ml = parse_dung_tich(inp.dung_tich)
    hang_ruou = extract_rank(inp.ten)
    loai_ruou = inp.loai_ruou or "Rượu Vang Đỏ"
    quoc_gia = inp.quoc_gia or "France"
    year_is_null = 1 if nam is not None else 0

    row = {
        "nong_do_num":  nong_do_num,
        "dung_tich_ml": dung_tich_ml,
        "tuoi":         tuoi,
        "Year_is_Null": year_is_null,
        "giong_nho":    inp.giong_nho or "Blend",
        "nha_san_xuat": inp.nha_san_xuat or "Unknown",
    }

    # OHE columns từ notebook: hang_ruou, loai_ruou, quoc_gia (drop_first=True)
    for cat, val in [("hang_ruou", hang_ruou), ("loai_ruou", loai_ruou), ("quoc_gia", quoc_gia)]:
        col = f"{cat}_{val}"
        row[col] = 1

    return pd.DataFrame([row]).fillna(0)


# ─── Endpoints ───────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model_pipeline is not None,
        "model_path": str(MODEL_PATH),
    }


@app.get("/meta")
def meta():
    """Trả về các giá trị dropdown để FE dùng."""
    return {
        "giong_nho": [
            "Blend", "Bordeaux Blend", "Pinot Noir", "Chardonnay",
            "Nebbiolo", "Sangiovese", "Cabernet Sauvignon", "Merlot",
            "Syrah", "Grenache", "Sauvignon Blanc", "Riesling",
            "Tempranillo", "Zinfandel", "Viognier",
        ],
        "quoc_gia": [
            "France", "Italy", "Spain", "USA", "Australia",
            "Chile", "Argentina", "Germany", "Portugal",
            "South Africa", "New Zealand",
        ],
        "loai_ruou": [
            "Rượu Vang Đỏ", "Rượu Vang Trắng", "Rượu Vang Sủi",
            "Champagne", "Rượu Vang Hồng", "Rượu Vang Ngọt",
            "Rượu Vang Cường Hóa", "Rượu Vang Không Cồn", "Rượu Vang Organic",
        ],
        "dung_tich": ["375ml", "500ml", "750ml", "1500ml", "3000ml"],
    }


@app.post("/predict", response_model=PredictResponse)
def predict(inp: WineInput):
    if model_pipeline is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Model chưa được load. "
                "Hãy đặt file model.pkl vào cùng thư mục với app.py và khởi động lại server."
            ),
        )

    try:
        # Thử build feature row phù hợp với pipeline
        X = build_feature_row(inp)

        # Nếu model pipeline có preprocessor (ColumnTransformer) → dùng trực tiếp
        log_pred = model_pipeline.predict(X)[0]

    except Exception as e_pipe:
        # Fallback: thử với OHE row (model train trên processed CSV)
        try:
            X = build_ohe_row(inp)
            # Căn chỉnh cột với model nếu có feature_names_in_
            if hasattr(model_pipeline, "feature_names_in_"):
                X = X.reindex(columns=model_pipeline.feature_names_in_, fill_value=0)
            log_pred = model_pipeline.predict(X)[0]
        except Exception as e_ohe:
            raise HTTPException(
                status_code=422,
                detail=f"Lỗi khi dự đoán: pipeline={e_pipe} | ohe={e_ohe}",
            )

    # ── Chuyển ngược log-transform → VNĐ ──
    gia_pred = float(np.expm1(log_pred))

    # ── Khoảng tin cậy ~±1 RMSE (KNN RMSE ≈ 2.51 triệu) ──
    RMSE_EST = 2_538_079.0
    gia_thap = max(0.0, gia_pred - RMSE_EST)
    gia_cao  = gia_pred + RMSE_EST

    # ── Điểm tin cậy (giả định dựa trên R² của KNN = 0.60) ──
    do_tin_cay = round(59.35, 2)   # placeholder; có thể tính động nếu có ensemble

    return PredictResponse(
        gia_du_doan=round(gia_pred),
        gia_thap=round(gia_thap),
        gia_cao=round(gia_cao),
        do_tin_cay=do_tin_cay,
        features_used={
            "giong_nho":    inp.giong_nho,
            "nha_san_xuat": inp.nha_san_xuat,
            "quoc_gia":     inp.quoc_gia,
            "nong_do":      inp.nong_do,
            "dung_tich":    inp.dung_tich,
            "loai_ruou":    inp.loai_ruou,
            "nam_trich":    extract_year(inp.ten),
            "hang_ruou":    extract_rank(inp.ten),
        },
    )


# ─── Chạy trực tiếp ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)