"""
Wine Price Prediction API — FastAPI Backend
===========================================
- POST /predict   → nhận form data, tiền xử lý, trả về giá dự đoán
- GET  /health    → kiểm tra server + model sẵn sàng
- GET  /meta      → trả về danh sách giống nho, quốc gia… để FE dùng
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
from typing import Optional, Literal

# ─── Khởi tạo app ────────────────────────────────────────────────────────────

app = FastAPI(
    title="Vitis Analytics API",
    description="Wine price prediction using KNN/Random Forest models trained on winecellar.vn data",
    version="1.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # Đổi thành domain FE cụ thể khi production
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Load models ─────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).parent

MODEL_PATHS = {
    "knn":           BASE_DIR / "knn_model.pkl",
    "random_forest": BASE_DIR / "random_forest_model.pkl",
}

# RMSE ước tính riêng cho mỗi model — chỉnh lại theo số đo thật của bạn
MODEL_RMSE = {
    "knn":           2_538_079.0,
    "random_forest": 2_566_902.0,
}

# R² ước tính riêng cho mỗi model — chỉnh lại theo số đo thật của bạn
MODEL_R2 = {
    "knn":           59.35,
    "random_forest": 58.47,
}

MODEL_DISPLAY_NAME = {
    "knn":           "KNN",
    "random_forest": "Random Forest",
}

models: dict = {"knn": None, "random_forest": None}

DEFAULT_MODEL = "knn"


def _load_one(key: str, path: Path):
    if not path.exists():
        print(f"[WARN] {path} không tồn tại — model '{key}' sẽ không khả dụng.")
        return None
    try:
        try:
            obj = joblib.load(path)
        except Exception:
            with open(path, "rb") as f:
                obj = pickle.load(f)
        print(f"[OK] Đã load model '{key}' từ {path}")
        return obj
    except Exception as e:
        print(f"[ERROR] Không load được model '{key}': {e}")
        return None


def load_models():
    for key, path in MODEL_PATHS.items():
        models[key] = _load_one(key, path)


load_models()

# ─── Schema ──────────────────────────────────────────────────────────────────

ModelName = Literal["knn", "random_forest"]

class WineInput(BaseModel):
    ten:           Optional[str]   = Field(None,   description="Tên rượu (để trích xuất năm sản xuất)")
    giong_nho:     Optional[str]   = Field("Blend", description="Giống nho")
    nha_san_xuat:  Optional[str]   = Field(None,   description="Nhà sản xuất")
    quoc_gia:      Optional[str]   = Field("France", description="Quốc gia")
    nong_do:       Optional[float] = Field(13.5,   description="Độ cồn (%)")
    dung_tich:     Optional[str]   = Field("750ml", description="Dung tích (e.g. '750ml', '1500ml')")
    loai_ruou:     Optional[str]   = Field("Rượu Vang Đỏ", description="Loại rượu (danh mục web)")
    model:         Optional[ModelName] = Field("knn", description="Model dùng để dự đoán: 'knn' hoặc 'random_forest'")

class PredictResponse(BaseModel):
    gia_du_doan:   float   = Field(..., description="Giá dự đoán (VNĐ)")
    gia_thap:      float   = Field(..., description="Giá thấp ước tính (VNĐ)")
    gia_cao:       float   = Field(..., description="Giá cao ước tính (VNĐ)")
    do_tin_cay:    float   = Field(..., description="Điểm tin cậy giả định (0-100)")
    model_used:    str     = Field(..., description="Model đã dùng để dự đoán")
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
        "models_loaded": {
            key: (m is not None) for key, m in models.items()
        },
        "model_paths": {key: str(path) for key, path in MODEL_PATHS.items()},
        "default_model": DEFAULT_MODEL,
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
        "models": [
            {"id": "knn", "label": "KNN", "available": models["knn"] is not None},
            {"id": "random_forest", "label": "Random Forest", "available": models["random_forest"] is not None},
        ],
        "default_model": DEFAULT_MODEL,
    }


@app.post("/predict", response_model=PredictResponse)
def predict(inp: WineInput):
    model_key = inp.model or DEFAULT_MODEL
    if model_key not in models:
        raise HTTPException(status_code=400, detail=f"Model không hợp lệ: '{model_key}'. Chọn 'knn' hoặc 'random_forest'.")

    model_pipeline = models.get(model_key)

    if model_pipeline is None:
        raise HTTPException(
            status_code=503,
            detail=(
                f"Model '{model_key}' chưa được load. "
                f"Hãy đặt file {MODEL_PATHS[model_key].name} vào cùng thư mục với app.py và khởi động lại server."
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
                detail=f"Lỗi khi dự đoán ({model_key}): pipeline={e_pipe} | ohe={e_ohe}",
            )

    # ── Chuyển ngược log-transform → VNĐ ──
    gia_pred = float(np.expm1(log_pred))

    # ── Khoảng tin cậy ~±1 RMSE theo từng model ──
    rmse_est = MODEL_RMSE.get(model_key, 2_538_079.0)
    gia_thap = max(0.0, gia_pred - rmse_est)
    gia_cao  = gia_pred + rmse_est

    # ── Điểm tin cậy (giả định dựa trên R² của từng model) ──
    do_tin_cay = round(MODEL_R2.get(model_key, 59.35), 2)

    return PredictResponse(
        gia_du_doan=round(gia_pred),
        gia_thap=round(gia_thap),
        gia_cao=round(gia_cao),
        do_tin_cay=do_tin_cay,
        model_used=MODEL_DISPLAY_NAME.get(model_key, model_key),
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