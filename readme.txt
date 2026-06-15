Project: Dự đoán giá rượu (Nhóm 10)
=====================================

Mục đích
--------
Bộ mã thu thập dữ liệu từ winecellar.vn, tiền xử lý, huấn luyện mô hình và cung cấp API
FastAPI để dự đoán giá rượu cùng giao diện front-end.

Cấu trúc chính
----------------
- `Crawl_Data/` — scraper Selenium (`winecellar.py`) và dữ liệu mẫu `wines1.csv`.
- `Preprocessing/`, `EDA/`, `Model_and_Evaluate/` — notebook phân tích, tiền xử lý và huấn luyện.
- `Web/backend/` — FastAPI server: `app.py`, file mô hình `model.pkl`, `requirements.txt`.
- `Web/frontend/` — giao diện `predict.html` (tập tin tĩnh, dùng CDN Tailwind).

Yêu cầu (Prerequisites)
-----------------------
- Python 3.10+ (3.11 recommended).
- Pip (`pip`), venv.
- Chrome browser (nếu chạy scraper bằng Selenium).
- ChromeDriver khớp với phiên bản Chrome, hoặc dùng `webdriver-manager` để quản lý tự động.

Thiết lập môi trường (Windows)
-------------------------------
1. Tạo virtualenv và kích hoạt:

```bash
python -m venv venv
.\venv\Scripts\activate
```

2. Cài đặt dependency backend (chạy từ thư mục gốc hoặc `Web/backend`):

```bash
pip install -r Web/backend/requirements.txt
# Thêm các gói còn thiếu cần cho scraper / server
pip install selenium webdriver-manager python-multipart
```

Ghi chú: `Web/backend/requirements.txt` đã liệt kê FastAPI, Uvicorn, scikit-learn, pandas, numpy, joblib. Nếu bạn không cần chạy scraper có thể bỏ qua `selenium`.

Chuẩn bị mô hình
-----------------
- Nếu bạn đã có file mô hình `model.pkl`, đặt nó vào `Web/backend/model.pkl` (đây là nơi `app.py` tìm).
- Nếu không có, mở notebook `Model_and_Evaluate/Model_and_Evaluate.ipynb` để huấn luyện và lưu pipeline dưới tên `model.pkl` vào thư mục backend.

Chạy backend (API)
-------------------
1. Chuyển vào thư mục backend (tùy chọn):

```bash
cd Web/backend
```

2. Chạy server bằng Uvicorn:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

3. Kiểm tra API health:

```bash
curl http://127.0.0.1:8000/health
```

Gọi endpoint predict (ví dụ):

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"ten":"Château Margaux 2015","giong_nho":"Cabernet Sauvignon","nha_san_xuat":"Château Margaux","quoc_gia":"France","nong_do":13.5,"dung_tich":"750ml","loai_ruou":"Rượu Vang Đỏ"}'
```

Chạy frontend
--------------
- Frontend là file tĩnh `Web/frontend/predict.html`. Mở trực tiếp file trong trình duyệt hoặc host bằng một static file server.
- Lưu ý: `predict.html` mặc định gọi API tại `http://localhost:8000` (biến `API_BASE` trong file). Nếu backend chạy ở địa chỉ khác, sửa biến này.

Chạy scraper (thu thập dữ liệu)
--------------------------------
- Yêu cầu: Chrome + ChromeDriver hoặc `webdriver-manager`.
- Ví dụ chạy:

```bash
cd Crawl_Data
python winecellar.py
```

- Kết quả: file CSV `Crawl_Data/wines1.csv` (UTF-8 BOM) sẽ được tạo/ghi đè.

Lời khuyên Selenium (Windows)
- Đặt `chromedriver.exe` trong PATH hoặc cùng thư mục script, hoặc cài `webdriver-manager` và sửa `winecellar.py` để dùng nó.

Notebooks
---------
- Mở `Preprocessing/Preprocessing.ipynb`, `EDA/EDA.ipynb` và `Model_and_Evaluate/Model_and_Evaluate.ipynb` bằng Jupyter/Colab để xem pipeline tiền xử lý, feature engineering và huấn luyện.

Vấn đề thường gặp & khắc phục nhanh
-----------------------------------
- Backend báo `Model chưa được load` → đảm bảo `Web/backend/model.pkl` tồn tại và có thể load bằng `joblib` hoặc `pickle`.
- Lỗi Selenium `WebDriverException` → kiểm tra ChromeDriver phiên bản khớp với Chrome, hoặc dùng `webdriver-manager`.
- Lỗi import hoặc version → kiểm tra Python version và cài lại bằng `pip install -r Web/backend/requirements.txt`.
- Port 8000 đang dùng → đổi port khi chạy `uvicorn`.


