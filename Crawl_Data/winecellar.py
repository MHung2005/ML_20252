from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException, TimeoutException
import pandas as pd
import time

# ─── Cấu hình ────────────────────────────────────────────────────────────────

CATEGORIES = [
    ("Rượu Vang Đỏ",        "https://winecellar.vn/ruou-vang-do/"),
    ("Rượu Vang Trắng",     "https://winecellar.vn/ruou-vang-trang/"),
    ("Rượu Vang Sủi",       "https://winecellar.vn/ruou-vang-sui/"),
    ("Champagne",           "https://winecellar.vn/ruou-champagne/"),
    ("Rượu Vang Hồng",      "https://winecellar.vn/ruou-vang-hong/"),
    ("Rượu Vang Ngọt",      "https://winecellar.vn/ruou-vang-ngot/"),
    ("Rượu Vang Cường Hóa", "https://winecellar.vn/ruou-vang-cuong-hoa/"),
    ("Rượu Vang Không Cồn", "https://winecellar.vn/ruou-vang-khong-con/"),
    ("Rượu Vang Organic",   "https://winecellar.vn/ruou-vang-organic/"),
]

DELAY  = 1.5
OUTPUT = "wines1.csv"

# ─── Khởi động trình duyệt ───────────────────────────────────────────────────

options = webdriver.ChromeOptions()
options.add_argument("--headless")
options.add_argument("--no-sandbox")
options.add_argument("--disable-dev-shm-usage")
driver = webdriver.Chrome(options=options)
wait   = WebDriverWait(driver, 10)

# ─── Hàm tiện ích ────────────────────────────────────────────────────────────

def get_text(el, selector):
    """Lấy text, bỏ tên cột thừa: 'GIỐNG NHO\nBlend' → 'Blend'"""
    try:
        text = el.find_element(By.CSS_SELECTOR, selector).text.strip()
        return text.split("\n")[-1].strip()
    except NoSuchElementException:
        return None

def get_price(driver):
    """
    Lấy giá sản phẩm — thử nhiều selector theo thứ tự ưu tiên.
    Lý do dùng hàm riêng: giá load sau h1 (lazy), cần wait + fallback.
    """
    PRICE_SELECTORS = [
        "p.product-page-price bdi",       # selector ổn định nhất (trang chi tiết)
        ".price-wrapper .price bdi",       # selector gốc trong code
        "p.price bdi",                     # fallback chung
        ".woocommerce-Price-amount bdi",   # fallback WooCommerce chuẩn
    ]
    # Đợi ít nhất 1 selector xuất hiện (tối đa 8s)
    for sel in PRICE_SELECTORS:
        try:
            el = WebDriverWait(driver, 8).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, sel))
            )
            # Dùng JS để lấy text thuần, tránh ký tự &nbsp; bị bỏ sót
            raw = driver.execute_script("return arguments[0].innerText;", el)
            return raw.strip() if raw else None
        except TimeoutException:
            continue
        except Exception:
            continue
    return None

def has_next_page():
    try:
        driver.find_element(By.CSS_SELECTOR, "link[rel='next']")
        return True
    except NoSuchElementException:
        return False

# ─── Bước 1: Thu thập link từ tất cả danh mục ───────────────────────────────

print("=" * 60)
print("BƯỚC 1: Thu thập link sản phẩm")
print("=" * 60)

product_links = {}  # {url: loai_danh_muc}

for loai, base_url in CATEGORIES:
    page = 1
    print(f"\n>> Danh mục: {loai}")

    while True:
        url = f"{base_url}page/{page}/" if page > 1 else base_url
        try:
            driver.get(url)
            wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, ".product-small")))
        except TimeoutException:
            break

        links = driver.find_elements(By.CSS_SELECTOR, ".product-small a.woocommerce-loop-product__link")
        for a in links:
            href = a.get_attribute("href")
            if href and href not in product_links:
                product_links[href] = loai

        print(f"   Trang {page} — {len(links)} link | Tổng duy nhất: {len(product_links)}")

        if not has_next_page():
            break
        page += 1
        time.sleep(DELAY)

print(f"\n>> Tổng link duy nhất: {len(product_links)}")

# ─── Bước 2: Crawl chi tiết từng sản phẩm ───────────────────────────────────

print("\n" + "=" * 60)
print("BƯỚC 2: Crawl chi tiết sản phẩm")
print("=" * 60)

data  = []
total = len(product_links)

for i, (url, loai_danh_muc) in enumerate(product_links.items(), 1):
    try:
        driver.get(url)
        wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "h1.product_title")))

        data.append({
            "ten"          : get_text(driver, "h1.product_title"),
            "gia"          : get_price(driver),  
            "giong_nho"    : get_text(driver, ".product-attribute__item.pa_giong-nho"),
            "nha_san_xuat" : get_text(driver, ".product-attribute__item.pa_nha-san-xuat"),
            "quoc_gia"     : get_text(driver, ".product-attribute__item.pa_quoc-gia"),
            "nong_do"      : get_text(driver, ".product-attribute__item.pa_nong-do"),
            "dung_tich"    : get_text(driver, ".product-attribute__item.pa_dung-tich"),
            "loai_ruou"    : loai_danh_muc,
            "url"          : url,
        })

        print(f"  [{i}/{total}] {data[-1]['ten']} — {data[-1]['gia']}")

    except Exception as e:
        print(f"  [LỖI {i}/{total}] {url}: {e}")

    time.sleep(DELAY)

# ─── Bước 3: Lưu CSV ─────────────────────────────────────────────────────────

driver.quit()

df = pd.DataFrame(data)
df.to_csv(OUTPUT, index=False, encoding="utf-8-sig")

co_gia = df["gia"].notna().sum()
print(f"\n>> Hoàn thành! {co_gia}/{len(df)} sản phẩm có giá → '{OUTPUT}'")