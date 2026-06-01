"""
PHẦN 1: THU THẬP DỮ LIỆU (SELENIUM VERSION)
Crawl dữ liệu rượu vang từ winecellar.vn

Các sửa đổi so với phiên bản cũ:
- parse_product_detail: dùng đúng selector của winecellar.vn
    • Giá      : .product-page-price .woocommerce-Price-amount (tránh lấy giá SP liên quan)
    • Thuộc tính: .product-attribute__item.pa_<slug> .pa-info__value
    • Rating   : .kksr-legend (text dạng "4.2/5 - (16 votes)")
    • wine_type: đọc từ pa_loai-vang thay vì breadcrumb
    • Image     : og:image meta tag
- get_product_links_from_category: thêm selector wcl-product-col
- clean_alcohol: giữ nguyên % (không chia 100) vì site ghi "13% ABV"
"""

from __future__ import annotations

import re
import time
import random
import os
import sys
import unicodedata

import pandas as pd


def configure_output_encoding() -> None:
    """Make Vietnamese console output work on Windows terminals."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


configure_output_encoding()

try:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.common.exceptions import (
        TimeoutException,
        NoSuchElementException,
        WebDriverException,
    )
    SELENIUM_AVAILABLE = True
    SELENIUM_IMPORT_ERROR: ImportError | None = None
except ImportError as exc:
    webdriver = None
    Options = None
    By = None
    WebDriverWait = None
    EC = None
    SELENIUM_AVAILABLE = False
    SELENIUM_IMPORT_ERROR = exc

    class TimeoutException(Exception):
        pass

    class NoSuchElementException(Exception):
        pass

    class WebDriverException(Exception):
        pass

# ─── cấu hình ────────────────────────────────────────────────────────────────

BASE_URL = "https://winecellar.vn"

CATEGORIES = [
    "/ruou-vang-do",
    "/ruou-vang-trang",
    "/ruou-vang-hong",
    "/ruou-vang-ngot",
    "/ruou-vang-sui",
    "/ruou-champagne",
    "/ruou-vang-organic",
    "/ruou-vang-cuong-hoa",
    "/ruou-vang-khong-con",
]

PAGE_LOAD_TIMEOUT = 20
IMPLICIT_WAIT     = 5


# ─── khởi tạo driver ─────────────────────────────────────────────────────────

def ensure_selenium_available() -> None:
    if not SELENIUM_AVAILABLE:
        raise RuntimeError(
            "Selenium is required only for crawl mode. Install it with: "
            "python -m pip install selenium"
        ) from SELENIUM_IMPORT_ERROR


def build_driver(headless: bool = True) -> webdriver.Chrome:
    ensure_selenium_available()

    opts = Options()
    if headless:
        opts.add_argument("--headless=new")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--disable-blink-features=AutomationControlled")
    opts.add_experimental_option("excludeSwitches", ["enable-automation"])
    opts.add_experimental_option("useAutomationExtension", False)
    opts.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
    opts.add_argument("--lang=vi-VN")
    opts.add_argument("--window-size=1920,1080")

    driver = webdriver.Chrome(options=opts)
    driver.execute_cdp_cmd(
        "Page.addScriptToEvaluateOnNewDocument",
        {"source": "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"},
    )
    driver.set_page_load_timeout(PAGE_LOAD_TIMEOUT)
    driver.implicitly_wait(IMPLICIT_WAIT)
    return driver


# ─── helpers ─────────────────────────────────────────────────────────────────

class PageResult:
    OK        = "ok"
    NOT_FOUND = "404"
    ERROR     = "error"


def safe_get(driver: webdriver.Chrome, url: str, retries: int = 3) -> str:
    for attempt in range(retries):
        try:
            driver.get(url)
            WebDriverWait(driver, PAGE_LOAD_TIMEOUT).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            page_title = unicodedata.normalize("NFC", driver.title.lower())
            body_text  = unicodedata.normalize("NFC",
                         driver.find_element(By.TAG_NAME, "body").text.lower())

            is_404 = (
                "404"                          in page_title
                or "not found"                 in page_title
                or "không tìm thấy"            in page_title
                or "nội dung không được tìm thấy" in page_title
                or "page not found"            in body_text
                or "error 404"                 in body_text
                or "không tìm thấy trang"      in body_text
                or "nội dung không được tìm thấy" in body_text
            )
            if is_404:
                print(f"  [404] Trang không tồn tại: {url}")
                return PageResult.NOT_FOUND
            return PageResult.OK

        except TimeoutException:
            print(f"  [!] Timeout lần {attempt + 1}: {url}")
        except WebDriverException as e:
            print(f"  [!] WebDriver lỗi lần {attempt + 1}: {e}")
        time.sleep(random.uniform(2.0, 4.0))

    return PageResult.ERROR


def scroll_to_bottom(driver: webdriver.Chrome, pause: float = 1.2) -> None:
    last_height = driver.execute_script("return document.body.scrollHeight")
    while True:
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(pause)
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            break
        last_height = new_height


def get_text(driver: webdriver.Chrome, css: str) -> str | None:
    try:
        el = driver.find_element(By.CSS_SELECTOR, css)
        return el.text.strip() or None
    except NoSuchElementException:
        return None


def get_attr(driver: webdriver.Chrome, css: str, attr: str) -> str | None:
    try:
        el = driver.find_element(By.CSS_SELECTOR, css)
        return el.get_attribute(attr)
    except NoSuchElementException:
        return None


# ─── clean helpers ───────────────────────────────────────────────────────────

def clean_price(raw: str | None) -> float | None:
    if not raw:
        return None
    digits = re.sub(r"[^\d]", "", raw)
    return float(digits) if digits else None


def clean_alcohol(raw: str | None) -> float | None:
    """Trả về số % (vd 13.0), KHÔNG chia 100."""
    if not raw:
        return None
    m = re.search(r"(\d+[\.,]?\d*)", raw)
    if not m:
        return None
    val = float(m.group(1).replace(",", "."))
    # Nếu giá trị quá nhỏ (< 1) thì có thể đã bị chia 100 ở đâu đó → nhân lại
    return val if val >= 1 else val * 100


def clean_volume(raw: str | None) -> float | None:
    if not raw:
        return None
    raw_lower = raw.lower()
    m = re.search(r"(\d+[\.,]?\d*)", raw_lower)
    if not m:
        return None
    val = float(m.group(1).replace(",", "."))
    if "l" in raw_lower and "ml" not in raw_lower:
        val *= 1000
    return val


# ─── lấy thuộc tính theo pa_slug (cấu trúc riêng winecellar.vn) ──────────────

def get_pa_value(driver: webdriver.Chrome, pa_slug: str) -> str | None:
    """
    winecellar.vn hiển thị thuộc tính trong:
      <div class="product-attribute__item pa_<slug>">
        <div class="pa-info">
          <div class="pa-info__value"><p>VALUE</p></div>
        </div>
      </div>
    """
    css = f".product-attribute__item.{pa_slug} .pa-info__value"
    try:
        el = driver.find_element(By.CSS_SELECTOR, css)
        return el.text.strip() or None
    except NoSuchElementException:
        return None


# ─── lấy danh sách link sản phẩm ─────────────────────────────────────────────

def build_page_url(cat_path: str, page: int) -> str:
    base = f"{BASE_URL}{cat_path.rstrip('/')}/"
    if page == 1:
        return base
    return f"{base}page/{page}/"


def get_product_links_from_category(
    driver: webdriver.Chrome, cat_path: str, max_pages: int = 10
) -> list[str]:
    links: list[str] = []

    print(f"\n{'─'*60}")
    print(f"  [CATEGORY] Bắt đầu: {cat_path}")
    print(f"{'─'*60}")

    for page in range(1, max_pages + 1):
        target_url = build_page_url(cat_path, page)
        print(f"  → Đang tải trang {page}: {target_url}")

        status = safe_get(driver, target_url)

        # ── Log debug giống code cũ ──────────────────────────────────────────
        print(f"  [DEBUG] URL={target_url} | status={status} | title={driver.title!r}")

        if status == PageResult.NOT_FOUND:
            print(f"  [404] Trang {page} không tồn tại"
                  f" → kết thúc danh mục [{cat_path}], chuyển sang danh mục tiếp theo.")
            break

        if status == PageResult.ERROR:
            print(f"  [!] Không tải được trang {page} sau nhiều lần thử"
                  f" → dừng danh mục [{cat_path}], chuyển sang danh mục tiếp theo.")
            break

        actual_url   = driver.current_url.rstrip("/")
        expected_url = target_url.rstrip("/")
        if page > 1 and actual_url != expected_url:
            print(f"  [i] Redirect phát hiện (trang {page}) → dừng phân trang, chuyển category.")
            print(f"      Mong đợi : {expected_url}")
            print(f"      Thực tế  : {actual_url}")
            break

        scroll_to_bottom(driver)

        # ── Selector lấy link sản phẩm (ưu tiên từ trên xuống) ──────────────
        LINK_SELECTORS = [
            # Theme winecellar.vn custom
            ".wcl-product-col a.woocommerce-LoopProduct-link",
            ".wcl-product-col a.woocommerce-loop-product__link",
            # WooCommerce chuẩn
            "a.woocommerce-LoopProduct-link",
            "a.woocommerce-loop-product__link",
            "p.product-title a",
            "h2.woocommerce-loop-product__title a",
            ".product-name a",
        ]

        found: list[str] = []
        for sel in LINK_SELECTORS:
            elements = driver.find_elements(By.CSS_SELECTOR, sel)
            if elements:
                for el in elements:
                    href = el.get_attribute("href") or ""
                    if href:
                        found.append(href if href.startswith("http") else BASE_URL + href)
                break

        # Fallback: quét toàn bộ <a>
        if not found:
            all_anchors = driver.find_elements(By.TAG_NAME, "a")
            for a in all_anchors:
                href = a.get_attribute("href") or ""
                if (
                    re.search(r"/(ruou-vang-phap|ruou-vang-y|ruou-vang|ruou|vang)/", href)
                    and "?" not in href
                    and href != target_url.rstrip("/") + "/"
                ):
                    found.append(href)

        found = list(dict.fromkeys(found))

        if not found:
            print(f"  Không tìm thấy sản phẩm ở trang {page}"
                  f" → dừng phân trang [{cat_path}], chuyển sang danh mục tiếp theo.")
            break

        links.extend(found)
        print(f"  [{cat_path}] trang {page}: +{len(found)} sản phẩm (tổng tạm: {len(links)})")
        time.sleep(random.uniform(1.0, 2.0))

    unique_links = list(dict.fromkeys(links))
    print(f"  [CATEGORY DONE] {cat_path}: {len(unique_links)} link duy nhất")
    return unique_links


# ─── lấy chi tiết sản phẩm ───────────────────────────────────────────────────

def parse_product_detail(driver: webdriver.Chrome, url: str) -> dict | None:
    status = safe_get(driver, url)
    if status != PageResult.OK:
        return None

    scroll_to_bottom(driver, pause=0.8)

    record: dict = {"url": url}

    # ── Tên sản phẩm ─────────────────────────────────────────────────────────
    try:
        record["product_name"] = driver.find_element(
            By.CSS_SELECTOR, "h1.product_title, h1.product-title, h1"
        ).text.strip() or None
    except NoSuchElementException:
        record["product_name"] = None

    # ── SKU ──────────────────────────────────────────────────────────────────
    record["sku"] = get_text(driver, ".sku")

    # ── Giá (chỉ lấy giá chính, không lấy giá sản phẩm liên quan) ───────────
    # Selector cụ thể theo cấu trúc trang: .product-page-price hoặc .product-price-container
    price_raw = None
    for sel in [
        ".product-page-price .woocommerce-Price-amount",
        ".product-price-container .woocommerce-Price-amount",
        ".hide-for-small .woocommerce-Price-amount",   # desktop price block
        ".summary .woocommerce-Price-amount",
        ".price > .woocommerce-Price-amount",
    ]:
        price_raw = get_text(driver, sel)
        if price_raw:
            break
    record["price_vnd"] = clean_price(price_raw)

    # ── Thuộc tính từ .product-attribute__item.pa_<slug> ─────────────────────
    # Slug tương ứng với class CSS trên trang winecellar.vn
    record["grape_variety"]  = get_pa_value(driver, "pa_giong-nho")
    record["wine_type"]      = get_pa_value(driver, "pa_loai-vang")
    record["brand"]          = get_pa_value(driver, "pa_nha-san-xuat")
    record["origin_country"] = get_pa_value(driver, "pa_quoc-gia")
    record["alcohol_content"]= clean_alcohol(get_pa_value(driver, "pa_nong-do"))
    record["volume"]         = clean_volume(get_pa_value(driver, "pa_dung-tich"))

    # ── Vùng sản xuất (region) — đọc từ breadcrumb ──────────────────────────
    # Vd: Trang chủ / Rượu Vang Pháp / Rượu Vang Bourgogne / <tên SP>
    region = None
    try:
        crumb_els = driver.find_elements(
            By.CSS_SELECTOR,
            ".woocommerce-breadcrumb a, .breadcrumbs a"
        )
        if len(crumb_els) >= 3:
            # Phần tử áp cuối (trước tên SP) thường là vùng/danh mục chi tiết
            region = crumb_els[-1].text.strip() or None
    except Exception:
        pass
    record["region"] = region

    # ── Vintage ──────────────────────────────────────────────────────────────
    vintage_raw = None
    name = record.get("product_name") or ""
    m = re.search(r"\b(19[5-9]\d|20[0-2]\d)\b", name)
    vintage_raw = m.group(1) if m else None
    record["vintage"] = int(vintage_raw) if vintage_raw else None

    # ── Rating ───────────────────────────────────────────────────────────────
    # winecellar.vn dùng plugin kk-star-ratings, hiển thị text dạng "4.2/5 - (16 votes)"
    rating_score = None
    rating_count = None
    rating_raw = get_text(driver, ".kksr-legend")
    if rating_raw:
        m_score = re.search(r"(\d+[\.,]\d+|\d+)\s*/\s*5", rating_raw)
        m_count = re.search(r"\((\d+)\s*(votes?|đánh giá|bình chọn)", rating_raw, re.IGNORECASE)
        if m_score:
            rating_score = float(m_score.group(1).replace(",", "."))
        if m_count:
            rating_count = int(m_count.group(1))
    else:
        # Fallback: đọc từ JSON-LD schema nếu có
        try:
            schema_el = driver.find_element(
                By.CSS_SELECTOR,
                'script[type="application/ld+json"]'
            )
            import json
            data = json.loads(schema_el.get_attribute("innerHTML") or "{}")
            if isinstance(data, dict):
                ar = data.get("aggregateRating", {})
                if ar:
                    try:
                        rating_score = float(ar.get("ratingValue", 0)) or None
                        rating_count = int(ar.get("ratingCount", 0)) or None
                    except (ValueError, TypeError):
                        pass
        except (NoSuchElementException, Exception):
            pass

    record["rating_score"] = rating_score
    record["rating_count"] = rating_count

    # ── Ảnh sản phẩm (og:image) ──────────────────────────────────────────────
    record["image_url"] = get_attr(driver, 'meta[property="og:image"]', "content")

    # ── Mô tả ngắn ───────────────────────────────────────────────────────────
    short_desc = get_text(driver, ".product-short-description")
    record["short_description"] = short_desc

    return record


def generate_sample_data(
    n: int = 200,
    seed: int = 42,
    output_path: str = "data/raw_wines.csv",
) -> pd.DataFrame:
    """Generate a deterministic sample dataset with the same schema as crawled data."""
    rng = random.Random(seed)

    wine_types = [
        "Rượu Vang Đỏ",
        "Rượu Vang Trắng",
        "Rượu Vang Hồng",
        "Rượu Vang Ngọt",
        "Rượu Vang Sủi",
        "Champagne",
        "Rượu Vang Organic",
        "Rượu Vang Không Cồn",
    ]
    grape_varieties = [
        "Cabernet Sauvignon",
        "Merlot",
        "Pinot Noir",
        "Chardonnay",
        "Sauvignon Blanc",
        "Syrah",
        "Malbec",
        "Riesling",
    ]
    countries = ["Pháp", "Ý", "Chile", "Mỹ", "Úc", "Tây Ban Nha", "Argentina"]
    regions_by_country = {
        "Pháp": ["Bordeaux", "Bourgogne", "Champagne", "Rhône"],
        "Ý": ["Tuscany", "Piedmont", "Veneto"],
        "Chile": ["Maipo Valley", "Colchagua Valley"],
        "Mỹ": ["Napa Valley", "Sonoma"],
        "Úc": ["Barossa Valley", "Margaret River"],
        "Tây Ban Nha": ["Rioja", "Ribera Del Duero"],
        "Argentina": ["Mendoza", "Salta"],
    }
    brands = [
        "Chateau Demo",
        "Domaine Sample",
        "Casa Vista",
        "Reserve Cellar",
        "Estate Select",
        "Grand Valley",
    ]
    type_price_factor = {
        "Rượu Vang Đỏ": 1.05,
        "Rượu Vang Trắng": 0.9,
        "Rượu Vang Hồng": 0.85,
        "Rượu Vang Ngọt": 0.95,
        "Rượu Vang Sủi": 1.15,
        "Champagne": 1.8,
        "Rượu Vang Organic": 1.25,
        "Rượu Vang Không Cồn": 0.7,
    }
    country_price_factor = {
        "Pháp": 1.45,
        "Ý": 1.25,
        "Chile": 0.85,
        "Mỹ": 1.2,
        "Úc": 1.0,
        "Tây Ban Nha": 0.95,
        "Argentina": 0.8,
    }

    records = []
    for idx in range(1, n + 1):
        wine_type = rng.choice(wine_types)
        country = rng.choice(countries)
        region = rng.choice(regions_by_country[country])
        grape = rng.choice(grape_varieties)
        brand = rng.choice(brands)
        vintage = rng.choice(list(range(2008, 2023)) + [None])
        rating_score = round(rng.uniform(3.4, 4.9), 1)
        rating_count = rng.randint(3, 160)
        volume = rng.choices([375.0, 750.0, 1500.0], weights=[8, 85, 7], k=1)[0]

        if wine_type == "Rượu Vang Không Cồn":
            alcohol_content = round(rng.uniform(0.0, 0.5), 1)
        else:
            alcohol_content = round(rng.uniform(11.0, 15.5), 1)

        vintage_factor = 1.0
        if vintage:
            vintage_factor += max(0, 2024 - vintage) * 0.018

        price = (
            420_000
            * type_price_factor[wine_type]
            * country_price_factor[country]
            * (volume / 750.0) ** 0.85
            * (0.85 + rating_score / 8)
            * vintage_factor
            * rng.uniform(0.75, 1.35)
        )
        price_vnd = int(round(max(price, 120_000), -3))
        product_name = f"{brand} {grape} {vintage or 'NV'}"

        records.append({
            "url": f"{BASE_URL}/sample-wine-{idx:04d}",
            "product_name": product_name,
            "sku": f"SAMPLE-{idx:04d}",
            "price_vnd": price_vnd,
            "grape_variety": grape,
            "wine_type": wine_type,
            "brand": brand,
            "origin_country": country,
            "alcohol_content": alcohol_content,
            "volume": volume,
            "region": region,
            "vintage": vintage,
            "rating_score": rating_score,
            "rating_count": rating_count,
            "image_url": "",
            "short_description": f"Dữ liệu mẫu cho {wine_type.lower()} từ {country}.",
        })

    df = pd.DataFrame(records)
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"✓ Đã tạo {len(df)} bản ghi mẫu tại {output_path}")
    return df


# ─── main ─────────────────────────────────────────────────────────────────────

def crawl_all(
    max_pages_per_cat: int = 5,
    max_products: int = 300,
    headless: bool = True,
) -> pd.DataFrame:
    driver = build_driver(headless=headless)

    try:
        all_links: list[str] = []

        print("=== THU THẬP LINK SẢN PHẨM ===")
        for idx, cat in enumerate(CATEGORIES, 1):
            print(f"\n[{idx}/{len(CATEGORIES)}] Đang xử lý category: {cat}")
            links = get_product_links_from_category(driver, cat, max_pages=max_pages_per_cat)
            all_links.extend(links)
            print(f"  ✓ Xong [{cat}]: {len(links)} link | Tổng tích lũy: {len(all_links)}")

        all_links = list(dict.fromkeys(all_links))[:max_products]
        print(f"\nTổng: {len(all_links)} sản phẩm (sau dedup)\n")

        records = []
        print("=== CÀO CHI TIẾT SẢN PHẨM ===")
        for i, url in enumerate(all_links, 1):
            print(f"  [{i}/{len(all_links)}] {url}")
            rec = parse_product_detail(driver, url)
            if rec:
                records.append(rec)
            time.sleep(random.uniform(1.0, 2.5))

    finally:
        driver.quit()

    df = pd.DataFrame(records)
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/raw_wines.csv", index=False, encoding="utf-8-sig")
    print(f"\n✓ Đã lưu {len(df)} bản ghi vào data/raw_wines.csv")
    return df


if __name__ == "__main__":
    df = crawl_all(max_pages_per_cat=50, max_products=20000, headless=True)
    print(df.head())
    print(df.dtypes)
