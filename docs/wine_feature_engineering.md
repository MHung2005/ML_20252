# Wine Price Feature Engineering

## 1. Target transformation

```python
df["log_price"] = np.log1p(df["price_vnd"])
```

Inverse prediction:

```python
df["pred_price_vnd"] = np.expm1(df["pred_log_price"])
```

---

## 2. Volume features

```python
df["volume_factor"] = df["volume"] / 750
```

```python
df["log_volume"] = np.log1p(df["volume"])
```

```python
df["is_half_bottle"] = (df["volume"] == 375).astype(int)
df["is_standard_bottle"] = (df["volume"] == 750).astype(int)
df["is_magnum"] = (df["volume"] == 1500).astype(int)
df["is_large_bottle"] = (df["volume"] >= 1500).astype(int)
```

---

## 3. Vintage / age features

```python
CURRENT_YEAR = pd.Timestamp.today().year

df["wine_age"] = CURRENT_YEAR - df["vintage"]
```

```python
df["log_wine_age"] = np.log1p(df["wine_age"])
```

```python
df["vintage_decade"] = (df["vintage"] // 10 * 10).astype(str) + "s"
```

```python
df["age_group"] = pd.cut(
    df["wine_age"],
    bins=[-1, 2, 5, 10, 20, 100],
    labels=["0-2", "3-5", "6-10", "11-20", "20+"]
)
```

```python
df["is_old_wine"] = (df["wine_age"] >= 10).astype(int)
df["is_very_old_wine"] = (df["wine_age"] >= 20).astype(int)
```

---

## 4. Rating features

```python
df["rating_bin"] = pd.cut(
    df["rating_score"],
    bins=[0, 3.5, 4.0, 4.3, 4.6, 5.0],
    labels=["low", "mid", "good", "very_good", "excellent"]
)
```

```python
df["is_high_rated"] = (df["rating_score"] >= 4.3).astype(int)
df["is_excellent_rated"] = (df["rating_score"] >= 4.6).astype(int)
```

```python
df["log_rating_count"] = np.log1p(df["rating_count"])
```

```python
df["rating_count_bin"] = pd.qcut(
    df["rating_count"],
    q=4,
    labels=[
        "low_popularity",
        "medium_popularity",
        "high_popularity",
        "very_high_popularity"
    ],
    duplicates="drop"
)
```

```python
global_rating_mean = df["rating_score"].mean()
m = df["rating_count"].median()

df["bayesian_rating"] = (
    df["rating_count"] / (df["rating_count"] + m) * df["rating_score"]
    + m / (df["rating_count"] + m) * global_rating_mean
)
```

```python
df["rating_x_popularity"] = df["rating_score"] * np.log1p(df["rating_count"])
```

```python
df["high_rating_high_count"] = (
    (df["rating_score"] >= 4.3)
    & (df["rating_count"] >= df["rating_count"].median())
).astype(int)
```

---

## 5. Alcohol features

```python
df["alcohol_bin"] = pd.cut(
    df["alcohol_content"],
    bins=[0, 10, 12, 13.5, 15, 100],
    labels=["very_low", "low", "medium", "high", "very_high"]
)
```

```python
df["alcohol_diff_from_median"] = (
    df["alcohol_content"] - df["alcohol_content"].median()
)
```

```python
df["is_low_alcohol"] = (df["alcohol_content"] <= 11).astype(int)
df["is_high_alcohol"] = (df["alcohol_content"] >= 14).astype(int)
```

---

## 6. Wine type tier features

```python
wine_type_tier = {
    "Champagne": 4,
    "Rượu Vang Sủi": 3,
    "Rượu Vang Organic": 3,
    "Rượu Vang Trắng": 2,
    "Rượu Vang Đỏ": 2,
    "Rượu Vang Hồng": 1,
    "Rượu Vang Ngọt": 1,
    "Rượu Vang Không Cồn": 0
}

df["wine_type_price_tier"] = df["wine_type"].map(wine_type_tier)
```

```python
df["is_champagne"] = (df["wine_type"] == "Champagne").astype(int)
```

---

## 7. Origin country tier features

```python
country_tier = {
    "Pháp": 3,
    "Ý": 3,
    "Mỹ": 3,
    "Úc": 2,
    "Tây Ban Nha": 1,
    "Chile": 1,
    "Argentina": 1
}

df["origin_price_tier"] = df["origin_country"].map(country_tier)
```

```python
df["is_french"] = (df["origin_country"] == "Pháp").astype(int)
```

```python
df["is_premium_origin"] = df["origin_country"].isin([
    "Pháp", "Ý", "Mỹ"
]).astype(int)
```

```python
df["is_low_price_origin"] = df["origin_country"].isin([
    "Chile", "Argentina", "Tây Ban Nha"
]).astype(int)
```

---

## 8. Grape variety features

```python
red_grapes = [
    "Merlot",
    "Syrah",
    "Malbec",
    "Cabernet Sauvignon",
    "Pinot Noir"
]

white_grapes = [
    "Chardonnay",
    "Riesling",
    "Sauvignon Blanc"
]

df["grape_color_group"] = np.select(
    [
        df["grape_variety"].isin(red_grapes),
        df["grape_variety"].isin(white_grapes)
    ],
    ["red_grape", "white_grape"],
    default="other"
)
```

```python
min_count = 10
counts = df["grape_variety"].value_counts()

df["grape_variety_grouped"] = df["grape_variety"].where(
    df["grape_variety"].map(counts) >= min_count,
    "Other"
)
```

---

## 9. Brand features

```python
brand_freq = df["brand"].value_counts(normalize=True)

df["brand_freq"] = df["brand"].map(brand_freq)
```

---

## 10. Interaction features

```python
df["country_x_type"] = (
    df["origin_country"] + "_" + df["wine_type"]
)
```

```python
df["type_x_volume"] = (
    df["wine_type"] + "_" + df["volume"].astype(str)
)
```

```python
df["brand_x_type"] = (
    df["brand"] + "_" + df["wine_type"]
)
```

```python
df["country_x_grape"] = (
    df["origin_country"] + "_" + df["grape_variety"]
)
```

```python
df["type_x_alcohol_bin"] = (
    df["wine_type"] + "_" + df["alcohol_bin"].astype(str)
)
```

```python
df["premium_type_origin"] = (
    df["wine_type"].isin([
        "Champagne",
        "Rượu Vang Sủi",
        "Rượu Vang Organic"
    ])
    & df["origin_country"].isin([
        "Pháp",
        "Ý",
        "Mỹ"
    ])
).astype(int)
```

---

## 11. Normalized price target alternative

```python
df["price_per_750ml"] = df["price_vnd"] / df["volume"] * 750
```

```python
df["log_price_per_750ml"] = np.log1p(df["price_per_750ml"])
```

---

## 12. Group statistical encoding features

```python
group_cols = [
    "wine_type",
    "origin_country",
    "grape_variety",
    "brand",
    "country_x_type",
    "brand_x_type",
    "type_x_volume"
]

for col in group_cols:
    stats = train_df.groupby(col)["log_price"].agg([
        "median",
        "mean",
        "std",
        "count"
    ])

    stats.columns = [
        f"{col}_log_price_median",
        f"{col}_log_price_mean",
        f"{col}_log_price_std",
        f"{col}_count"
    ]

    train_df = train_df.merge(stats, on=col, how="left")
    valid_df = valid_df.merge(stats, on=col, how="left")
```

---

## 13. One-hot encoding features

```python
one_hot_cols = [
    "wine_type",
    "origin_country",
    "grape_variety_grouped",
    "grape_color_group",
    "rating_bin",
    "rating_count_bin",
    "age_group",
    "alcohol_bin",
    "country_x_type",
    "type_x_volume"
]

df = pd.get_dummies(
    df,
    columns=one_hot_cols,
    drop_first=False
)
```
