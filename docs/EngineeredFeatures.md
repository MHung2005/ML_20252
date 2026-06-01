# EngineeredFeatures v2 Explanation

Tài liệu này giải thích biến thể `EngineeredFeatures` hiện tại trong `notebooks/wine_price_eda_training.ipynb`, các feature đã thêm hoặc loại bỏ, và vì sao kết quả hiện tại tốt hơn `BaselineOriginal`.

## 1. Bối cảnh so sánh

Dữ liệu sau lọc target có `200` dòng. Notebook dùng cùng một cách chia dữ liệu cho cả hai pipeline:

- Train: `160` dòng
- Holdout test: `40` dòng
- Split: stratified theo 5 quantile bins của `price_vnd`
- CV: 5-fold shuffled KFold trên train split
- Target: `price_vnd`
- Best model của cả hai pipeline: `ElasticNetLogTarget`

`BaselineOriginal` dùng các feature gốc có thể dùng được:

- Numeric: `alcohol_content`, `volume`, `vintage`, `rating_score`, `rating_count`
- Categorical: `grape_variety`, `wine_type`, `brand`, `origin_country`, `region`
- Text: `combined_text = product_name + short_description`
- Loại khỏi baseline: `url`, `sku`, `image_url`

`url` và `sku` bị loại vì gần như là định danh từng dòng, dễ làm model học thuộc thay vì học quy luật giá. `image_url` bị loại vì missing 100%.

## 2. Feature đã thêm trong EngineeredFeatures v2

### 2.1 Missingness flags

```python
is_missing_vintage = vintage.isna()
is_missing_alcohol = alcohol_content.isna()
```

Ý nghĩa:

- Nếu một trường quan trọng bị thiếu, bản thân việc thiếu dữ liệu có thể là tín hiệu.
- Ví dụ sản phẩm không có vintage hoặc nồng độ cồn có thể thuộc nhóm thông tin chưa đầy đủ, khác với sản phẩm cao cấp có metadata đầy đủ.

### 2.2 Volume features

```python
log_volume = log1p(volume)
```

Ý nghĩa:

- `volume` là feature numeric có tương quan mạnh với giá.
- `log_volume` giúp model tuyến tính học quan hệ mềm hơn giữa dung tích và giá, thay vì coi tăng từ 375ml lên 750ml giống hệt tăng từ 750ml lên 1500ml theo scale thô.

Engineered v2 vẫn giữ `volume` gốc và thêm `log_volume`.

### 2.3 Vintage / age features

```python
wine_age = REFERENCE_YEAR - vintage
age_group = cut(wine_age, ["0-2", "3-5", "6-10", "11-20", "20+"])
```

Ý nghĩa:

- `wine_age` trực quan hơn `vintage`: model nhìn trực tiếp độ tuổi chai rượu.
- `age_group` giúp bắt quan hệ phi tuyến, ví dụ rượu quá mới, trung bình, hoặc lâu năm có hành vi giá khác nhau.

Trong v2, `vintage` gốc không nằm trong feature numeric chính của engineered set; nó được thay bằng `wine_age`, `age_group`, và `is_missing_vintage`.

### 2.4 Rating and popularity features

```python
log_rating_count = log1p(rating_count)
bayesian_rating = weighted average of rating_score and global mean
rating_x_popularity = rating_score * log1p(rating_count)
rating_bin = cut(rating_score, ["low", "mid", "good", "very_good", "excellent"])
```

Ý nghĩa:

- `log_rating_count`: giảm ảnh hưởng của các sản phẩm có số lượng rating quá lớn.
- `bayesian_rating`: rating 4.8 với ít lượt đánh giá không nên được tin ngang rating 4.8 với nhiều lượt đánh giá.
- `rating_x_popularity`: kết hợp chất lượng cảm nhận và độ phổ biến.
- `rating_bin`: giúp model học các ngưỡng rating dễ hiểu hơn, ví dụ từ `good` sang `very_good`.

Trong v2, `rating_count` gốc được thay bằng `log_rating_count` và các biến rating tổng hợp.

### 2.5 Alcohol features

```python
alcohol_diff_from_median = alcohol_content - median(alcohol_content)
alcohol_bin = cut(alcohol_pct, ["very_low", "low", "medium", "high", "very_high"])
```

Ý nghĩa:

- `alcohol_diff_from_median` cho biết nồng độ cồn lệch khỏi mức phổ biến bao nhiêu.
- `alcohol_bin` giúp model học các nhóm nồng độ cồn thay vì chỉ học quan hệ tuyến tính.

`alcohol_pct` được tạo như biến trung gian để chia bin, không được dùng trực tiếp làm predictor cuối cùng.

### 2.6 Grape group feature

```python
grape_color_group = red_grape / white_grape / other
```

Ý nghĩa:

- Feature này gom `grape_variety` thành nhóm rộng hơn.
- Cardinality thấp, ổn định hơn các interaction nhiều level.
- Giữ thêm tín hiệu về phong cách rượu mà không làm model quá phức tạp.

### 2.7 Light text features

```python
text_has_year = product_name contains a 4-digit year
description_length = len(short_description)
```

Ý nghĩa:

- `text_has_year`: tên sản phẩm có năm có thể bổ sung tín hiệu vintage khi `vintage` thiếu hoặc không sạch.
- `description_length`: độ dài mô tả là proxy nhẹ cho mức độ đầy đủ thông tin sản phẩm.

Engineered v2 vẫn giữ `combined_text` như baseline, đồng thời thêm hai feature text nhẹ này.

## 3. Feature đã loại bỏ so với EngineeredFeatures v1

Engineered v1 thêm quá nhiều feature phức tạp cho dataset chỉ khoảng 200 dòng. V2 đã loại các nhóm sau.

### 3.1 High-cardinality interactions

Đã loại:

- `country_x_type`
- `type_x_volume`
- `brand_x_type`
- `country_x_grape`
- `type_x_alcohol_bin`

Lý do:

- Các feature này tạo quá nhiều category hiếm.
- Ví dụ `country_x_type` và `country_x_grape` có hơn 50 level trong khi dataset chỉ có 200 dòng.
- Nhiều level có 1-2 mẫu, làm model dễ học nhiễu của train split thay vì quy luật tổng quát.

### 3.2 Group target-stat encodings

Đã loại các thống kê theo nhóm như:

- group mean/median/std/count của `log_price`
- target-stat theo interaction features

Lý do:

- Dù đã làm leakage-safe trong pipeline, các nhóm nhỏ vẫn tạo thống kê không ổn định.
- Với 200 dòng, target-stat trên nhóm hiếm dễ làm model overfit.

### 3.3 Manual price tiers

Đã loại:

- `wine_type_price_tier`
- `origin_price_tier`
- các tier thủ công tương tự

Lý do:

- One-hot của `wine_type` và `origin_country` đã chứa thông tin này.
- Tier thủ công có thể áp đặt giả định mạnh, không chắc đúng khi dataset nhỏ.

### 3.4 Duplicate binary category flags

Đã loại:

- `is_champagne`
- `is_french`
- `is_premium_origin`
- `is_low_price_origin`
- `is_old_wine`
- `is_very_old_wine`
- `is_high_rated`
- `is_excellent_rated`
- `is_low_alcohol`
- `is_high_alcohol`

Lý do:

- Nhiều flag trùng thông tin với one-hot category hoặc bin feature.
- Giữ quá nhiều biến trùng nhau có thể làm ElasticNet phải xử lý nhiễu không cần thiết.

## 4. Kết quả hiện tại

Theo `docs/evaluation.md`, best model của cả hai pipeline là `ElasticNetLogTarget`.

| Metric | BaselineOriginal | EngineeredFeatures v2 | Chênh lệch |
| --- | ---: | ---: | ---: |
| CV_MAE_mean | 139,777 | 123,322 | giảm 16,455 |
| CV_RMSE_mean | 188,700 | 159,141 | giảm 29,559 |
| CV_R2_mean | 0.6877 | 0.7595 | tăng 0.0718 |
| Holdout_MAE | 129,107 | 116,207 | giảm 12,900 |
| Holdout_RMSE | 173,991 | 162,241 | giảm 11,750 |
| Holdout_R2 | 0.7953 | 0.8220 | tăng 0.0267 |
| Holdout_MAPE | 0.1633 | 0.1556 | giảm 0.0077 |

Kết luận trực tiếp:

- `EngineeredFeatures v2` tốt hơn `BaselineOriginal`.
- Holdout MAE giảm `12,900` VND, tương đương cải thiện khoảng `9.99%`.
- Cải thiện xuất hiện ở cả CV metrics và holdout metrics, nên đáng tin hơn so với chỉ cải thiện trên một holdout split.

## 5. Ý nghĩa từng metric evaluation

### 5.1 CV_MAE_mean

`CV_MAE_mean` là trung bình MAE qua 5 fold cross-validation trên train split.

MAE là Mean Absolute Error:

```text
MAE = trung bình |giá thật - giá dự đoán|
```

Ý nghĩa:

- Cho biết model sai trung bình bao nhiêu VND.
- Càng thấp càng tốt.
- Dễ diễn giải vì cùng đơn vị với target: VND.

Trong kết quả hiện tại:

- Baseline: `139,777` VND
- Engineered v2: `123,322` VND
- Engineered v2 tốt hơn `16,455` VND trên cross-validation.

Điều này cho thấy feature mới không chỉ may mắn trên holdout, mà còn giúp model ổn định hơn trên train folds.

### 5.2 CV_RMSE_mean

`CV_RMSE_mean` là trung bình RMSE qua 5 fold cross-validation.

RMSE là Root Mean Squared Error:

```text
RMSE = sqrt(trung bình (giá thật - giá dự đoán)^2)
```

Ý nghĩa:

- Càng thấp càng tốt.
- RMSE phạt lỗi lớn mạnh hơn MAE vì sai số bị bình phương.
- Nếu RMSE giảm, model thường ít tạo ra dự đoán lệch nặng hơn.

Trong kết quả hiện tại:

- Baseline: `188,700` VND
- Engineered v2: `159,141` VND
- Engineered v2 giảm `29,559` VND.

Điều này cho thấy biến thể engineered không chỉ giảm lỗi trung bình, mà còn giảm các lỗi lớn trong CV.

### 5.3 CV_R2_mean

`CV_R2_mean` là trung bình R2 qua 5 fold cross-validation.

R2 đo tỷ lệ biến thiên của target được model giải thích:

```text
R2 = 1 - residual_variance / target_variance
```

Ý nghĩa:

- Càng cao càng tốt.
- `1.0` là dự đoán hoàn hảo.
- `0.0` tương đương dự đoán không tốt hơn trung bình.
- Âm nghĩa là tệ hơn baseline rất đơn giản.

Trong kết quả hiện tại:

- Baseline: `0.6877`
- Engineered v2: `0.7595`
- Engineered v2 tăng `0.0718`.

Điều này nghĩa là feature engineered giúp model giải thích biến thiên giá tốt hơn trong cross-validation.

### 5.4 Holdout_MAE

`Holdout_MAE` là MAE trên tập test giữ riêng 40 dòng.

Ý nghĩa:

- Đây là metric dễ hiểu nhất để chọn model dự đoán giá.
- Cho biết khi gặp dữ liệu chưa dùng để train, model sai trung bình bao nhiêu VND.
- Càng thấp càng tốt.

Trong kết quả hiện tại:

- Baseline: `129,107` VND
- Engineered v2: `116,207` VND
- Engineered v2 giảm `12,900` VND.

Đây là cải thiện chính: lỗi dự đoán trung bình trên holdout giảm khoảng `9.99%`.

### 5.5 Holdout_RMSE

`Holdout_RMSE` là RMSE trên holdout test.

Ý nghĩa:

- Càng thấp càng tốt.
- Nhạy với lỗi lớn.
- Nếu MAE giảm nhưng RMSE tăng, có thể model tốt hơn ở lỗi thường nhưng tệ hơn ở vài case lệch mạnh. Ở đây cả hai đều giảm.

Trong kết quả hiện tại:

- Baseline: `173,991` VND
- Engineered v2: `162,241` VND
- Engineered v2 giảm `11,750` VND.

Điều này cho thấy feature v2 cũng giảm mức độ lỗi lớn trên holdout.

### 5.6 Holdout_R2

`Holdout_R2` là R2 trên holdout test.

Ý nghĩa:

- Càng cao càng tốt.
- Cho biết model giải thích biến thiên giá trên dữ liệu chưa thấy tốt đến đâu.

Trong kết quả hiện tại:

- Baseline: `0.7953`
- Engineered v2: `0.8220`
- Engineered v2 tăng `0.0267`.

R2 tăng nghĩa là model engineered nắm được cấu trúc giá tốt hơn, không chỉ giảm lỗi tuyệt đối.

### 5.7 Holdout_MAPE

`Holdout_MAPE` là Mean Absolute Percentage Error:

```text
MAPE = trung bình |giá thật - giá dự đoán| / giá thật
```

Ý nghĩa:

- Càng thấp càng tốt.
- Diễn giải như phần trăm sai số tương đối.
- Hữu ích khi muốn biết model sai bao nhiêu phần trăm so với giá thật.
- Với target giá luôn dương như `price_vnd`, MAPE dùng được.

Trong kết quả hiện tại:

- Baseline: `0.1633`, tức khoảng `16.33%`
- Engineered v2: `0.1556`, tức khoảng `15.56%`
- Engineered v2 giảm khoảng `0.77` điểm phần trăm.

Điều này nghĩa là xét theo tỷ lệ so với giá thật, engineered v2 cũng tốt hơn.

### 5.8 Delta_MAE_vs_BaselineBest

Metric này so sánh MAE của pipeline với best baseline.

Ý nghĩa:

- Giá trị âm: pipeline tốt hơn baseline.
- Giá trị dương: pipeline tệ hơn baseline.
- Đơn vị là VND.

Trong kết quả hiện tại:

```text
Delta_MAE_vs_BaselineBest = -12,900
```

Nghĩa là EngineeredFeatures v2 giảm lỗi trung bình `12,900` VND so với best baseline.

### 5.9 Delta_MAE_pct_vs_BaselineBest

Metric này là phần trăm thay đổi MAE so với best baseline.

Ý nghĩa:

- Giá trị âm: cải thiện.
- Giá trị dương: tệ hơn.
- Giúp đánh giá mức cải thiện tương đối thay vì chỉ nhìn số VND.

Trong kết quả hiện tại:

```text
Delta_MAE_pct_vs_BaselineBest = -9.9916%
```

Nghĩa là EngineeredFeatures v2 cải thiện gần `10%` theo MAE.

## 6. Vì sao EngineeredFeatures v2 tốt hơn baseline

EngineeredFeatures v2 tốt hơn vì nó thêm tín hiệu có ích nhưng không làm feature space quá nhiễu:

- `wine_age`, `age_group`: biểu diễn vintage theo cách dễ học hơn.
- `log_volume`: giữ tín hiệu dung tích nhưng giảm scale thô.
- `log_rating_count`, `bayesian_rating`, `rating_x_popularity`: tận dụng rating và độ phổ biến tốt hơn raw rating.
- `alcohol_diff_from_median`, `alcohol_bin`: giúp model học nồng độ cồn theo cả dạng liên tục và nhóm.
- `is_missing_vintage`, `is_missing_alcohol`: giữ tín hiệu từ missingness.
- `grape_color_group`, `rating_bin`, `age_group`, `alcohol_bin`: categorical ít level, ít rủi ro overfit.
- `text_has_year`, `description_length`: thêm tín hiệu text nhẹ, không làm mô hình quá phức tạp.

Đồng thời v2 loại những feature gây overfit trong v1:

- interaction nhiều level,
- target-stat theo nhóm hiếm,
- tier thủ công,
- binary flag trùng thông tin.

Kết quả là model `ElasticNetLogTarget` vẫn giữ được ưu điểm ổn định của linear model có regularization, nhưng có thêm các biến biểu diễn tốt hơn cho quan hệ giá.

## 7. Lưu ý khi diễn giải

Dataset chỉ có khoảng `200` dòng, nên kết quả holdout có thể dao động nếu đổi random split. Tuy nhiên, EngineeredFeatures v2 cải thiện đồng thời:

- CV_MAE,
- CV_RMSE,
- CV_R2,
- Holdout_MAE,
- Holdout_RMSE,
- Holdout_R2,
- Holdout_MAPE.

Vì vậy, đây là một cải thiện hợp lý hơn biến thể EngineeredFeatures v1, vốn thêm quá nhiều feature sparse và chỉ cải thiện một số model nhất định.
