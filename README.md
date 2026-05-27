# Project GDBT Housing (R)

## Struktur
- `data/housing.csv` : dataset
- `src/gbdt_housing_all_in_one.R` : satu file untuk install paket (opsional), training, evaluasi, dan prediksi
- `models/` : output model `.rds`
- `reports/` : output metrik dan file prediksi

## 1) Masuk ke folder project
```bash
cd "/Users/joicejunansitandirerung/Documents/KULIAH/SEMESTER 2/3. STA2582 TEKNIK ANALITIKA DATA BESAR/project_gbdt_housing"
```

## 2) Jalankan satu file script
```bash
Rscript src/gbdt_housing_all_in_one.R \
  --mode=train \
  --auto_install=true \
  --data=data/housing.csv \
  --target=median_house_value \
  --model_out=models/gbdt_housing_model.rds \
  --metrics_out=reports/metrics_gbdt_r.csv \
  --pred_test_out=reports/predictions_test_gbdt_r.csv
```

Output utama:
- `models/gbdt_housing_model.rds`
- `reports/metrics_gbdt_r.csv`
- `reports/predictions_test_gbdt_r.csv`

## 3) Prediksi data baru (mode inferensi)
```bash
Rscript src/gbdt_housing_all_in_one.R \
  --mode=predict \
  --model=models/gbdt_housing_model.rds \
  --predict_input=data/housing.csv \
  --predict_out=reports/predictions_new_data_gbdt_r.csv
```

## 4) Train + predict sekaligus
```bash
Rscript src/gbdt_housing_all_in_one.R \
  --mode=both \
  --auto_install=true \
  --data=data/housing.csv \
  --target=median_house_value \
  --predict_input=data/housing.csv \
  --model_out=models/gbdt_housing_model.rds \
  --metrics_out=reports/metrics_gbdt_r.csv \
  --pred_test_out=reports/predictions_test_gbdt_r.csv \
  --predict_out=reports/predictions_new_data_gbdt_r.csv
```
