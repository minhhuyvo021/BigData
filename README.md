# Phan tich Heart Failure bang PySpark

Project nay thuc hien phan 4 cua de bai mon Khai thac du lieu lon tren file
`heart_failure_clinical_records_dataset.csv` duoc doc tu HDFS.

## 1. Muc tieu phan 4

Chon 3 giai thuat trong so do Data Mining va danh gia ket qua:

- `LogisticRegression`: phan loai `DEATH_EVENT`.
- `KMeans`: phan cum benh nhan theo cac chi so lam sang.
- `FPGrowth`: khai pha tap pho bien va luat ket hop tu cac bien da roi rac hoa.

## 2. Cai dat

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

May can co Java. Kiem tra bang:

```bash
java -version
```

## 3. Chay phan tich

```bash
python src/heart_failure_pyspark.py \
  --input hdfs://localhost:9000/user/tony/input/dataset/heart_failure_clinical_records_dataset.csv \
  --output outputs
```

Neu dung dung duong dan HDFS mac dinh trong de bai, co the chay ngan gon:

```bash
python src/heart_failure_pyspark.py --output outputs
```

## 4. Ket qua sinh ra

Sau khi chay, thu muc `outputs/` se co:

- `classification_metrics/`: cac chi so Accuracy, Precision, Recall, F1, ROC AUC, PR AUC.
- `classification_predictions/`: mau du doan cua Logistic Regression.
- `kmeans_k_selection/`: silhouette score cho cac gia tri `k`.
- `kmeans_cluster_metrics/`: purity va silhouette cua mo hinh KMeans tot nhat.
- `kmeans_cluster_profile/`: mo ta tung cum, gom so luong, ty le tu vong va gia tri trung binh.
- `fpgrowth_itemsets/`: top tap pho bien.
- `fpgrowth_rules/`: top luat ket hop kem lift, all-confidence, Kulczynski va cosine.
- `comparison_summary.md`: tom tat so sanh 3 phuong phap de dua vao bao cao.

## 5. Goi y dua vao bao cao

- Voi phan loai: dung `classification_metrics` de nhan xet hieu suat mo hinh.
- Voi phan cum: dung silhouette cho danh gia khong giam sat, dung purity theo `DEATH_EVENT`
  cho danh gia co giam sat.
- Voi tap pho bien/luat ket hop: dung lift va cosine/Kulczynski de nhan xet muc do lien he
  giua cac dieu kien va ket qua.

## 6. Train model va mo form du doan

Train 1 model Logistic Regression va luu vao thu muc `models/`:

```bash
python src/train_heart_failure_model.py
```

Neu muon truyen ro duong dan HDFS:

```bash
python src/train_heart_failure_model.py \
  --input hdfs://localhost:9000/user/tony/input/dataset/heart_failure_clinical_records_dataset.csv \
  --model-output models/logistic_regression_pipeline
```

Mo form Tkinter cho nguoi dung nhap input va xem ket qua du doan:

```bash
python src/predict_heart_failure_tkinter.py
```

Ket qua tren form gom:

- Nhan du doan: nguy co thap hoac co nguy co tu vong.
- Xac suat `DEATH_EVENT = 1`.
