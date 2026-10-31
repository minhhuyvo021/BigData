🫀 Phân tích dữ liệu Heart Failure bằng PySpark

Dự án này thực hiện Phần 4 của bài tập lớn môn Khai thác dữ liệu lớn, sử dụng tập dữ liệu heart_failure_clinical_records_dataset.csv được đọc từ hệ thống tệp phân tán HDFS.

🎯 1. Mục tiêu Phần 4

Dự án áp dụng và đánh giá 3 thuật toán tiêu biểu trong quy trình Data Mining:

⚬ Logistic Regression: Phân loại biến DEATH_EVENT (dự đoán khả năng tử vong của bệnh nhân).
⚬ KMeans: Phân cụm bệnh nhân dựa trên các chỉ số lâm sàng để tìm ra các nhóm đặc trưng.
⚬ FP-Growth: Khai phá tập phổ biến và luật kết hợp từ các biến đã được rời rạc hóa.

⚙️ 2. Cài đặt môi trường

Thiết lập môi trường ảo Python và cài đặt các thư viện cần thiết:

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt


⚠️ Lưu ý quan trọng:
PySpark yêu cầu máy tính phải cài đặt sẵn Java. Kiểm tra phiên bản Java hiện tại trên máy của bạn bằng lệnh:

java -version


🚀 3. Chạy phân tích

Chạy script chính để thực hiện quá trình phân tích.

Đảm bảo bạn đã thay thế <YOUR_HDFS_PATH> bằng đường dẫn thực tế chứa dữ liệu trên hệ thống HDFS của bạn:

python src/heart_failure_pyspark.py --input <YOUR_HDFS_PATH>/heart_failure_clinical_records_dataset.csv --output outputs


📂 4. Kết quả sinh ra

Sau khi quá trình chạy hoàn tất, toàn bộ kết quả sẽ được xuất ra thư mục outputs/, bao gồm:

⚬ 📁 classification_metrics/: Các chỉ số đánh giá mô hình phân loại (Accuracy, Precision, Recall, F1, ROC AUC, PR AUC).
⚬ 📁 classification_predictions/: Tập mẫu dự đoán của thuật toán Logistic Regression.
⚬ 📁 kmeans_k_selection/: Bảng điểm Silhouette để đánh giá và lựa chọn giá trị k tối ưu.
⚬ 📁 kmeans_cluster_metrics/: Độ thuần nhất (Purity) và điểm Silhouette của mô hình KMeans tốt nhất.
⚬ 📁 kmeans_cluster_profile/: Mô tả chi tiết từng cụm (số lượng bệnh nhân, tỷ lệ tử vong, các giá trị trung bình của từng chỉ số).
⚬ 📁 fpgrowth_itemsets/: Top các tập phổ biến được khai phá.
⚬ 📁 fpgrowth_rules/: Top các luật kết hợp kèm theo các chỉ số tin cậy (Lift, All-confidence, Kulczynski và Cosine).
⚬ 📄 comparison_summary.md: File tóm tắt so sánh 3 phương pháp, phục vụ trực tiếp cho việc viết báo cáo.

💻 5. Huấn luyện mô hình và Ứng dụng dự đoán

Để xây dựng một ứng dụng thực tế từ kết quả phân tích, bạn có thể huấn luyện một mô hình Logistic Regression riêng và lưu vào thư mục models/:

python src/train_heart_failure_model.py


Nếu cần truyền rõ đường dẫn HDFS cụ thể, hãy chạy:

python src/train_heart_failure_model.py \
  --input <YOUR_HDFS_PATH>/heart_failure_clinical_records_dataset.csv \
  --model-output models/logistic_regression_pipeline


Mở giao diện người dùng (GUI):
Sau khi có mô hình, bạn có thể mở giao diện Tkinter để người dùng nhập các chỉ số sức khỏe thực tế và xem kết quả dự đoán trực tiếp:

python src/predict_heart_failure_tkinter.py
