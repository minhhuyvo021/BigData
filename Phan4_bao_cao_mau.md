# PHẦN 4. ĐÁNH GIÁ CÁC MẪU KẾT QUẢ THU ĐƯỢC CỦA CHƯƠNG TRÌNH

## 4.1. Lựa chọn các giải thuật để thực hiện khai thác dữ liệu

Trong phần này, đề tài lựa chọn 3 giải thuật tiêu biểu thuộc 3 hướng khai thác dữ liệu khác nhau để đánh giá trên bộ dữ liệu suy tim `heart_failure_clinical_records_dataset.csv`, gồm: `Logistic Regression`, `KMeans` và `FPGrowth`. Việc lựa chọn này nhằm bảo đảm tính đa dạng trong cách tiếp cận dữ liệu, bao gồm dự đoán nhãn đầu ra, phân nhóm đối tượng tương đồng và phát hiện các mẫu kết hợp thường xuyên xuất hiện trong dữ liệu.

Giải thuật `Logistic Regression` được sử dụng cho bài toán phân loại nhằm dự đoán biến mục tiêu `DEATH_EVENT`. Đây là phương pháp phù hợp vì bộ dữ liệu có nhãn đầu ra rõ ràng và mục tiêu của đề tài là dự đoán nguy cơ tử vong của bệnh nhân dựa trên các chỉ số lâm sàng. Tiếp theo, `KMeans` được chọn để thực hiện phân cụm bệnh nhân theo mức độ tương đồng của các thuộc tính đầu vào, qua đó hỗ trợ quan sát cấu trúc tiềm ẩn của dữ liệu. Cuối cùng, `FPGrowth` được sử dụng để khai phá tập phổ biến và luật kết hợp, giúp tìm ra những tổ hợp đặc trưng thường đi cùng nhau và có liên quan đến kết quả tử vong.

## 4.2. Đánh giá kết quả của từng giải thuật

### 4.2.1. Đánh giá mô hình Logistic Regression

Đối với phương pháp phân loại, đề tài sử dụng mô hình `Logistic Regression` và đánh giá bằng phương pháp `Holdout`, trong đó dữ liệu được chia thành tập huấn luyện và tập kiểm tra. Sau khi làm sạch, bộ dữ liệu còn 299 dòng, trong đó 238 dòng được dùng để huấn luyện và 61 dòng được dùng để kiểm tra mô hình. Các chỉ số đánh giá thu được cho thấy mô hình hoạt động khá tốt trên bộ dữ liệu này.

Cụ thể, độ chính xác (`Accuracy`) của mô hình đạt `0.8689`, chỉ số `F1-score` đạt `0.8707`, `Weighted Precision` đạt `0.8747` và `Weighted Recall` đạt `0.8689`. Đối với lớp tử vong `DEATH_EVENT = 1`, độ chính xác riêng của lớp này (`precision_death_class`) đạt `0.7500`, còn độ bao phủ (`recall_death_class`) đạt `0.8333`. Ngoài ra, `specificity` đối với lớp sống sót đạt `0.8837`. Chỉ số `ROC AUC` đạt `0.9264`, cho thấy mô hình có khả năng phân biệt khá tốt giữa hai lớp tử vong và không tử vong. Bên cạnh đó, `PR AUC` đạt `0.7842`, phản ánh mô hình vẫn giữ được mức dự đoán tốt đối với lớp dương tính.

Từ các kết quả trên có thể nhận thấy `Logistic Regression` là phương pháp phù hợp nhất trong 3 phương pháp đã chọn nếu mục tiêu chính là dự đoán nguy cơ tử vong của bệnh nhân. Mô hình cho kết quả ổn định, dễ diễn giải và có thể tích hợp thành giao diện dự đoán để người dùng nhập chỉ số lâm sàng và nhận kết quả dự báo.

### 4.2.2. Đánh giá mô hình KMeans

Đối với phương pháp phân cụm, đề tài sử dụng giải thuật `KMeans` để nhóm các bệnh nhân theo mức độ tương đồng của các chỉ số lâm sàng. Theo đúng gợi ý của đề bài, kết quả phân cụm được đánh giá bằng hai cách: phương pháp không giám sát và phương pháp có giám sát.

Ở góc độ không giám sát, đề tài sử dụng `Silhouette Score` để đánh giá chất lượng phân cụm. Kết quả thử nghiệm cho các giá trị `k` khác nhau cho thấy mô hình đạt kết quả tốt nhất tại `k = 2`, với `Silhouette Score = 0.2132`. Giá trị này chưa cao, cho thấy các cụm có mức độ tách biệt vừa phải và dữ liệu bệnh nhân không tạo thành các nhóm quá rõ ràng.

Ở góc độ có giám sát, đề tài sử dụng độ đo `Purity` dựa trên nhãn thật `DEATH_EVENT` để xem mỗi cụm có đồng nhất về kết quả tử vong hay không. Kết quả thu được `Purity = 0.6789`, nghĩa là khoảng 67.89% các phần tử trong cụm thuộc về nhãn chiếm đa số của cụm đó. Điều này cho thấy `KMeans` có khả năng tạo ra các nhóm bệnh nhân có tính chất tương đối giống nhau, tuy nhiên kết quả phân cụm vẫn chưa đủ mạnh để thay thế một mô hình phân loại chuyên biệt.

Từ kết quả trên có thể kết luận rằng `KMeans` phù hợp với mục tiêu khám phá dữ liệu và hỗ trợ phân nhóm bệnh nhân hơn là phục vụ dự đoán trực tiếp. Phương pháp này có giá trị trong việc quan sát xu hướng phân bố và hỗ trợ phân tích mô tả.

### 4.2.3. Đánh giá mô hình FPGrowth

Đối với bài toán khai phá luật kết hợp, đề tài sử dụng giải thuật `FPGrowth` sau khi đã rời rạc hóa một số thuộc tính số thành các mức như thấp, trung bình và cao. Kết quả thu được `3245` tập phổ biến và `7450` luật kết hợp. Trong số đó có `61` luật có vế phải là `death=yes`, tức liên quan trực tiếp đến khả năng tử vong của bệnh nhân.

Theo gợi ý của đề bài, đề tài sử dụng các thang đo `Lift`, `All-confidence`, `Kulczynski` và `Cosine` để đánh giá chất lượng luật. Một trong những luật nổi bật nhất là:

`time=low, serum_sodium=low -> death=yes`

Luật này có `confidence = 0.8378`, `lift = 2.6095`, `all-confidence = 0.3229`, `Kulczynski = 0.5804` và `cosine = 0.5201`. Kết quả này cho thấy khi bệnh nhân có thời gian theo dõi thấp và nồng độ natri huyết thanh thấp thì khả năng xuất hiện nhãn tử vong cao hơn đáng kể so với ngẫu nhiên. Giá trị `lift` lớn hơn 1 và khá cao chứng tỏ luật có ý nghĩa thực tế trong dữ liệu. Ngoài ra, các độ đo `Kulczynski` và `Cosine` cũng cho thấy mức liên hệ tương đối mạnh giữa các thuộc tính ở vế trái và kết quả tử vong ở vế phải.

Nhìn chung, `FPGrowth` không nhằm mục tiêu dự đoán từng cá nhân như `Logistic Regression`, nhưng lại rất hữu ích trong việc giải thích dữ liệu và phát hiện những mẫu điều kiện có khả năng liên quan đến kết quả lâm sàng. Đây là phương pháp phù hợp khi cần rút ra các quy luật mô tả từ dữ liệu.

## 4.3. So sánh và đánh giá 3 phương pháp đã thực hiện

Qua quá trình thực nghiệm, có thể thấy mỗi phương pháp đóng một vai trò khác nhau trong khai thác dữ liệu. `Logistic Regression` là phương pháp phù hợp nhất cho mục tiêu dự đoán vì đây là mô hình có giám sát, có đầu ra rõ ràng và đạt các chỉ số đánh giá cao như `Accuracy = 0.8689` và `ROC AUC = 0.9264`. Kết quả này cho thấy mô hình có khả năng ứng dụng thực tế để hỗ trợ dự đoán nguy cơ tử vong của bệnh nhân dựa trên thông tin đầu vào.

Trong khi đó, `KMeans` thích hợp hơn cho bài toán phân nhóm bệnh nhân. Phương pháp này không sử dụng nhãn đầu ra để huấn luyện nên phù hợp với mục tiêu khám phá dữ liệu và nhận diện cấu trúc tiềm ẩn. Tuy nhiên, với `Silhouette Score = 0.2132` và `Purity = 0.6789`, có thể thấy chất lượng cụm chỉ ở mức trung bình và không mạnh bằng mô hình phân loại khi xét trên mục tiêu dự đoán.

Đối với `FPGrowth`, điểm mạnh lớn nhất là khả năng phát hiện các luật kết hợp có ý nghĩa và hỗ trợ giải thích dữ liệu. Kết quả luật `time=low, serum_sodium=low -> death=yes` với `lift = 2.6095` cho thấy giữa các thuộc tính lâm sàng và kết quả tử vong tồn tại những quan hệ đáng chú ý. Tuy nhiên, phương pháp này thiên về mô tả và giải thích hơn là đưa ra một quyết định phân loại trực tiếp cho từng bệnh nhân.

Từ các kết quả trên, có thể kết luận rằng nếu mục tiêu chính của bài toán là dự đoán nguy cơ tử vong thì `Logistic Regression` là phương pháp phù hợp nhất. Nếu mục tiêu là phân nhóm bệnh nhân để hỗ trợ phân tích mô tả thì `KMeans` là lựa chọn phù hợp. Nếu mục tiêu là tìm các mẫu điều kiện và quy luật tiềm ẩn trong dữ liệu thì `FPGrowth` mang lại giá trị cao. Việc kết hợp cả 3 phương pháp trong cùng một đề tài giúp quá trình phân tích dữ liệu trở nên toàn diện hơn, vừa có khả năng dự đoán, vừa có khả năng mô tả và giải thích dữ liệu.

## Gợi ý chèn vào báo cáo

Trong phần báo cáo chính thức, nên chèn thêm các bảng và hình sau để phần 4 đầy đủ và thuyết phục hơn:

1. Bảng tổng hợp các chỉ số của `Logistic Regression`.
2. Bảng kết quả `Silhouette Score` theo từng giá trị `k` của `KMeans`.
3. Bảng `Purity` và mô tả đặc trưng của từng cụm.
4. Bảng các luật kết hợp tiêu biểu của `FPGrowth`.
5. Ảnh chụp giao diện Tkinter nhập chỉ số và hiển thị kết quả dự đoán.

Bạn có thể dùng nguyên văn nội dung này làm phần khung cho báo cáo, sau đó chỉnh lại cách hành văn cho phù hợp với phong cách cá nhân hoặc yêu cầu trình bày của giảng viên.
