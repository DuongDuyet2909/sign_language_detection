import pickle                                           # Nạp pickle để đọc dataset và lưu mô hình.
import os                                               # Nạp os để ghép đường dẫn cạnh file Python.
import numpy as np                                      # Nạp NumPy để tạo mảng số cho huấn luyện.
from sklearn.ensemble import RandomForestClassifier     # Nạp RandomForestClassifier: bộ phân loại kết hợp nhiều cây quyết định.
from sklearn.model_selection import train_test_split    # Nạp hàm chia dữ liệu thành tập huấn luyện và tập kiểm tra.
from sklearn.metrics import accuracy_score, classification_report  #Nạp cách tính độ chính xác và báo cáo precision, recall, F1 cho từng lớp.

data_path = os.path.join(                               # Bắt đầu ghép đường dẫn đầy đủ đến dataset.
    os.path.dirname(os.path.abspath(__file__)),         # Lấy đường dẫn thư mục chứa file Python này.
    "data.pkl"                                          # Tên file dataset cần đọc trong thư mục đó.
)  

with open(data_path, "rb") as f:                        # Mở dataset để đọc nhị phân; with tự đóng file sau khi đọc.
    data_dict = pickle.load(f)                          # Đọc từ điển chứa các mẫu data và các nhãn labels.

data = np.asarray(data_dict["data"], dtype=np.float32)  # Chuyển mẫu thành mảng float32; các mẫu phải có kích thước đồng nhất.
data = data.reshape(data.shape[0], -1)                  # Giữ số mẫu, trải phẳng phần còn lại; (N,21,2) trở thành (N,42).
labels = np.asarray(data_dict["labels"])                # Đổi danh sách nhãn thành mảng, vẫn giữ nhãn chuỗi 0,1,2.

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, random_state=42, stratify=labels)  # Chia 80% học, 20% kiểm tra; xáo trộn, seed 42, giữ gần tỷ lệ lớp bằng stratify; ảnh liên tiếp dễ làm điểm số quá lạc quan.

model = RandomForestClassifier(n_estimators=100, random_state=42)  # Tạo RandomForest gồm 100 cây; seed 42 giúp tái lập kết quả trong cùng điều kiện.

model.fit(X_train, y_train)                             # Huấn luyện mô hình từ đặc trưng và nhãn của tập học.

y_pred = model.predict(X_test)                          # Dự đoán nhãn cho các mẫu trong tập kiểm tra.

score = accuracy_score(y_test, y_pred)                  # Tính tỷ lệ mẫu kiểm tra dự đoán đúng, trong khoảng 0 đến 1.
print("Accuracy:", score)                               # In accuracy dạng tỷ lệ; 0.95 tương ứng 95%.
print("Classification Report:")                         # In tiêu đề báo cáo đánh giá chi tiết.
print(classification_report(y_test, y_pred))            # In precision, recall, F1 và số mẫu kiểm tra của từng lớp.

model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.p")    # Chọn đường dẫn model.p cạnh file Python này.
with open(model_path, "wb") as f:                                                   # Mở file ghi nhị phân; ghi đè mô hình cũ cùng tên; with tự đóng file.
    pickle.dump({"model": model}, f)                                                # Lưu từ điển có khóa model để file inference đọc theo cùng cấu trúc.