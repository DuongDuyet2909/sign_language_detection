import os                                           # Nạp công cụ làm việc với thư mục và đường dẫn.
import cv2                                          # Nạp OpenCV để đọc ảnh và chuyển hệ màu.
import mediapipe as mp                              # Nạp MediaPipe, viết tắt mp, để nhận diện các điểm bàn tay.
import matplotlib.pyplot as plt                     # Nạp công cụ vẽ ảnh; plt hiện không được sử dụng trong file này.
import pickle                                       # Nạp pickle để lưu cấu trúc dữ liệu Python vào file nhị phân.

mp_hands = mp.solutions.hands                       # Lấy mô-đun nhận diện bàn tay của MediaPipe Solutions.
mp_drawing = mp.solutions.drawing_utils             # Lấy công cụ vẽ điểm; biến này hiện không được dùng.
mp_drawing_styles = mp.solutions.drawing_styles     # Lấy kiểu vẽ mặc định; biến này hiện không được dùng.

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.5)  # Nhận diện riêng từng ảnh với ngưỡng 0.5; mặc định có thể tìm tối đa 2 bàn tay.

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")   # Xác định thư mục ảnh data nằm cạnh file Python này.

data = []                                           # Danh sách lưu các mẫu tọa độ dùng để huấn luyện.
labels = []                                         # Danh sách nhãn tương ứng với từng mẫu trong data.
for dir_ in os.listdir(DATA_DIR):                   # Duyệt tên mục trong data; code giả định mỗi mục là thư mục lớp.
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):  # Duyệt tất cả tên file trong thư mục lớp hiện tại.
        data_aux = []                               # Tạo danh sách tọa độ mới cho mỗi ảnh.
        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))  # Đọc ảnh theo đường dẫn đầy đủ; nếu thất bại thì img là None.
        if img is None:                             # Kiểm tra ảnh không đọc được.
            print("Không đọc được ảnh:", img_path)  # In tên file không đọc được để kiểm tra lại.
            continue                                # Bỏ qua ảnh lỗi và chuyển sang ảnh tiếp theo.
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)           # Chuyển BGR của OpenCV sang RGB mà MediaPipe yêu cầu.

        results = hands.process(img_rgb)            # Nhận diện bàn tay và trả về các điểm landmark trong ảnh.

        if results.multi_hand_landmarks:            # Chỉ lấy tọa độ khi nhận diện được ít nhất một bàn tay.
            for hand_landmarks in results.multi_hand_landmarks:  # Duyệt từng bàn tay; nhiều bàn tay sẽ được gộp vào cùng một mẫu trong code hiện tại.
                for i in range(len(hand_landmarks.landmark)):    # Duyệt các điểm của bàn tay, thông thường có 21 điểm.
                    x = hand_landmarks.landmark[i].x             # Lấy tọa độ x chuẩn hóa theo chiều rộng ảnh, thường trong khoảng 0 đến 1.
                    y = hand_landmarks.landmark[i].y             # Lấy tọa độ y chuẩn hóa theo chiều cao ảnh, thường trong khoảng 0 đến 1.
                    data_aux.append((x, y))                      # Thêm cặp (x,y); 21 điểm tạo mẫu (21,2), chưa trải phẳng thành 42 số.

            data.append(data_aux)                                # thêm mẫu của ảnh vào dataset; ảnh không thấy bàn tay sẽ không được thêm.
            labels.append(dir_)                                  # Thêm tên thư mục lớp làm nhãn chuỗi, như '0', '1', '2'.

save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data.pkl")    # Chọn đường dẫn data.pkl nằm cạnh file Python này.
with open(save_path, "wb") as f:                                                    # Mở file để ghi nhị phân; ghi đè file cũ; with tự đóng file.
    pickle.dump({"data": data, "labels": labels}, f)                                # Lưu từ điển chứa danh sách mẫu và nhãn bằng pickle.

print("đã lưu dữ liệu vào file data.pkl", save_path)                                # In đường dẫn nơi vừa lưu dataset.
print("Số mẫu:", len(data))                                                         # In tổng số mẫu nhận diện được, có thể ít hơn số ảnh đầu vào.
print("Số nhãn:", len(labels))                                                      # In tổng số nhãn; mỗi mẫu cần đúng một nhãn.

if not data:                                                                        # Kiểm tra dataset không có mẫu; kiểm tra này diễn ra sau khi file đã được ghi.
    raise ValueError("Dataset rỗng")                                                # Báo lỗi nếu dataset rỗng, dù data.pkl đã được tạo trước đó.

if len(data) != len(labels):                                                        # Kiểm tra số mẫu và số nhãn không bằng nhau.
    raise ValueError("Số mẫu và số nhãn không khớp")                                # Dừng và báo lỗi dữ liệu không khớp; file trước đó không tự bị xóa.