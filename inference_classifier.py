import pickle  # Giải thích: Nạp pickle để đọc mô hình đã huấn luyện.
import os  # Giải thích: Nạp công cụ ghép đường dẫn.
import cv2  # Giải thích: Nạp OpenCV để đọc webcam, đổi màu và vẽ kết quả.
import mediapipe as mp  # Giải thích: Nạp MediaPipe để tìm các điểm bàn tay.
import numpy as np  # Giải thích: Nạp NumPy để tạo mảng đặc trưng.

model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.p")  # Giải thích: Xác định model.p nằm cạnh file Python đang chạy.
with open(model_path, "rb") as f:  # Giải thích: Mở mô hình để đọc nhị phân; with tự đóng file.
    model_dict = pickle.load(f)  # Giải thích: Đọc từ điển chứa mô hình từ file pickle.
model = model_dict["model"]  # Giải thích: Lấy đối tượng mô hình trong khóa model.

cap = cv2.VideoCapture(0)  # Giải thích: Mở webcam số 0.
if not cap.isOpened():  # Giải thích: Kiểm tra webcam không mở được.
    cap.release()  # Giải thích: Giải phóng tài nguyên camera trước khi báo lỗi.
    raise RuntimeError("Cannot open camera 0.")  # Giải thích: Dừng chương trình và báo không mở được camera.

mp_hands = mp.solutions.hands  # Giải thích: Lấy mô-đun nhận diện bàn tay.
mp_drawing = mp.solutions.drawing_utils  # Giải thích: Lấy công cụ vẽ landmark và đường nối.
mp_drawing_styles = mp.solutions.drawing_styles  # Giải thích: Lấy kiểu vẽ mặc định cho các điểm và đường nối.

# The trained model uses 21 points (42 features) for one hand.  # Giải thích: Mô hình hiện dùng 21 điểm, mỗi điểm có x,y nên có 42 đặc trưng.
hands = mp_hands.Hands(  # Giải thích: Bắt đầu tạo bộ nhận diện bàn tay.
    static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5  # Giải thích: Nhận diện từng ảnh, tối đa 1 bàn tay, ngưỡng tin cậy tối thiểu 0.5.
)  # Giải thích: Kết thúc khởi tạo và lưu đối tượng vào hands.

label_mapping = {0: "A", 1: "B", 2: "L"}  # Giải thích: Đổi mã lớp số nguyên 0,1,2 thành chữ A,B,L; cần khớp ký hiệu đã thu.

try:  # Giải thích: Bắt đầu khối xử lý có finally để luôn dọn tài nguyên khi rời vòng lặp.
    while True:  # Giải thích: Giữ vòng lặp webcam liên tục như cấu trúc bản gốc.

        data_aux = []  # Giải thích: Tạo danh sách tọa độ mới cho từng khung hình.
        x_ = []  # Giải thích: Danh sách x dùng để tìm giới hạn ngang của bàn tay.
        y_ = []  # Giải thích: Danh sách y dùng để tìm giới hạn dọc của bàn tay.

        ret, frame = cap.read()  # Giải thích: Đọc ảnh camera; ret cho biết thành công, frame chứa ảnh.
        if not ret or frame is None:  # Giải thích: Kiểm tra ảnh đọc thất bại hoặc bị rỗng.
            break  # Giải thích: Thoát vòng lặp khi không còn ảnh hợp lệ.

        # Match the mirrored images collected for training.  # Giải thích: Giữ chiều ảnh webcam nhất quán với ảnh đã thu để huấn luyện.
        frame = cv2.flip(frame, 1)  # Giải thích: Lật ngang ảnh để nhìn như gương.
        H, W, _ = frame.shape  # Giải thích: Lấy chiều cao H và chiều rộng W; bỏ qua số kênh màu.

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Giải thích: Đổi BGR của OpenCV sang RGB cho MediaPipe.
        results = hands.process(frame_rgb)  # Giải thích: Nhận diện các điểm bàn tay trong khung hình.

        if results.multi_hand_landmarks:  # Giải thích: Chỉ vẽ các điểm khi có bàn tay.
            for hand_landmarks in results.multi_hand_landmarks:  # Giải thích: Duyệt bàn tay được tìm thấy; cấu hình giới hạn tối đa 1 bàn tay.
                mp_drawing.draw_landmarks(  # Giải thích: Bắt đầu vẽ các điểm và đường nối lên frame.
                    frame,  # Giải thích: Ảnh camera là nơi nhận các nét vẽ.
                    hand_landmarks,  # Giải thích: Các điểm landmark của bàn tay hiện tại.
                    mp_hands.HAND_CONNECTIONS,  # Giải thích: Danh sách cặp điểm nối để thể hiện cấu trúc bàn tay.
                    mp_drawing_styles.get_default_hand_landmarks_style(),  # Giải thích: Chọn kiểu vẽ mặc định cho điểm.
                    mp_drawing_styles.get_default_hand_connections_style()  # Giải thích: Chọn kiểu vẽ mặc định cho đường nối.
                )  # Giải thích: Kết thúc lời gọi vẽ landmark.

        if results.multi_hand_landmarks:  # Giải thích: Chỉ tính khung và dự đoán khi có bàn tay, tránh biến chưa được tạo.
            for hand_landmarks in results.multi_hand_landmarks:  # Giải thích: Duyệt bàn tay để lấy tọa độ.
                for i in range(len(hand_landmarks.landmark)):  # Giải thích: Duyệt 21 điểm theo thứ tự MediaPipe.
                    x = hand_landmarks.landmark[i].x  # Giải thích: Lấy x chuẩn hóa của điểm i.
                    y = hand_landmarks.landmark[i].y  # Giải thích: Lấy y chuẩn hóa của điểm i.
                    data_aux.append((x, y))  # Giải thích: Thêm cặp x,y theo cùng thứ tự dữ liệu huấn luyện.
                    x_.append(x)  # Giải thích: Thêm x vào danh sách dùng tính khung.
                    y_.append(y)  # Giải thích: Thêm y vào danh sách dùng tính khung.

            x1 = max(0, min(W - 1, int(min(x_) * W) - 10))  # Giải thích: Tính mép trái, mở rộng 10 pixel và giới hạn trong chiều rộng ảnh.
            y1 = max(0, min(H - 1, int(min(y_) * H) - 10))  # Giải thích: Tính mép trên, mở rộng 10 pixel và giới hạn trong chiều cao ảnh.
            x2 = max(0, min(W - 1, int(max(x_) * W) + 10))  # Giải thích: Tính mép phải, mở rộng 10 pixel và giới hạn trong chiều rộng ảnh.
            y2 = max(0, min(H - 1, int(max(y_) * H) + 10))  # Giải thích: Tính mép dưới, mở rộng 10 pixel và giới hạn trong chiều cao ảnh.

            # Flatten (21, 2) to (1, 42), matching train_classifier.py.  # Giải thích: Đổi 21 cặp tọa độ thành 1 mẫu gồm 42 số, giống train_classifier.py.
            features = np.asarray(data_aux, dtype=np.float32).reshape(1, -1)  # Giải thích: Tạo mảng float32 và trải phẳng thành hàng (1,42).
            prediction = model.predict(features)  # Giải thích: Dự đoán mã lớp cho một bàn tay bằng mô hình đã học.
            predicted_character = label_mapping[int(prediction[0])]  # Giải thích: Đổi nhãn chuỗi thành số nguyên rồi tra bảng A,B,L.

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 4)  # Giải thích: Vẽ khung màu xanh lá BGR (0,255,0), dày 4; đây là màu hiện có trong file.
            text_y = y1 - 10 if y1 >= 45 else min(H - 10, y1 + 40)  # Giải thích: Đặt chữ phía trên khung hoặc xuống trong khung nếu gần mép trên.
            cv2.putText(  # Giải thích: Bắt đầu ghi chữ dự đoán lên ảnh.
                frame, predicted_character, (x1, text_y),  # Giải thích: Chọn ảnh, chữ dự đoán và vị trí đặt chữ.
                cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA  # Giải thích: Phông Hershey Simplex, cỡ 1.3, màu xanh lá, nét 3, làm mượt LINE_AA.
            )  # Giải thích: Kết thúc lời gọi vẽ chữ.

        cv2.imshow("Hand Detection", frame)  # Giải thích: Hiển thị ảnh, kể cả khi không nhận diện được bàn tay.

        if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q"), 27):  # Giải thích: Chờ khoảng 1 ms, lấy mã phím và kiểm tra q, Q hoặc Esc (27).
            break  # Giải thích: Thoát vòng lặp khi người dùng nhấn phím kết thúc.
finally:  # Giải thích: Luôn thực hiện dọn tài nguyên khi rời try, kể cả lúc phát sinh lỗi.
    hands.close()  # Giải thích: Đóng bộ nhận diện MediaPipe và giải phóng tài nguyên của nó.
    cap.release()  # Giải thích: Trả camera về cho hệ điều hành.
    cv2.destroyAllWindows()  # Giải thích: Đóng tất cả cửa sổ OpenCV.
