import os                                               # Nạp os để tạo thư mục và ghép đường dẫn.
import cv2                                              # Nạp OpenCV để đọc camera, hiển thị và mã hóa ảnh.

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")  # Đường dẫn thư mục data nằm cạnh file Python này, không phụ thuộc nơi chạy terminal.
if not os.path.exists(DATA_DIR):                        # Kiểm tra thư mục data chưa tồn tại.
    os.makedirs(DATA_DIR)                               # Tạo thư mục data để chứa ảnh thu thập.

number_of_classes = 3                                   # Thu dữ liệu cho 3 lớp, có mã 0, 1 và 2.
dataset_size = 100                                      # Mỗi lớp dự kiến thu 100 ảnh; chạy lại có thể ghi đè tên ảnh cũ.

cap = cv2.VideoCapture(0)                               # Mở camera có chỉ số 0, thường là webcam mặc định.
for j in range(number_of_classes):                      # Lần lượt thu ảnh cho từng lớp j từ 0 đến 2.
    if not os.path.exists(os.path.join(DATA_DIR, str(j))):  # Kiểm tra thư mục riêng của lớp hiện tại chưa tồn tại.
        os.makedirs(os.path.join(DATA_DIR, str(j)))     # Tạo thư mục data/0, data/1 hoặc data/2 tương ứng.

    print('Collecting data for class {}'.format(j))     # In lớp đang thu để người dùng biết cần tạo ký hiệu nào.

    done = False                                        # Gán False cho biến done; biến này hiện không được dùng ở phần sau.
    while True:                                         # Lặp hiển thị camera cho đến khi người dùng nhấn q.
        if not cap.isOpened():                          # camera không mở được; khác với kiểm tra đọc ảnh thành công.
            raise RuntimeError("Không mở được camera")  # Dừng chương trình và báo không mở được camera.
        ret, frame = cap.read()                         # Đọc ảnh camera; ret báo thành công, frame chứa ảnh; hiện chưa kiểm tra ret ở đây.
        frame = cv2.flip(frame, 1)                      # Lật ảnh theo chiều ngang để hiển thị như gương; cần frame hợp lệ.
        cv2.putText(frame, 'Ready? Press "Q" ! :)', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3,  # Vẽ lời nhắc tại (100, 50), cỡ 1.3, màu xanh lá BGR (0,255,0), nét dày 3.
                    cv2.LINE_AA)                        # Dùng LINE_AA để làm mượt cạnh chữ và kết thúc lời gọi putText.
        cv2.imshow('frame', frame)                      # Hiển thị ảnh trong cửa sổ tên frame.
        if cv2.waitKey(25) == ord('q'):                 # Chờ phím khoảng 25 ms; chỉ nhận q thường khi cửa sổ camera có tiêu điểm.
            break                                       # Thoát vòng chờ để bắt đầu thu ảnh cho lớp hiện tại.

    counter = 0                                         # Đặt chỉ số ảnh đầu tiên là 0 cho mỗi lớp.
    while counter < dataset_size:                       # Tiếp tục thu khi số ảnh đã ghi chưa đạt dataset_size.
        if not cap.isOpened():                          # Kiểm tra trạng thái mở của camera trong lúc thu ảnh.
            raise RuntimeError("Không mở được camera")  # Dừng và báo lỗi nếu camera không mở.
        ret, frame = cap.read()                         # Đọc khung hình mới; ret vẫn chưa được kiểm tra trước khi xử lý.
        frame = cv2.flip(frame, 1)                      # Lật ngang ảnh; ảnh lưu sau đó cũng có chiều như gương.
        cv2.imshow('frame', frame)                      # Hiển thị khung hình đang được thu.
        cv2.waitKey(25)                                 # Chờ khoảng 25 ms và xử lý sự kiện cửa sổ; kết quả phím bị bỏ qua.
        image_path = os.path.join(DATA_DIR, str(j), f"{counter}.jpg")  # Tạo tên ảnh như data/0/0.jpg; counter giúp mỗi ảnh có chỉ số riêng trong lượt thu.

        # OpenCV mã hóa ảnh thành JPEG trong bộ nhớ  
        success, encoded_image = cv2.imencode(".jpg", frame)  # Đổi frame thành dữ liệu JPEG; success báo mã hóa thành công.
        if not success:                                       # Kiểm tra mã hóa JPEG thất bại.
            raise RuntimeError("Không mã hóa được ảnh JPEG")  # Dừng và báo lỗi nếu không tạo được dữ liệu JPEG.

        # Python ghi file, hỗ trợ đường dẫn có dấu 
        with open(image_path, "wb") as file:            # Mở file ở chế độ ghi nhị phân; wb ghi đè nếu tên file đã tồn tại; with tự đóng file.
            file.write(encoded_image.tobytes())         # Chuyển dữ liệu JPEG thành bytes rồi ghi vào file ảnh.

        print("Đã lưu:", image_path)                    # In đường dẫn ảnh vừa ghi để theo dõi tiến độ.
        counter += 1                                    # Tăng bộ đếm sau khi ghi file hoàn tất.

cap.release()                                           # Giải phóng camera sau khi thu xong; lỗi trước đó có thể khiến dòng này không được chạy.
cv2.destroyAllWindows()                                 # Đóng tất cả cửa sổ OpenCV.