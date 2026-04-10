from ultralytics import YOLO
import cv2

# 加载预训练的 YOLOv8 模型
model = YOLO("yolov8n.pt")  # nano版本小巧，训练快

# 视频路径
video_path = "/Users/misharau/dance-ai/train1.mp4"
cap = cv2.VideoCapture(video_path)

# 输出视频设置
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("output.mp4", fourcc, 30, (int(cap.get(3)), int(cap.get(4))))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO 检测
    results = model.predict(frame)

    # 只画人的框（class=0 通常是 person）
    for r in results:
        for box in r.boxes:  # boxes 是预测的框
            if int(box.cls[0]) == 0:  # 0 = person
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

    out.write(frame)
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
