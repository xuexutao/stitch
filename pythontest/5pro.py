import cv2

# 读取视频
video_path = 'video50.mp4'
cap = cv2.VideoCapture(video_path)

# 获取视频的原始宽度和高度
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 定义输出视频的编解码器和视频写入器
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('video5_1k.mp4', fourcc, 20.0, (1920, 1080))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 使用cv2.resize提升分辨率
    resized_frame = cv2.resize(frame, (1920, 1080), interpolation=cv2.INTER_LINEAR)
    out.write(resized_frame)

# 释放资源
cap.release()
out.release()
cv2.destroyAllWindows()