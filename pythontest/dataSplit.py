import cv2

# 读取视频
video_path = 'video50.mp4'
cap = cv2.VideoCapture(video_path)

# 获取视频的宽度和高度
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 定义输出视频的编解码器和视频写入器
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

# 四个象限的视频写入器（原始分辨率）
out_top_left = cv2.VideoWriter('top_left.mp4', fourcc, 20.0, (width // 2 - 50, height // 2 - 50))
out_top_right = cv2.VideoWriter('top_right.mp4', fourcc, 20.0, (width // 2 - 50, height // 2 - 50))
out_bottom_left = cv2.VideoWriter('bottom_left.mp4', fourcc, 20.0, (width // 2 - 50, height // 2 - 50))
out_bottom_right = cv2.VideoWriter('bottom_right.mp4', fourcc, 20.0, (width // 2 - 50, height // 2 - 50))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 将帧分为四个象限
    top_left = frame[:height // 2  - 50, :width // 2 - 50]
    top_right = frame[:height // 2 - 50, width // 2 + 50:]
    bottom_left = frame[height // 2 + 50:, :width // 2 - 50]
    bottom_right = frame[height // 2 + 50:, width // 2 + 50:]

    # 写入四个象限的视频（原始分辨率）
    out_top_left.write(top_left)
    out_top_right.write(top_right)
    out_bottom_left.write(bottom_left)
    out_bottom_right.write(bottom_right)

# 释放原始视频读取和写入资源
cap.release()
out_top_left.release()
out_top_right.release()
out_bottom_left.release()
out_bottom_right.release()

# 分别读取四个象限的视频并提升分辨率
quadrant_videos = ['top_left.mp4', 'top_right.mp4', 'bottom_left.mp4', 'bottom_right.mp4']
new_width, new_height = 1920, 1080

for video_name in quadrant_videos:
    cap = cv2.VideoCapture(video_name)
    out = cv2.VideoWriter(f'./fourvideo/{video_name[:-4]}_1k.mp4', fourcc, 20.0, (new_width, new_height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 使用 cv2.resize 提升分辨率
        resized_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        out.write(resized_frame)

    cap.release()
    out.release()

cv2.destroyAllWindows()