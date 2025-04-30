import cv2

# 输入视频文件路径
input_video_path = './fourvideo/video5_1k.mp4'
# 输出视频文件路径
output_video_path = './fourvideo/video5_1k_gray.mp4'

# 打开输入视频文件
cap = cv2.VideoCapture(input_video_path)

# 获取视频的宽度、高度和帧率
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# 定义输出视频的编解码器和视频写入器
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height), isColor=False)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 将帧转换为灰度图
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 将灰度帧写入输出视频
    out.write(gray_frame)

# 释放资源
cap.release()
out.release()
cv2.destroyAllWindows()