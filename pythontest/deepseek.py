import cv2
import numpy as np
import time
from colorama import Fore, Style

# 全局变量用于保存单应性矩阵、拼接图尺寸及首次运行标志
homography_matrices = None
panorama_size = None
first_frame = True

def find_homography(img_a, img_b, visualize_matches=False):
    orb = cv2.ORB_create(nfeatures=5000,
                         scaleFactor=1.3,
                         edgeThreshold=15,
                         patchSize=31)
    kp_a, des_a = orb.detectAndCompute(img_a, None)
    kp_b, des_b = orb.detectAndCompute(img_b, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(des_b, des_a, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    if len(good_matches) >= 4:
        src_pts = np.float32([kp_b[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_a[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        return M
    else:
        return None

def calculate_panorama_params(frames, matrices):
    all_corners = []
    for i in range(4):
        h, w = frames[i].shape[:2]
        corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32).reshape(-1, 1, 2)
        transformed = cv2.perspectiveTransform(corners, matrices[i])
        all_corners.append(transformed.reshape(-1, 2))
    
    all_points = np.concatenate(all_corners)
    min_x, min_y = np.floor(np.min(all_points, axis=0)).astype(int)
    max_x, max_y = np.ceil(np.max(all_points, axis=0)).astype(int)
    
    # 计算全景图尺寸和偏移量
    panorama_width = max_x - min_x
    panorama_height = max_y - min_y
    
    # 修正单应性矩阵加入平移补偿
    adjusted_matrices = []
    for matrix in matrices:
        adjust_matrix = np.array([[1, 0, -min_x], [0, 1, -min_y], [0, 0, 1]])
        adjusted = adjust_matrix @ matrix
        adjusted_matrices.append(adjusted)
    
    return adjusted_matrices, (panorama_width, panorama_height)

def stitch(frames):
    global homography_matrices, panorama_size, first_frame

    if first_frame:
        print(Fore.YELLOW + "首次帧处理，计算单应性矩阵..." + Style.RESET_ALL)
        base_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
        matrices = [np.eye(3)]
        
        for i in range(1, 4):
            target_gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
            M = find_homography(base_gray, target_gray)
            if M is None:
                print(Fore.RED + f"错误：无法计算视频流{i}的单应性矩阵" + Style.RESET_ALL)
                return
            matrices.append(M)
        
        # 计算全景图参数并修正矩阵
        homography_matrices, panorama_size = calculate_panorama_params(frames, matrices)
        first_frame = False
        print(Fore.GREEN + "单应性矩阵计算完成，开始实时拼接" + Style.RESET_ALL)

    # 创建全景图画布
    panorama = np.zeros((panorama_size[1], panorama_size[0], 3), dtype=np.uint8)
    
    # 实时拼接处理
    for i in range(4):
        warped = cv2.warpPerspective(frames[i], homography_matrices[i], panorama_size)
        
        # 创建有效区域掩膜
        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        
        # 叠加图像
        panorama = cv2.bitwise_or(panorama, cv2.bitwise_and(warped, warped, mask=mask))
    
    # 显示结果（添加FPS显示）
    cv2.imshow('Real-time Panorama', panorama)

if __name__ == "__main__":
    video_paths = [
        './fourvideo/top_left_1k.mp4',
        './fourvideo/top_right_1k.mp4',
        './fourvideo/bottom_left_1k.mp4',
        './fourvideo/bottom_right_1k.mp4'
    ]

    caps = [cv2.VideoCapture(path) for path in video_paths]
    
    # 设置视频流参数（根据实际需要调整）
    target_fps = 25
    frame_interval = int(1000 / target_fps)

    while True:
        start_time = time.time()
        frames = []
        rets = []
        
        # 读取四路视频帧
        for cap in caps:
            ret, frame = cap.read()
            rets.append(ret)
            if ret:
                # 调整帧大小（可选）
                frames.append(cv2.resize(frame, (640, 360)))  # 调整为原尺寸的1/3
            else:
                frames.append(None)
        
        # 检查是否所有帧都读取成功
        if not all(rets):
            print(Fore.RED + "视频流结束或读取错误" + Style.RESET_ALL)
            break
        
        # 执行拼接
        stitch(frames)
        
        # 计算处理耗时
        processing_time = (time.time() - start_time) * 1000
        delay = max(1, int(frame_interval - processing_time))
        
        # 退出控制
        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break

    # 释放资源
    for cap in caps:
        cap.release()
    cv2.destroyAllWindows()