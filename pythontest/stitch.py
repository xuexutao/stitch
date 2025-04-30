import cv2
import numpy as np

import cv2
import numpy as np
import time
from colorama import Fore, Style
import argparse


def calculate_psnr(image1, image2):
    # 确保两个图像具有相同的尺寸
    if image1.shape!= image2.shape:
        raise ValueError("两个图像必须具有相同的尺寸")

    # 将图像转换为浮点数类型，以避免整数除法和溢出
    image1 = image1.astype(np.float64)
    image2 = image2.astype(np.float64)

    # 计算均方误差（MSE）
    mse = np.mean((image1 - image2) ** 2)

    if mse == 0:
        return float('inf')  # 如果MSE为0，PSNR为无穷大

    # 计算PSNR
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))

    return psnr


# img_a  :  大图
# img_b  :  小图

def find_homography(img_a, img_b, visualize_matches=False):
    orb = cv2.ORB_create(nfeatures=10000,
                            scaleFactor=1.3,      # 金字塔缩放因子(原1.2)
                            edgeThreshold=15,     # 边界阈值
                            patchSize=31          # 描述符区域大小
                        )                         # 限制特征点数量
    t1 = time.time()
    kp_a, des_a = orb.detectAndCompute(img_a, None)
    t2 = time.time()
    print("detectAndCompute time 00:\t" , t2 - t1, "大图像特征点个数为：\t", len(kp_a))
    
    t1 = time.time()
    kp_b, des_b = orb.detectAndCompute(img_b, None)
    t2 = time.time()
    print("detectAndCompute time 44:\t" , t2 - t1, "小图像特征点个数为：\t", len(kp_b))

    # ===============================================================

    t1 = time.time()
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False) 
    t2 = time.time()
    print("FlannBasedMatcher time:\t\t" , t2 - t1)

    # ===============================================================
    
    t1 = time.time()
    matches = bf.knnMatch(des_b, des_a, k=2)
    t2 = time.time()
    print("knnMatch time:\t\t\t" , t2 - t1)

    # ===============================================================

    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)
    print("len good matchers: " , len(good_matches))

    # ===============================================================

    if visualize_matches:
        # 创建匹配可视化图像
        img_matches = cv2.drawMatches(img_b, kp_b, img_a, kp_a, good_matches, None,
                                     flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        cv2.imwrite('Matchers.jpg', img_matches)

    if len(good_matches) >= 4:
        src_pts = np.float32([kp_b[m.queryIdx].pt for m in good_matches]).reshape(-1,1,2)
        dst_pts = np.float32([kp_a[m.trainIdx].pt for m in good_matches]).reshape(-1,1,2)
        t1 = time.time()
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        t2 = time.time()
        print("findHomography time:\t\t" , t2 - t1)
        return M
    else:
        return None
    
def merge_four_images(imgs):
    # 确保有四张图片
    assert len(imgs) == 4, "需要提供四张图片"
    
    # 获取每张图片的高度和宽度
    top_left = imgs[0]
    top_right = imgs[1]
    bottom_left = imgs[2]
    bottom_right = imgs[3]
    
    # 检查图片是否为空
    if top_left is None or top_right is None or bottom_left is None or bottom_right is None:
        raise ValueError("有一张或多张图片为空")
    
    # 获取图片的高度和宽度（假设所有图片大小相同）
    height, width = top_left.shape[:2]
    
    # 水平拼接上方两张图片
    top = np.hstack((top_left, top_right))
    # 水平拼接下方两张图片
    bottom = np.hstack((bottom_left, bottom_right))
    # 垂直拼接上下两部分
    merged_img = np.vstack((top, bottom))
    
    return merged_img


# input five images   
def stitch(retts : list, first, MS, padding_color = False ):

    img_a = retts[1] # 大视野图像
    img_bs = [retts[0],retts[2], retts[3], retts[4]]

    # 转换为灰度图像
    gray_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY)

    # 上采样A图像
    img_a_upscaled = cv2.resize(img_a, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    part_width = img_a_upscaled.shape[1] // 2
    part_height = img_a_upscaled.shape[0] // 2

    # ===============================================================
    imgs = []

    if first == 1:
        MS = []
    else: 
        MS = MS

    for i, img_b in enumerate(img_bs):
        gray_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY)
        # ===============================================================
        t1 = time.time()
        M = None
        if first == 1:
            # 计算单应性矩阵
            M = find_homography(gray_a, gray_b,  visualize_matches = True)
            if M is None:
                print("Not enough matches found")
                return
            MS.append(M)
        else:
            M = MS[i]
        # 计算B的角点在A中的坐标
        h_b, w_b = gray_b.shape
        pts_b = np.float32([[0,0], [0,h_b-1], [w_b-1,h_b-1], [w_b-1,0]]).reshape(-1,1,2)
        pts_a = cv2.perspectiveTransform(pts_b, M) # 对一组点进行透视变换

        # 计算最小包围矩形
        x_coords = [pt[0][0] for pt in pts_a]
        y_coords = [pt[0][1] for pt in pts_a]

        x_min, x_max = int(round(min(x_coords))), int(round(max(x_coords)))
        y_min, y_max = int(round(min(y_coords))), int(round(max(y_coords)))
        
        # 计算上采样后区域参数
        up_x = 2 * x_min
        up_y = 2 * y_min
        up_w = 2 * (x_max - x_min)
        up_h = 2 * (y_max - y_min)

        t2 = time.time()
        # ===============================================================

        up_w = max(1, up_w)
        up_h = max(1, up_h)
        # 调整B图像尺寸并替换
        img_b_resized = cv2.resize(img_b, (up_w, up_h), interpolation=cv2.INTER_CUBIC)

        up_x = max(0, up_x)
        up_y = max(0, up_y)
        up_x_end = min(up_x + up_w, img_a_upscaled.shape[1])
        up_y_end = min(up_y + up_h, img_a_upscaled.shape[0])
        up_w = up_x_end - up_x
        up_h = up_y_end - up_y

        def alpha_blending(img1, img2, blend_width):
            height, width = img1.shape[:2]
            alpha = np.zeros((height, width), dtype=np.float32)
            # 生成渐变的 alpha 通道
            for i in range(blend_width):
                alpha[:, width - blend_width + i] = i / blend_width
            blended_img = img1 * (1 - alpha[..., np.newaxis]) + img2 * alpha[..., np.newaxis]
            return blended_img.astype(np.uint8)
        
        if up_w > 0 and up_h > 0:
            img_b_resized = img_b_resized[:up_h, :up_w]
            # 提取要替换的区域
            target_region = img_a_upscaled[up_y:up_y + up_h, up_x:up_x + up_w]
            # 定义融合宽度
            blend_width = 20
            # 进行 Alpha 融合
            blended_region = alpha_blending(target_region, img_b_resized, blend_width)
            img_a_upscaled[up_y:up_y + up_h, up_x:up_x + up_w] = blended_region
            if padding_color:
                img_a_upscaled[up_y:up_y + up_h, up_x:up_x + 3] = [[0, 0, 255]]
                img_a_upscaled[up_y:up_y + 3, up_x:up_x + up_w] = [[0, 0, 255]]
                img_a_upscaled[up_y + up_h:up_y + up_h + 3, up_x:up_x + up_w] = [[0, 0, 255]]
                img_a_upscaled[up_y:up_y + up_h, up_x + up_w:up_x + up_w + 3] = [[0, 0, 255]]

        # if up_w > 0 and up_h > 0:
        #     img_b_resized = img_b_resized[:up_h, :up_w]
        #     img_a_upscaled[up_y:up_y + up_h, up_x:up_x + up_w] = img_b_resized
            # img_a_upscaled[up_y:up_y + up_h, up_x:up_x + 3] = [[0, 0, 255]]
            # img_a_upscaled[up_y:up_y + 3, up_x:up_x + up_w] = [[0, 0, 255]]
            # img_a_upscaled[up_y + up_h:up_y + up_h + 3, up_x:up_x + up_w] = [[0, 0, 255]]
            # img_a_upscaled[up_y:up_y + up_h, up_x + up_w:up_x + up_w + 3] = [[0, 0, 255]]
        else:
            print("替换区域无效")
            return

        print(f"total time:\t\t\t {Fore.RED}{ t2 - t1 }{Style.RESET_ALL}")

        # 输出坐标
        print(f"B在原始A中的坐标:({x_min}, {y_min})")
        print(f"B在上采样后A中的坐标:({up_x}, {up_y})")

        if i == 0:  # 左上.
            x, y = 0, 0
        elif i == 1:  # 右上
            x, y = part_width, 0
        elif i == 2:  # 左下
            x, y = 0, part_height
        elif i == 3:  # 右下
            x, y = part_width, part_height

        # 保存结果
        # cv2.imwrite('enhanced_8k.jpg', img_a_upscaled)
        imgs.append(img_a_upscaled[y:y + part_height, x:x + part_width])

    merged_img = merge_four_images(imgs)
    # cv2.imwrite("./merged_img.jpg", merged_img)
    # cv2.imshow("Merged Image", merged_img)
    # cv2.waitKey(0)
    return merged_img, MS


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-z", "--zhanshi", action='store_false')
    args = parser.parse_args()
    zhanshi = args.zhanshi
    global first
    video_paths = [
        './fourvideo/top_left_1k.mp4',
        './fourvideo/video5_1k.mp4',
        './fourvideo/top_right_1k.mp4',
        './fourvideo/bottom_left_1k.mp4',
        './fourvideo/bottom_right_1k.mp4'
    ]
    
    # 打开视频文件
    caps = [cv2.VideoCapture(path) for path in video_paths]
    first = 1
    MS = []
    choose = True
    padding_color = True
    while True:
        frames = []
        ret_frames = []
        for cap in caps:
            ret, frame = cap.read()
            ret_frames.append(ret)
            frames.append(frame)

        # 如果有任何一个视频读取失败，退出循环
        if not all(ret_frames):
            break
        
        if not choose:
            padding_color = False

        stitched_frame, _MS = stitch(frames, first, MS, padding_color)
        if choose:
            MS = _MS
        psnr = calculate_psnr(stitched_frame, cv2.resize(frames[1], None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC))
        cv2.putText(stitched_frame, f"PSNR: {psnr:.2f} dB", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.imshow("Stitched Video Stream", stitched_frame)
        cv2.waitKey(1)
        if zhanshi:
            choose = False
            first += 1

        if choose :
            key = cv2.waitKey(10000) & 0xFF
            if key == ord('q'):
                continue
            elif key == ord('a'): 
                print( "----------------" )
                choose = False
                first += 1
    for cap in caps:
        cap.release()
    cv2.destroyAllWindows()