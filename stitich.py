import cv2
import numpy as np
import time 

# 计算单应性矩阵
def find_homography(img_a, img_b):
    sift = cv2.SIFT_create()
    # 寻找关键点和描述符
    kp_a, des_a = sift.detectAndCompute(img_a, None)
    kp_b, des_b = sift.detectAndCompute(img_b, None)
    # 使用FLANN匹配器
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des_b, des_a, k=2)

    # 筛选优质匹配点
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    if len(good_matches) >= 4:
        src_pts = np.float32([kp_b[m.queryIdx].pt for m in good_matches]).reshape(-1,1,2)
        dst_pts = np.float32([kp_a[m.trainIdx].pt for m in good_matches]).reshape(-1,1,2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        return M
    else:
        return None
    
    
    

def main():
    # 读取图像
    img_a = cv2.imread('./images/image_4k.jpg')
    img_b = cv2.imread('./images/image_2_45.jpg')
    if img_a is None or img_b is None:
        print("Error loading images!")
        return
    
    # 转换为灰度图像
    gray_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY)


    # 上采样A图像
    img_a_upscaled = cv2.resize(img_a, None, fx=2, fy=2, 
                               interpolation=cv2.INTER_CUBIC)

    t1 = time.time()

    # 计算单应性矩阵
    M = find_homography(gray_a, gray_b)
    if M is None:
        print("Not enough matches found")
        return

    # 计算B的角点在A中的坐标
    h_b, w_b = gray_b.shape
    pts_b = np.float32([[0,0], [0,h_b-1], [w_b-1,h_b-1], [w_b-1,0]]).reshape(-1,1,2)
    pts_a = cv2.perspectiveTransform(pts_b, M)

    # 计算最小包围矩形
    x_coords = [pt[0][0] for pt in pts_a]
    y_coords = [pt[0][1] for pt in pts_a]
    x_min, x_max = int(round(min(x_coords))), int(round(max(x_coords)))
    y_min, y_max = int(round(min(y_coords))), int(round(max(y_coords)))

    t2 = time.time()

    # 计算上采样后区域参数
    up_x = 2 * x_min
    up_y = 2 * y_min
    up_w = 2 * (x_max - x_min)
    up_h = 2 * (y_max - y_min)

    # 创建目标区域的掩码
    mask = np.zeros_like(img_a_upscaled)
    
    # 将变换后的B图像绘制到掩码上
    warped_b = cv2.warpPerspective(img_b, M, (img_a.shape[1], img_a.shape[0]))

    warped_b_upscaled = cv2.resize(warped_b, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    
    # 创建一个ROI区域
    roi = img_a_upscaled[up_y:up_y+up_h, up_x:up_x+up_w]

    # 创建一个与ROI大小相同的warped_b区域
    warped_roi = warped_b_upscaled[up_y:up_y+up_h, up_x:up_x+up_w]
    
    # 创建一个掩码，只选择warped_roi中非黑色的部分
    mask_roi = cv2.cvtColor(warped_roi, cv2.COLOR_BGR2GRAY)
    _, mask_roi = cv2.threshold(mask_roi, 1, 255, cv2.THRESH_BINARY)
    mask_roi = mask_roi.astype(bool)
    
    # 只替换非黑色部分
    roi[mask_roi] = warped_roi[mask_roi]
    img_a_upscaled[up_y:up_y+up_h, up_x:up_x+up_w] = roi

    print(t2 - t1)

    # 输出坐标
    print(f"B在原始A中的坐标：({x_min}, {y_min})")
    print(f"B在上采样后A中的坐标：({up_x}, {up_y})")

    # 保存结果
    cv2.imwrite('enhanced_8k.jpg', img_a_upscaled)
    print("图像处理完成，保存为 enhanced_8k.jpg")

if __name__ == "__main__":
    main()