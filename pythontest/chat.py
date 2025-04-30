import cv2
import numpy as np
import time


def find_homography(img_a, img_b):
    akaze = cv2.AKAZE_create()
    kp_a, des_a = akaze.detectAndCompute(img_a, None)
    kp_b, des_b = akaze.detectAndCompute(img_b, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(des_b, des_a, k=2)

    good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]

    if len(good_matches) >= 4:
        src_pts = np.float32([kp_b[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_a[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        return M
    else:
        return None


def apply_homography_and_blend(img_a_up, img_b, M):
    h_b, w_b = img_b.shape[:2]
    warped = cv2.warpPerspective(img_b, M * 2.0, (img_a_up.shape[1], img_a_up.shape[0]))

    mask = (cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY) > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 复制原图用于画边缘
    result = img_a_up.copy()
    result = cv2.drawContours(result, contours, -1, (0, 0, 255), 2)  # 红色边缘

    img_a_up = cv2.seamlessClone(warped, result, mask, (img_a_up.shape[1] // 2, img_a_up.shape[0] // 2), cv2.NORMAL_CLONE)
    return img_a_up


class FovEnhancer:
    def __init__(self):
        self.homographies = [None] * 4
        self.initialized = False

    def process_first_frame(self, img_a, img_bs):
        gray_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY)
        for i, img_b in enumerate(img_bs):
            gray_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY)
            M = find_homography(gray_a, gray_b)
            if M is not None:
                self.homographies[i] = M
            else:
                raise RuntimeError(f"匹配第{i}个小图像失败，未能获得单应性矩阵")
        self.initialized = True

    def stitch(self, img_a, img_bs):
        if not self.initialized:
            self.process_first_frame(img_a, img_bs)

        img_a_up = cv2.resize(img_a, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

        for i, (img_b, M) in enumerate(zip(img_bs, self.homographies)):
            if M is not None:
                img_a_up = apply_homography_and_blend(img_a_up, img_b, M)
            else:
                print(f"跳过第{i}个图像，无效的单应性矩阵")

        return img_a_up


if __name__ == '__main__':
    video_paths = [
        './fourvideo/top_left_1k.mp4',
        './fourvideo/video5_1k.mp4',
        './fourvideo/top_right_1k.mp4',
        './fourvideo/bottom_left_1k.mp4',
        './fourvideo/bottom_right_1k.mp4'
    ]

    caps = [cv2.VideoCapture(p) for p in video_paths]
    enhancer = FovEnhancer()

    while True:
        frames = []
        rets = []
        for cap in caps:
            ret, frame = cap.read()
            rets.append(ret)
            frames.append(frame)
        if not all(rets):
            break

        small_fovs = [frames[0], frames[2], frames[3], frames[4]]
        large_fov = frames[1]
        enhanced = enhancer.stitch(large_fov, small_fovs)

        cv2.imshow("Enhanced FOV", enhanced)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    for cap in caps:
        cap.release()
    cv2.destroyAllWindows()
