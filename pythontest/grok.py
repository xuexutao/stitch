import cv2
import numpy as np
import time
from colorama import Fore, Style

def preprocess_image(img):
    """Enhance image contrast for better feature detection."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(img)

def find_homography(img_a, img_b, visualize_matches=False, video_idx=0, use_sift=False):
    if use_sift:
        try:
            detector = cv2.SIFT_create(nfeatures=10000)
        except AttributeError:
            print("SIFT not available. Install opencv-contrib-python or set use_sift=False.")
            return None
    else:
        detector = cv2.ORB_create(
            nfeatures=15000,  # Increased for robustness
            scaleFactor=1.2,
            edgeThreshold=5,  # Relaxed to detect more features
            patchSize=31
        )
    
    t1 = time.time()
    kp_a, des_a = detector.detectAndCompute(img_a, None)
    t2 = time.time()
    print(f"detectAndCompute time (large, video {video_idx}):\t{t2 - t1:.4f} Features in large image:\t{len(kp_a)}")
    
    t1 = time.time()
    kp_b, des_b = detector.detectAndCompute(img_b, None)
    t2 = time.time()
    print(f"detectAndCompute time (small, video {video_idx}):\t{t2 - t1:.4f} Features in small image:\t{len(kp_b)}")

    if des_a is None or des_b is None or len(kp_a) < 4 or len(kp_b) < 4:
        print(f"Insufficient features for video {video_idx}")
        return None

    t1 = time.time()
    if use_sift:
        matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    else:
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = matcher.knnMatch(des_b, des_a, k=2)
    t2 = time.time()
    print(f"knnMatch time (video {video_idx}):\t\t\t{t2 - t1:.4f}")

    good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]
    print(f"Good matches (video {video_idx}):\t\t\t{len(good_matches)}")

    if visualize_matches and len(good_matches) > 0:
        img_matches = cv2.drawMatches(img_b, kp_b, img_a, kp_a, good_matches, None,
                                      flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        cv2.imwrite(f'matches_video_{video_idx}.jpg', img_matches)
        print(f"Saved match visualization for video {video_idx} as matches_video_{video_idx}.jpg")

    if len(good_matches) >= 4:
        src_pts = np.float32([kp_b[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_a[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        t1 = time.time()
        M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        t2 = time.time()
        print(f"findHomography time (video {video_idx}):\t\t{t2 - t1:.4f}")
        return M
    print(f"Not enough good matches for video {video_idx}")
    return None

def merge_four_images(imgs, labels, colors):
    assert len(imgs) == 4, "Four images required"
    height, width = imgs[0].shape[:2]
    output_imgs = []
    
    for img, label, color in zip(imgs, labels, colors):
        if img is None:
            img = np.zeros((height, width, 3), dtype=np.uint8)
            cv2.putText(img, "INVALID", (width//4, height//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            img = img.copy()
            cv2.rectangle(img, (0, 0), (width-1, height-1), color, 3)
            cv2.putText(img, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        output_imgs.append(img)
    
    top = np.hstack((output_imgs[0], output_imgs[1]))
    bottom = np.hstack((output_imgs[2], output_imgs[3]))
    return np.vstack((top, bottom))

def display_input_frames(frames, labels, colors):
    small_frames = [cv2.resize(f, (320, 240)) for f in frames]
    for i, (frame, label, color) in enumerate(zip(small_frames, labels, colors)):
        cv2.rectangle(frame, (0, 0), (319, 239), color, 2)
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    top = np.hstack((small_frames[0], small_frames[2]))
    bottom = np.hstack((small_frames[3], small_frames[4]))
    input_display = np.vstack((top, bottom))
    cv2.imshow("Input Frames", input_display)

def stitch_videos(video_paths, use_sift=False):
    caps = [cv2.VideoCapture(path) for path in video_paths]
    if not all(cap.isOpened() for cap in caps):
        print("Error: Could not open one or more video files")
        return

    homographies = [None] * 4
    first_frame = True
    part_width = part_height = 0
    
    # Quadrant labels and colors for visualization
    labels = ["Top-Left", "Top-Right", "Bottom-Left", "Bottom-Right"]
    colors = [(0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255)]  # Green, Red, Cyan, Magenta

    while True:
        frames = []
        for i, cap in enumerate(caps):
            ret, frame = cap.read()
            if not ret:
                print(f"End of video {i} or error reading frame")
                for cap in caps:
                    cap.release()
                cv2.destroyAllWindows()
                return
            frames.append(frame)

        # Large FOV: frames[1], Small FOVs: [frames[0], frames[2], frames[3], frames[4]]
        img_a = frames[1]  # Large view (center)
        # Reorder to match quadrants: top-left, top-right, bottom-left, bottom-right
        img_bs = [frames[0], frames[2], frames[3], frames[4]]

        gray_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY)
        gray_a = preprocess_image(gray_a)
        img_a_upscaled = cv2.resize(img_a, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        
        if first_frame:
            part_width = img_a_upscaled.shape[1] // 2
            part_height = img_a_upscaled.shape[0] // 2
            for i, img_b in enumerate(img_bs):
                gray_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY)
                gray_b = preprocess_image(gray_b)
                M = find_homography(gray_a, gray_b, visualize_matches=True, video_idx=i, use_sift=use_sift)
                if M is None:
                    print(f"Warning: Homography failed for video {i} ({labels[i]})")
                    homographies[i] = None
                else:
                    homographies[i] = M
            first_frame = False

        imgs = []
        t1 = time.time()
        for i, (img_b, M) in enumerate(zip(img_bs, homographies)):
            if M is None:
                print(f"Skipping video {i} ({labels[i]}) due to invalid homography")
                imgs.append(None)
                continue

            h_b, w_b = img_b.shape[:2]
            pts_b = np.float32([[0,0], [0,h_b-1], [w_b-1,h_b-1], [w_b-1,0]]).reshape(-1,1,2)
            pts_a = cv2.perspectiveTransform(pts_b, M)

            x_coords = [pt[0][0] for pt in pts_a]
            y_coords = [pt[0][1] for pt in pts_a]
            x_min, x_max = int(round(min(x_coords))), int(round(max(x_coords)))
            y_min, y_max = int(round(min(y_coords))), int(round(max(y_coords)))

            up_x = 2 * x_min
            up_y = 2 * y_min
            up_w = 2 * (x_max - x_max)
            up_h = 2 * (y_max - y_min)

            up_w = max(1, up_w)
            up_h = max(1, up_h)
            img_b_resized = cv2.resize(img_b, (up_w, up_h), interpolation=cv2.INTER_CUBIC)

            up_x = max(0, up_x)
            up_y = max(0, up_y)
            up_x_end = min(up_x + up_w, img_a_upscaled.shape[1])
            up_y_end = min(up_y + up_h, img_a_upscaled.shape[0])
            up_w = up_x_end - up_x
            up_h = up_y_end - up_y

            if up_w > 0 and up_h > 0:
                img_b_resized = img_b_resized[:up_h, :up_w]
                img_a_upscaled[up_y:up_y + up_h, up_x:up_x + up_w] = img_b_resized
                img_a_upscaled[up_y:up_y + up_h, up_x:up_x + 3] = colors[i]
                img_a_upscaled[up_y:up_y + 3, up_x:up_x + up_w] = colors[i]
                img_a_upscaled[up_y + up_h:up_y + up_h + 3, up_x:up_x + up_w] = colors[i]
                img_a_upscaled[up_y:up_y + up_h, up_x + up_w:up_x + up_w + 3] = colors[i]
            else:
                print(f"Invalid replacement area for video {i} ({labels[i]})")
                imgs.append(None)
                continue

            if i == 0:  # Top-left
                x, y = 0, 0
            elif i == 1:  # Top-right
                x, y = part_width, 0
            elif i == 2:  # Bottom-left
                x, y = 0, part_height
            elif i == 3:  # Bottom-right
                x, y = part_width, part_height

            imgs.append(img_a_upscaled[y:y + part_height, x:x + part_width])
            
            cv2.imwrite(img_a_upscaled[y:y + part_height, x:x + part_width], f"{i + 1}.jpg")
            

        merged_img = merge_four_images(imgs, labels, colors)
        if merged_img is not None:
            cv2.imshow("Enhanced Image", merged_img)
        
        # Display input frames for comparison
        display_input_frames(frames, ["Top-Left", "Large", "Top-Right", "Bottom-Left", "Bottom-Right"], colors + [(255, 255, 255)])

        t2 = time.time()
        print(f"Frame processing time:\t\t{Fore.RED}{t2 - t1:.4f}{Style.RESET_ALL}")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    for cap in caps:
        cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_paths = [
        './fourvideo/top_left_1k.mp4',
        './fourvideo/video5_1k_gray.mp4',
        './fourvideo/top_right_1k.mp4',
        './fourvideo/bottom_left_1k.mp4',
        './fourvideo/bottom_right_1k.mp4'
    ]
    stitch_videos(video_paths, use_sift=False)  # Set use_sift=True if opencv-contrib-python is installed