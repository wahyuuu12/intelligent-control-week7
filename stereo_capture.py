import cv2
import numpy as np

# Inisialisasi StereoSGBM
min_disp = 0
num_disp = 16 * 6  # Harus kelipatan 16
block_size = 5

stereo = cv2.StereoSGBM_create(
    minDisparity=min_disp,
    numDisparities=num_disp,
    blockSize=block_size,
    P1=8 * 3 * block_size**2,
    P2=32 * 3 * block_size**2,
    disp12MaxDiff=1,
    uniquenessRatio=10,
    speckleWindowSize=100,
    speckleRange=32
)

def compute_depth(left_image, right_image):
    """Menghitung peta kedalaman dari dua gambar stereo"""
    grayL = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)

    disparity = stereo.compute(grayL, grayR).astype(np.float32) / 16.0

    # Normalisasi agar bisa ditampilkan
    disp_norm = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    disp_norm = np.uint8(disp_norm)

    return disp_norm

# Buka video capture dari dua kamera
cam_left = cv2.VideoCapture(1)
cam_right = cv2.VideoCapture(2)

if not cam_left.isOpened() or not cam_right.isOpened():
    print("Tidak dapat membuka salah satu atau kedua kamera.")
    exit()

while True:
    retL, frameL = cam_left.read()
    retR, frameR = cam_right.read()

    if not retL or not retR:
        print("Gagal membaca frame.")
        break

    depth_map = compute_depth(frameL, frameR)

    # Tampilkan frame asli
    cv2.imshow("Kamera Kiri", frameL)
    cv2.imshow("Kamera Kanan", frameR)

    # Tampilkan peta kedalaman
    cv2.imshow("Depth Map (SGBM)", depth_map)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam_left.release()
cam_right.release()
cv2.destroyAllWindows()
