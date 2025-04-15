import cv2
import numpy as np
import math
import csv

# === Parameter Kalibrasi Stereo (ubah sesuai hasil kalibrasi kamu)
focal_length = 700     # dalam pixel
baseline = 6.0         # dalam cm
scale_cm_per_pixel = 0.05  # contoh konversi skala piksel ke cm

# === Inisialisasi StereoBM
stereo = cv2.StereoBM_create(numDisparities=64, blockSize=15)

def compute_depth_map(left_img, right_img):
    grayL = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
    disparity = stereo.compute(grayL, grayR).astype(np.float32) / 16.0
    disparity[disparity <= 0] = 0.1  # hindari divide by zero
    depth_map = (focal_length * baseline) / disparity
    return depth_map

def analyze_track(depth_map, edges):
    height, width = depth_map.shape

    # Estimasi posisi rel
    x_center = width // 2
    x_left = x_center - 100
    x_right = x_center + 100
    y_top = height // 3
    y_bottom = height - 10

    # Lebar Rel
    left_depth = np.mean(depth_map[y_top:y_bottom, x_left])
    right_depth = np.mean(depth_map[y_top:y_bottom, x_right])
    lebar_cm = abs(x_right - x_left) * scale_cm_per_pixel

    # Panjang Rel
    panjang_cm = np.percentile(depth_map[y_top:, x_center - 20:x_center + 20], 95)

    # Kemiringan Rel
    kemiringan_rad = math.atan2(left_depth - right_depth, lebar_cm)
    kemiringan_deg = math.degrees(kemiringan_rad)

    # Deformasi Ballast
    y_ballast = int(height * 0.9)
    ballast_depth = depth_map[y_ballast-20:y_ballast, x_left:x_right]
    deformasi = np.std(ballast_depth)

    return round(lebar_cm, 2), round(panjang_cm, 1), round(kemiringan_deg, 2), round(deformasi, 2)

# === Buka video dari dua sumber kamera
capL = cv2.VideoCapture(1)
capR = cv2.VideoCapture("http://192.168.1.122:4747/video")

# === Setup file CSV
csv_file = open('tabel_inspeksi.csv', mode='w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["No Track", "Lebar Rel (cm)", "Panjang Rel (cm)", "Kemiringan Rel (°)", "Identifikasi Deformasi Ballast (cm)"])

no_track = 1
while True:
    retL, frameL = capL.read()
    retR, frameR = capR.read()
    if not retL or not retR:
        break

    # Resize
    frameL = cv2.resize(frameL, (640, 480))
    frameR = cv2.resize(frameR, (640, 480))

    # Hitung Depth Map
    depth_map = compute_depth_map(frameL, frameR)

    # Deteksi Canny Edge
    edges = cv2.Canny(cv2.cvtColor(frameL, cv2.COLOR_BGR2GRAY), 100, 200)

    # Tampilkan video dan depth map
    cv2.imshow("Left", frameL)
    norm_depth = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
    cv2.imshow("Depth Map", norm_depth.astype(np.uint8))

    key = cv2.waitKey(1) & 0xFF

    if key == ord('s'):
        # Analisis Track
        lebar, panjang, kemiringan, deformasi = analyze_track(depth_map, edges)

        # Simpan ke CSV
        csv_writer.writerow([no_track, lebar, panjang, kemiringan, deformasi])

        # Tampilkan hasil
        print(f"[Track {no_track}] Lebar: {lebar} cm | Panjang: {panjang} cm | Kemiringan: {kemiringan}° | Deformasi: {deformasi} cm")

        # Tampilkan teks di layar
        cv2.putText(frameL, f"Track {no_track} Disimpan", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Left", frameL)
        cv2.waitKey(500)  # tampilkan selama 0.5 detik

        no_track += 1

    elif key == ord('q'):
        break

# === Cleanup
capL.release()
capR.release()
csv_file.close()
cv2.destroyAllWindows()