import os
import time
import math

import numpy as np
import cv2
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from pyts.image import GramianAngularField
import matplotlib.pyplot as plt
from scipy.signal import medfilt
# === PARAMETERS ===
VIDEO_PATH = "C:/Users/Dell/Desktop/New folder (8)/vid.mp4"
MIN_AREA = 200
CIRC_RANGE = (90, 350)
MEDIAN_FILTER_K = 5
GAUSSIAN_BLUR_K = (7, 7)
GAF_WINDOW = 15  # must be odd

# === VIDEO & PUPIL TRACKING ===
cap = cv2.VideoCapture(VIDEO_PATH)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Total frames: {total_frames}")

center_list = {}
prev_center = (0, 0)
frame_no = 1

while True:
    ret, frame = cap.read()
    if not ret:
        break

    start = time.perf_counter()
    roi=frame[200:400,150:450]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, MEDIAN_FILTER_K)
    blur = cv2.GaussianBlur(blur, GAUSSIAN_BLUR_K, 1.5)
    edges = cv2.Canny(blur, 10, 50)
    edges = cv2.morphologyEx(
        edges,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))
    )

    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    filtered = []
    for c in contours:
        hull = cv2.convexHull(c)
        area = cv2.contourArea(hull)
        if area < MIN_AREA:
            continue
        perim = cv2.arcLength(hull, True)
        if not (CIRC_RANGE[0] <= perim <= CIRC_RANGE[1]):
            continue
        circ = (4 * math.pi * area) / (perim ** 2)
        if circ > 0.85:
            filtered.append(hull)

    # pick best ellipse by minimal 1/circularity
    best = None
    best_score = float('inf')
    for hull in filtered:
        try:
            e = cv2.fitEllipse(hull)
        except cv2.error:
            continue
        perim = cv2.arcLength(hull, True)
        area = cv2.contourArea(hull)
        score = perim**2 / (4 * math.pi * area)  # lower is more circular
        if score < best_score:
            best_score, best = score, e

    drawing = roi.copy()
    center = None
    if best:
        pts = cv2.ellipse2Poly(
            (int(best[0][0]), int(best[0][1])),
            (int(best[1][0] / 2), int(best[1][1] / 2)),
            int(best[2]), 0, 360, 1
        )
        M = cv2.moments(pts)
        if M['m00'] > 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            center = (cx, cy)
            cv2.circle(drawing, center, 3, (0, 255, 0), -1)
            cv2.ellipse(drawing, best, (0, 255, 0), 2)

    # fallback
    center_list[frame_no] = center or prev_center
    prev_center = center_list[frame_no]

    # display FPS & frame
    fps = 1.0 / (time.perf_counter() - start)
    cv2.putText(
        drawing, f"FPS: {fps:.2f}",
        (10, drawing.shape[0] - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
        (0, 255, 255), 2
    )
    cv2.imshow("Pupil Detection", drawing)
    frame_no += 1
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
print(f"Tracked centers: {len(center_list)}")

# === TIME SERIES PREP ===
frames = sorted(center_list)
x_series = np.array([center_list[f][0] for f in frames], dtype=float)

# rolling median for outlier removal (replace cv2.medianBlur with medfilt)
window = max(3, GAF_WINDOW // 5 | 1)  # ensure a small odd window
med = medfilt(x_series, kernel_size=window)  # 1D median
diff = np.abs(x_series - med)
sigma = np.std(diff)
mask = diff > 1.5 * sigma
x_series[mask] = np.nan


# linear interp for gaps
idx = np.arange(len(x_series))
good = ~np.isnan(x_series)
f_interp = interp1d(
    idx[good], x_series[good], kind='linear',
    fill_value="extrapolate"
)
x_series = f_interp(idx)

# adaptive Savitzky–Golay
win = min(GAF_WINDOW, len(x_series) if len(x_series)%2 else len(x_series)-1)
x_smooth = savgol_filter(x_series, win, polyorder=3)

# replace any remaining NaNs
x_smooth = np.nan_to_num(x_smooth, nan=np.nanmean(x_smooth))

# === GAF COMPUTATION ===
def make_gaf(ts, method):
    gaf = GramianAngularField(method=method)
    return gaf.fit_transform(ts.reshape(1, -1))[0]

gasf = make_gaf(x_smooth, "summation")
gadf = make_gaf(x_smooth, "difference")

# === PLOTTING ===
plt.figure(figsize=(12, 8))

plt.subplot(2,2,1)
plt.plot(frames, x_series, '.', alpha=0.4, label="cleaned")
plt.plot(frames, x_smooth, '-', linewidth=2, label="smoothed")
plt.legend()
plt.title("Pupil X over Frames")

plt.subplot(2,2,2)
plt.imshow(gasf, origin='lower', cmap='rainbow')
plt.title("GASF")

plt.subplot(2,2,4)
plt.imshow(gadf, origin='lower', cmap='rainbow')
plt.title("GADF")

plt.tight_layout()
plt.show()

# === SAVE WINDOWED GAFs ===
def save_gafs(ts, window_size, out="GAFs"):
    os.makedirs(out, exist_ok=True)
    n = len(ts) // window_size
    gasf_tr = GramianAngularField(method="summation")
    gadf_tr = GramianAngularField(method="difference")
    for i in range(n):
        seg = ts[i*window_size:(i+1)*window_size]
        if len(seg) != window_size:
            continue
        plt.imsave(f"{out}/gasf_{i}.png",
                   gasf_tr.fit_transform(seg.reshape(1,-1))[0],
                   cmap="rainbow")
        plt.imsave(f"{out}/gadf_{i}.png",
                   gadf_tr.fit_transform(seg.reshape(1,-1))[0],
                   cmap="rainbow")
    print(f"Saved {2*n} images to '{out}'.")

save_gafs(x_smooth, window_size=266)
