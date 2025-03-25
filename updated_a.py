import numpy as np
import matplotlib.pyplot as plt
import cv2
import time
import math

# ========= Video processing (existing code) =========
video_path = "C:/Users/Dell/Desktop/New folder (8)/output0.mp4"  # Replace with your video file path
cap = cv2.VideoCapture(video_path)
Total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
center_list = {} 
frame_no = 1

print(f"Total number of frames in the video: {Total_frames}")
while cap.isOpened():
    
    ret, frame = cap.read()
    if not ret:
        break

    start = time.perf_counter()

    # --- Preprocessing ---
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 5)
    blurred = cv2.GaussianBlur(blur, (7, 7), 1.5, 1.5, cv2.BORDER_REPLICATE)
    edges = cv2.Canny(blurred, threshold1=10, threshold2=50)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    
    _contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
    _drawing = np.copy(frame)
    _contours_filtered = []

    # --- Filtering for circular contours ---
    for contour in _contours:
        convex_hull = cv2.convexHull(contour)
        area_hull = cv2.contourArea(convex_hull)
        if area_hull > 280:
            circumference_hull = cv2.arcLength(convex_hull, True)
            if circumference_hull <= 400:
                circularity_hull = (4 * np.pi * area_hull) / (circumference_hull ** 2)
                if circularity_hull > 0.85:
                    _contours_filtered.append(convex_hull)
    
    # --- Selecting the best circular contour (pupil) ---
    min_circularity = 1.5
    min_circularity_circle = None
    ellipses = []
    for c in _contours_filtered:
        try:
            ellipse = cv2.fitEllipse(c)
            ellipses.append(ellipse)
        except Exception:
            continue

    for i, ellipse in enumerate(ellipses):
        circumference = cv2.arcLength(_contours_filtered[i], True)
        circularity = circumference ** 2 / (4 * math.pi * cv2.contourArea(_contours_filtered[i]))
        if circularity < min_circularity:
            min_circularity = circularity
            min_circularity_circle = ellipse
            
    # --- Drawing the pupil ---
    if min_circularity_circle is not None:
        contour_points = cv2.ellipse2Poly(
            (int(min_circularity_circle[0][0]), int(min_circularity_circle[0][1])),
            (int(min_circularity_circle[1][0] / 2), int(min_circularity_circle[1][1] / 2)),
            int(min_circularity_circle[2]), 0, 360, 1
        )
        m = cv2.moments(contour_points)
        if m['m00'] != 0:
            center = (int(m['m10'] / m['m00']), int(m['m01'] / m['m00']))
            center_list[frame_no] = center  # Store pupil center
            cv2.circle(_drawing, center, 3, (0, 255, 0), -1)
        try:
            cv2.ellipse(_drawing, box=min_circularity_circle, color=(0, 255, 0), thickness=2)
        except:
            pass

    end = time.perf_counter()
    dtime = end - start
    fps = 1 / dtime if dtime > 0 else float('inf')
    cv2.putText(_drawing, str(round(fps,2)), (0, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.imshow('Pupil Detection', _drawing)
    frame_no += 1
    if cv2.waitKey(1) & 0xFF == 27:
        break

print("Extracted pupil centers per frame:")
print(center_list)
cap.release()
cv2.destroyAllWindows()

# ========= Time Series Imaging (TSI) =========

def normalize_ts(ts):
    """
    Normalize a time series to the range [-1, 1].
    """
    ts = np.array(ts)
    ts_min = np.min(ts)
    ts_max = np.max(ts) 
    ts_norm = (ts - ts_min) / (ts_max - ts_min) 
    ts_norm = ts_norm * 2 - 1
    return ts_norm

def compute_gaf(ts, method):
    """
    Compute the Gramian Angular Field (GAF) of a time series.
    
    Parameters:
      ts: 1D numpy array representing the time series (should be normalized to [-1, 1]).
      method: 'summation' for Gramian Angular Summation Field (GASF)
              'difference' for Gramian Angular Difference Field (GADF)
    
    Returns:
      A 2D numpy array representing the GAF image.
    """
    ts = np.array(ts)
    ts_norm = normalize_ts(ts) 
    phi = np.arccos(ts_norm)   
    phi = phi.reshape(-1, 1)   

    if method == 'summation': 
        gaf_image = np.cos(phi + phi.T)
    elif method == 'difference':
      
        gaf_image = np.sin(phi - phi.T)
    else:
        raise ValueError("Method should be either 'summation' or 'difference'")
    return gaf_image
 
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter  # Import Savitzky-Golay filter

import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from pyts.image import GramianAngularField

def compute_gaf(data, method):
    gaf_transformer = GramianAngularField(method=method)
    return gaf_transformer.fit_transform(data.reshape(1, -1))[0]

sorted_frames = sorted(center_list.keys())  
pupil_x_series = np.array([center_list[frame][0] for frame in sorted_frames])

q1 = np.percentile(pupil_x_series, 25)
q3 = np.percentile(pupil_x_series, 75)
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

outlier_mask = (pupil_x_series < lower_bound) | (pupil_x_series > upper_bound)

pupil_x_cleaned = np.copy(pupil_x_series).astype(float)
pupil_x_cleaned[outlier_mask] = np.nan

valid_indices = np.where(~outlier_mask)[0]
valid_values = pupil_x_cleaned[valid_indices]

interp_func = interp1d(valid_indices, valid_values, kind='cubic', fill_value='extrapolate')
pupil_x_cleaned[outlier_mask] = interp_func(np.where(outlier_mask)[0])

pupil_x_smoothed = savgol_filter(pupil_x_cleaned, window_length=51, polyorder=3)

gaf_summation = compute_gaf(pupil_x_smoothed, method='summation')
gaf_difference = compute_gaf(pupil_x_smoothed, method='difference')

plt.figure(figsize=(14, 8))

plt.subplot(2, 2, 1)
plt.scatter(sorted_frames, pupil_x_series, label="Original Data (with Outliers)", s=10, alpha=0.5)
plt.scatter(sorted_frames, pupil_x_cleaned, label="Cleaned Data (Interpolated)", s=10, color='red')
plt.plot(sorted_frames, pupil_x_smoothed, label="Smoothed Data (Savitzky-Golay)", color='green', linewidth=2)
plt.xlabel("Frame Number")
plt.ylabel("X Coordinate")
plt.title("Pupil X-coordinate Time Series (Outliers Interpolated & Smoothed)")
plt.legend()

plt.subplot(2, 2, 2)
plt.imshow(gaf_summation, cmap='rainbow', origin='lower')
plt.title("Gramian Angular Summation Field (GASF)")
plt.colorbar()

plt.subplot(2, 2, 4)
plt.imshow(gaf_difference, cmap='rainbow', origin='lower')
plt.title("Gramian Angular Difference Field (GADF)")
plt.colorbar()

plt.tight_layout()
plt.show()

def func(window_size, pupil_x_smoothed, sorted_frames, save_dir="GADF_GASF_Images"):
    os.makedirs(save_dir, exist_ok=True)
    num_partitions = len(sorted_frames) // window_size

    gaf_transformer = GramianAngularField(method="summation")
    gadf_transformer = GramianAngularField(method="difference")
    
    for i in range(num_partitions):
        start_idx = i * window_size
        end_idx = start_idx + window_size
        segment = pupil_x_smoothed[start_idx:end_idx]

        if len(segment) < window_size:
            continue

        segment = segment.reshape(1, -1)

        gasf_image = gaf_transformer.fit_transform(segment)[0]
        gadf_image = gadf_transformer.fit_transform(segment)[0]

        plt.imsave(os.path.join(save_dir, f"gasf_{i}.png"), gasf_image, cmap="rainbow")
        plt.imsave(os.path.join(save_dir, f"gadf_{i}.png"), gadf_image, cmap="rainbow")
    
    print(f"Saved {num_partitions * 2} images in '{save_dir}'.")

func(500, pupil_x_smoothed, sorted_frames)
