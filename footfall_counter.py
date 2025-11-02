# 🧠 Footfall Counter using Computer Vision
# 👩‍💻 Author: Anjali Gupta
# 🎯 Detects and counts number of people entering and exiting through a region

import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# ==============================
# 1️⃣ Load Model and Tracker
# ==============================
model = YOLO("yolov8n.pt")   # YOLOv8 Nano – fast & lightweight
tracker = DeepSort(max_age=30)  # To maintain tracking IDs

# ==============================
# 2️⃣ Load Video
# ==============================
video_path = "people_footage.mp4"
  # <-- your video file here
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

# Get video dimensions
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Save processed output video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, fps, (width, height))

# ==============================
# 3️⃣ Counting Line Setup
# ==============================
# If your video is vertical, use a horizontal line (adjust Y value)
# For horizontal videos, you may use a vertical line (adjust X value)
vertical_video = True   # change to False if your video is landscape

if vertical_video:
    line_y = int(height / 2)  # horizontal line
else:
    line_x = int(width / 2)   # vertical line

entries, exits = 0, 0
track_history = {}

# ==============================
# 4️⃣ Process Video Frame-by-Frame
# ==============================
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO detection
    results = model(frame, verbose=False)

    detections = []
    for r in results[0].boxes:
        cls_id = int(r.cls[0])
        conf = float(r.conf[0])
        if cls_id == 0 and conf > 0.5:  # Class 0 = Person
            x1, y1, x2, y2 = map(int, r.xyxy[0])
            detections.append(([x1, y1, x2 - x1, y2 - y1], conf, 'person'))

    # Update tracker
    tracks = tracker.update_tracks(detections, frame=frame)

    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        l, t, r, b = track.to_ltrb()
        cx, cy = int((l + r) / 2), int((t + b) / 2)

        # Draw bounding boxes
        cv2.rectangle(frame, (int(l), int(t)), (int(r), int(b)), (0, 255, 0), 2)
        cv2.putText(frame, f'ID {track_id}', (int(l), int(t) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Store previous positions
        if track_id not in track_history:
            track_history[track_id] = []
        track_history[track_id].append((cx, cy))

        # Count crossing
        if len(track_history[track_id]) >= 2:
            prev_cx, prev_cy = track_history[track_id][-2]
            curr_cx, curr_cy = track_history[track_id][-1]

            if vertical_video:
                # Horizontal line crossing
                if prev_cy < line_y <= curr_cy:
                    entries += 1
                elif prev_cy > line_y >= curr_cy:
                    exits += 1
            else:
                # Vertical line crossing
                if prev_cx < line_x <= curr_cx:
                    entries += 1
                elif prev_cx > line_x >= curr_cx:
                    exits += 1

    # ==============================
    # 5️⃣ Display Counting Line & Info
    # ==============================
    if vertical_video:
        cv2.line(frame, (0, line_y), (width, line_y), (0, 0, 255), 2)
    else:
        cv2.line(frame, (line_x, 0), (line_x, height), (0, 0, 255), 2)

    cv2.putText(frame, f'Entries: {entries}  Exits: {exits}',
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)

    out.write(frame)
    cv2.imshow("Footfall Counter", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ==============================
# 6️⃣ Cleanup
# ==============================
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"\n✅ Processing complete!")
print(f"Total Entries: {entries}")
print(f"Total Exits: {exits}")
