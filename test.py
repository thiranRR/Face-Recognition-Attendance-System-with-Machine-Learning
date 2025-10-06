from sklearn.neighbors import KNeighborsClassifier
import cv2
import pickle
import numpy as np
import os
import csv
import time
from datetime import datetime

try:
    from pyttsx3 import Dispatch
    def speak(text):
        try:
            s = Dispatch("SAPI.SpVoice")
            s.Speak(text)
        except Exception:
            pass
except Exception:
    def speak(text):
        # fallback: no-op
        pass

# Paths and constants
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
BACKGROUND_PATH = "background.png"
ATT_DIR = "Attendance"
os.makedirs(ATT_DIR, exist_ok=True)

# Load cascade
facedetect = cv2.CascadeClassifier(CASCADE_PATH)
if facedetect.empty():
    raise SystemExit(f"ERROR: could not load cascade at {CASCADE_PATH}")

# Load training data
with open('data/names.pkl', 'rb') as w:
    LABELS = pickle.load(w)
with open('data/faces_data.pkl', 'rb') as f:
    FACES = pickle.load(f)

FACES = np.array(FACES)
LABELS = np.array(LABELS)
print('Shape of Faces matrix -->', FACES.shape)

# Train KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(FACES, LABELS)

# Load background (optional)
image_back = cv2.imread(BACKGROUND_PATH)
if image_back is None:
    print(f"Note: '{BACKGROUND_PATH}' not found — using camera frames directly.")
else:
    # define overlay region and allow resizing if needed
    OVERLAY_Y, OVERLAY_X = 162, 55
    OVERLAY_H, OVERLAY_W = 480, 640

video = cv2.VideoCapture(0)
if not video.isOpened():
    raise SystemExit("ERROR: could not open webcam")

COL_NAMES = ['NAME', 'DATE', 'TIME']
recorded_today = set()   # to avoid duplicates in a session
last_detected = []       # store last detected names and timestamps

print("Press 'o' to record attendance for current detected face(s). Press 'q' to quit.")

try:
    while True:
        ret, frame = video.read()
        if not ret or frame is None:
            # small sleep to avoid busy-loop if camera fails
            time.sleep(0.05)
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = facedetect.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        # store detections for the most recent frame
        last_detected = []

        for (x, y, w, h) in faces:
            # ensure crop is valid
            crop = frame[y:y+h, x:x+w]
            if crop.size == 0:
                continue
            # resize and predict
            try:
                resized = cv2.resize(crop, (50, 50)).flatten().reshape(1, -1)
                pred = knn.predict(resized)[0]
            except Exception:
                pred = "Unknown"

            # draw rectangle and label (not filled)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 2)
            cv2.putText(frame, str(pred), (x, y-10), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 255, 255), 1)

            ts = time.time()
            date_str = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
            time_str = datetime.fromtimestamp(ts).strftime("%H:%M:%S")
            last_detected.append((str(pred), date_str, time_str))

        # Prepare display (overlay background if available)
        if image_back is not None:
            display = image_back.copy()
            h_frame, w_frame = frame.shape[:2]
            # if frame size equals overlay size, paste directly else resize
            if (h_frame, w_frame) == (OVERLAY_H, OVERLAY_W):
                display[OVERLAY_Y:OVERLAY_Y + OVERLAY_H, OVERLAY_X:OVERLAY_X + OVERLAY_W] = frame
            else:
                small = cv2.resize(frame, (OVERLAY_W, OVERLAY_H))
                display[OVERLAY_Y:OVERLAY_Y + OVERLAY_H, OVERLAY_X:OVERLAY_X + OVERLAY_W] = small
        else:
            display = frame

        cv2.imshow("Frame", display)
        k = cv2.waitKey(1) & 0xFF

        if k == ord('o'):
            if not last_detected:
                print("No faces detected right now — nothing to record.")
                speak("No faces detected.")
            else:
                # write attendance for each detected face (if not already recorded in session)
                saved_any = False
                for name_pred, date_str, time_str in last_detected:
                    if name_pred in recorded_today:
                        print(f"{name_pred} already recorded in this session — skipping.")
                        continue
                    csv_path = os.path.join(ATT_DIR, f"Attendance_{date_str}.csv")
                    file_exists = os.path.isfile(csv_path)
                    # open and append
                    with open(csv_path, 'a', newline='') as csvfile:
                        writer = csv.writer(csvfile)
                        if not file_exists:
                            writer.writerow(COL_NAMES)
                        writer.writerow([name_pred, date_str, time_str])
                    recorded_today.add(name_pred)
                    print(f"Recorded attendance: {name_pred} at {time_str}")
                    saved_any = True
                if saved_any:
                    speak("Attendance taken")
                else:
                    speak("No new attendance to record")

        if k == ord('q'):
            break

except KeyboardInterrupt:
    print("Interrupted by user (KeyboardInterrupt). Exiting gracefully...")

finally:
    video.release()
    cv2.destroyAllWindows()
    print("Cleaned up and exiting.")
