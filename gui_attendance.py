# gui_attendance.py
import cv2
import pickle
import numpy as np
import os
import csv
import time
import threading
from datetime import datetime
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import pyttsx3
from sklearn.neighbors import KNeighborsClassifier
import subprocess

# ---------- Configuration ----------
CASCADE = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
DATA_FACES = "data/faces_data.pkl"
DATA_NAMES = "data/names.pkl"
ATT_DIR = "Attendance"
os.makedirs(ATT_DIR, exist_ok=True)
IMG_SZ = (50, 50)
# -----------------------------------

# Load cascade
face_cascade = cv2.CascadeClassifier(CASCADE)
if face_cascade.empty():
    raise SystemExit("Cannot load Haar cascade.")

# Load trained data (expect train.py already created these)
with open(DATA_FACES, "rb") as f:
    FACES = pickle.load(f)
with open(DATA_NAMES, "rb") as f:
    LABELS = pickle.load(f)

FACES = np.array(FACES)
LABELS = np.array(LABELS)
print("Loaded training data:", FACES.shape, LABELS.shape)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(FACES, LABELS)

# TTS engine (pyttsx3) - non-blocking
tts = pyttsx3.init()
def speak_async(text):
    def _run():
        try:
            tts.say(text)
            tts.runAndWait()
        except Exception as e:
            print("TTS error:", e)
    threading.Thread(target=_run, daemon=True).start()

# Camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise SystemExit("Cannot open camera")

# Global state
last_detected = []         # [(name, date, time), ...] for current frame
recorded_today = set()     # to avoid duplicates in a session
running = True

# Tkinter GUI
root = tk.Tk()
root.title("Face Recognition Attendance")
root.geometry("1000x640")
root.resizable(False, False)

# Left: video display (simulate background)
video_frame = tk.Frame(root, bg="#222")
video_frame.place(x=10, y=10, width=700, height=620)
video_label = tk.Label(video_frame)
video_label.pack(expand=True)

# Right: Controls panel (acts as interface / background)
control_frame = tk.Frame(root, bg="#f0f0f0", bd=2, relief="ridge")
control_frame.place(x=720, y=10, width=270, height=620)

title = tk.Label(control_frame, text="Attendance Panel", font=("Helvetica", 14, "bold"), bg="#f0f0f0")
title.pack(pady=10)

btn_take = ttk.Button(control_frame, text="Take Attendance")
btn_show = ttk.Button(control_frame, text="Show Today's Records")
btn_clear = ttk.Button(control_frame, text="Clear Session Records")
btn_quit = ttk.Button(control_frame, text="Quit")

btn_take.pack(fill="x", padx=15, pady=6)
btn_show.pack(fill="x", padx=15, pady=6)
btn_clear.pack(fill="x", padx=15, pady=6)
btn_quit.pack(fill="x", padx=15, pady=6)

log_label = tk.Label(control_frame, text="Log", bg="#f0f0f0")
log_label.pack(pady=(12,0))
log_text = tk.Text(control_frame, height=18, width=30, state="disabled")
log_text.pack(padx=8, pady=6)

def log(msg):
    log_text.configure(state="normal")
    log_text.insert("end", f"[{datetime.now().strftime('%H:%M:%S')}] {msg}\n")
    log_text.see("end")
    log_text.configure(state="disabled")

# Camera processing thread
def camera_loop():
    global last_detected, running
    while running:
        ret, frame = cap.read()
        if not ret or frame is None:
            time.sleep(0.02)
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        detections = []
        for (x, y, w, h) in faces:
            crop = frame[y:y+h, x:x+w]
            if crop.size == 0:
                continue
            try:
                resized = cv2.resize(crop, IMG_SZ).flatten().reshape(1, -1)
                name = knn.predict(resized)[0]
            except Exception:
                name = "Unknown"

            # draw box and label
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
            cv2.putText(frame, str(name), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

            ts = time.time()
            date_str = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
            time_str = datetime.fromtimestamp(ts).strftime("%H:%M:%S")
            detections.append((str(name), date_str, time_str))

        last_detected = detections

        # convert and show in Tk label
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb).resize((700,620))
        imgtk = ImageTk.PhotoImage(img)

        def update_img():
            video_label.imgtk = imgtk
            video_label.configure(image=imgtk)
        root.after(0, update_img)

        time.sleep(0.02)

# Start camera thread
threading.Thread(target=camera_loop, daemon=True).start()

# Attendance functions
def take_attendance():
    global last_detected, recorded_today
    if not last_detected:
        log("No faces detected to record.")
        speak_async("No faces detected.")
        return
    saved=False
    for name, date_str, time_str in last_detected:
        if name in recorded_today:
            log(f"{name} already recorded this session.")
            continue
        csv_path = os.path.join(ATT_DIR, f"Attendance_{date_str}.csv")
        file_exists = os.path.isfile(csv_path)
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["NAME","DATE","TIME"])
            writer.writerow([name, date_str, time_str])
        recorded_today.add(name)
        log(f"Recorded: {name} at {time_str}")
        saved=True
    if saved:
        speak_async("Attendance recorded")
    else:
        speak_async("No new attendance")

def show_today():
    d = datetime.now().strftime("%d-%m-%Y")
    path = os.path.join(ATT_DIR, f"Attendance_{d}.csv")
    if not os.path.isfile(path):
        messagebox.showinfo("Records", f"No records for {d}")
        return
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    win = tk.Toplevel(root)
    win.title(f"Attendance {d}")
    t = tk.Text(win, width=50, height=20)
    t.pack(fill="both", expand=True)
    for r in rows:
        t.insert("end", ",".join(r) + "\n")
    t.configure(state="disabled")



def clear_session():
    global recorded_today
    recorded_today.clear()
    log("Session records cleared.")
    speak_async("Session records cleared.")

def quit_app():
    global running
    if messagebox.askokcancel("Quit","Exit the app?"):
        running = False
        cap.release()
        root.destroy()

# Wire buttons
btn_take.config(command=take_attendance)
btn_show.config(command=show_today)
btn_clear.config(command=clear_session)
btn_quit.config(command=quit_app)

root.protocol("WM_DELETE_WINDOW", quit_app)
root.mainloop()
running = False

subprocess.Popen(["streamlit", "run", "app.py"])
