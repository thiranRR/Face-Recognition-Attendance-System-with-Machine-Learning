import cv2
import pickle
import numpy as np
import os
import csv
import time
import threading
from datetime import datetime
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
from PIL import Image, ImageTk
import pyttsx3
from sklearn.neighbors import KNeighborsClassifier
import subprocess


CASCADE = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
DATA_FACES = "data/faces_data.pkl"
DATA_NAMES = "data/names.pkl"
ATT_DIR = "Attendance"
os.makedirs(ATT_DIR, exist_ok=True)
IMG_SZ = (50, 50)


face_cascade = cv2.CascadeClassifier(CASCADE)
if face_cascade.empty():
    raise SystemExit("Cannot load Haar cascade.")


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
btn_add  = ttk.Button(control_frame, text="Add Face")            # <-- new button
btn_show = ttk.Button(control_frame, text="Show Today's Records")
btn_clear = ttk.Button(control_frame, text="Clear Session Records")
btn_quit = ttk.Button(control_frame, text="Quit")

btn_take.pack(fill="x", padx=15, pady=6)
btn_add.pack(fill="x", padx=15, pady=6)                           # <-- pack new button
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
def _time_greeting():
    h = datetime.now().hour
    if 5 <= h < 12:
        return "Good morning"
    if 12 <= h < 17:
        return "Good afternoon"
    if 17 <= h < 21:
        return "Good evening"
    return "Good night"

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
        # Greeting voice for this recorded person
        greeting = f"{_time_greeting()} {name}"
        speak_async(greeting)

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

# --- Add-face helper functions ---
def load_training_data():
    if os.path.isfile(DATA_FACES) and os.path.isfile(DATA_NAMES):
        with open(DATA_FACES, "rb") as f:
            faces = pickle.load(f)
        with open(DATA_NAMES, "rb") as f:
            names = pickle.load(f)
        return np.array(faces), np.array(names)
    # empty fallback: shape compatible with IMG_SZ and 3 channels
    return np.empty((0, IMG_SZ[0]*IMG_SZ[1]*3)), np.array([])

def save_training_data(faces_arr, names_arr):
    with open(DATA_FACES, "wb") as f:
        pickle.dump(list(faces_arr), f)
    with open(DATA_NAMES, "wb") as f:
        pickle.dump(list(names_arr), f)

def retrain_knn(n_neighbors=5):
    global knn, FACES, LABELS
    faces_arr, names_arr = load_training_data()
    if names_arr.size == 0:
        return
    FACES = np.array(faces_arr)
    LABELS = np.array(names_arr)
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(FACES, LABELS)
    log("Classifier retrained with new data.")

def add_face_via_camera(samples=8, camera_index=0):
    """
    Open camera, collect `samples` face crops for a given name, append to data,
    save pickles and retrain classifier.
    """
    name = simpledialog.askstring("Add Face", "Enter the person's name:", parent=root)
    if not name:
        return

    # use a separate VideoCapture to avoid interfering with main preview
    cap2 = cv2.VideoCapture(camera_index)
    if not cap2.isOpened():
        messagebox.showerror("Camera Error", "Cannot open camera for capturing faces.")
        return

    collected = []
    last_capture = 0.0
    interval = 0.6

    messagebox.showinfo("Instructions",
                        f"A window will open. {samples} face samples will be captured automatically. Press 'q' to cancel.",
                        parent=root)
    try:
        while len(collected) < samples:
            ret, frame = cap2.read()
            if not ret or frame is None:
                time.sleep(0.02)
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
            if faces is not None and len(faces) > 0:
                x, y, w, h = sorted(faces, key=lambda r: r[2]*r[3], reverse=True)[0]
                cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
            cv2.putText(frame, f"Collected: {len(collected)}/{samples}", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
            cv2.imshow("Add Face - Press 'q' to cancel", frame)
            key = cv2.waitKey(1) & 0xFF

            now = time.time()
            if faces is not None and len(faces) > 0 and (now - last_capture) > interval:
                try:
                    crop = frame[y:y+h, x:x+w]
                    if crop.size == 0:
                        continue
                    resized = cv2.resize(crop, IMG_SZ)
                    flat = resized.flatten()
                    collected.append(flat)
                    last_capture = now
                except Exception:
                    pass

            if key == ord('q'):
                break
    finally:
        cap2.release()
        cv2.destroyWindow("Add Face - Press 'q' to cancel")

    if len(collected) == 0:
        messagebox.showinfo("Add Face", "No face samples captured.")
        return

    faces_arr, names_arr = load_training_data()
    if faces_arr.size == 0:
        faces_arr = np.array(collected)
        names_arr = np.array([name] * len(collected))
    else:
        faces_arr = np.vstack([faces_arr, np.array(collected)])
        names_arr = np.concatenate([names_arr, np.array([name] * len(collected))])

    save_training_data(faces_arr, names_arr)
    retrain_knn()
    messagebox.showinfo("Add Face", f"Saved {len(collected)} samples for '{name}'.")

# --- Modern styling (ttk) ---
style = ttk.Style()
try:
    style.theme_use('clam')   # modern-ish, available cross-platform
except Exception:
    pass
style.configure('Accent.TButton',
                font=('Segoe UI', 11, 'bold'),
                padding=8,
                foreground='#ffffff',
                background='#2b8cff')
style.map('Accent.TButton',
          background=[('active', '#1a6fe0'), ('pressed', '#155ecb')])

style.configure('Ghost.TButton',
                font=('Segoe UI', 10),
                padding=6,
                foreground='#333333',
                background='#f0f0f0')

# Wire the new button
btn_add.config(command=lambda: threading.Thread(target=add_face_via_camera, daemon=True).start())

# Wire buttons
btn_take.config(command=take_attendance)
btn_show.config(command=show_today)
btn_clear.config(command=clear_session)
btn_quit.config(command=quit_app)

# After buttons were created, apply styles for a modern look
btn_take.configure(style='Accent.TButton')
btn_add.configure(style='Accent.TButton')
btn_show.configure(style='Ghost.TButton')
btn_clear.configure(style='Ghost.TButton')
btn_quit.configure(style='Ghost.TButton')

root.protocol("WM_DELETE_WINDOW", quit_app)
root.mainloop()
running = False

subprocess.Popen(["streamlit", "run", "app.py"])
