import customtkinter as ctk
import cv2
import PIL.Image, PIL.ImageTk
import os
import pickle
import numpy as np
import csv
import threading
import dlib
from scipy.spatial import distance as dist
from datetime import datetime
from sklearn.neighbors import KNeighborsClassifier
import pyttsx3

# --- CONFIGURATION ---
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

def calculate_ear(eye):
    # Calculate the vertical distances (between eyelids)
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    # Calculate the horizontal distance (eye width)
    C = dist.euclidean(eye[0], eye[3])
    # Compute the Eye Aspect Ratio
    ear = (A + B) / (2.0 * C)
    return ear

class FaceAttendanceApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        # 1. Main Window Setup
        self.title("Face Recognition Attendance System")
        self.geometry("1100x700")

        # 2. Setup Variables
        self.cap = None
        self.running = False
        self.capture_active = False 
        self.capture_count = 0
        self.capture_limit = 100
        self.faces_buffer = []
        self.name_buffer = ""
        self.current_detected_name = None
        
        # --- BLINK DETECTION VARIABLES ---
        self.blink_counter = 0
        self.blink_verified = False
        self.EYE_AR_THRESH = 0.25  # If EAR < 0.25, eye is closed
        self.EYE_AR_CONSEC_FRAMES = 3 # How many frames eye must be closed to count as blink
        
        # Paths
        self.data_dir = "data"
        self.attendance_dir = "Attendance"
        self.faces_path = os.path.join(self.data_dir, "faces_data.pkl")
        self.names_path = os.path.join(self.data_dir, "names.pkl")
        self.predictor_path = os.path.join(self.data_dir, "shape_predictor_68_face_landmarks.dat") # NEW
        
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.attendance_dir, exist_ok=True)

        # Load Detectors
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Dlib Setup for Landmarks
        if os.path.exists(self.predictor_path):
            self.dlib_predictor = dlib.shape_predictor(self.predictor_path)
        else:
            print("WARNING: shape_predictor_68_face_landmarks.dat not found in data folder!")
            self.dlib_predictor = None

        # Initialize Voice Engine
        self.tts_engine = pyttsx3.init()

        # 3. Create the UI
        self.create_layout()
        
        # 4. Load Data & Train Model
        self.load_and_train()

    def create_layout(self):
        self.tab_view = ctk.CTkTabview(self, command=self.on_tab_change)
        self.tab_view.pack(fill="both", expand=True, padx=20, pady=20)

        self.tab_dashboard = self.tab_view.add("Dashboard")
        self.tab_admin = self.tab_view.add("Admin / Add Face")
        self.tab_reports = self.tab_view.add("Reports")

        # --- DASHBOARD TAB ---
        self.tab_dashboard.grid_columnconfigure(0, weight=3)
        self.tab_dashboard.grid_columnconfigure(1, weight=1)

        self.video_frame = ctk.CTkFrame(self.tab_dashboard)
        self.video_frame.grid(row=0, column=0, rowspan=4, padx=10, pady=10, sticky="nsew")
        
        self.dash_video_label = ctk.CTkLabel(self.video_frame, text="")
        self.dash_video_label.pack(fill="both", expand=True)

        self.btn_start = ctk.CTkButton(self.tab_dashboard, text="Start Camera", command=self.start_camera, height=40, fg_color="green")
        self.btn_start.grid(row=0, column=1, padx=20, pady=10, sticky="ew")

        self.btn_take = ctk.CTkButton(self.tab_dashboard, text="Mark Attendance", command=self.mark_attendance, height=40)
        self.btn_take.grid(row=1, column=1, padx=20, pady=10, sticky="ew")

        # Visual indicator for liveness
        self.lbl_liveness = ctk.CTkLabel(self.tab_dashboard, text="Liveness: Waiting...", font=("Arial", 14), text_color="orange")
        self.lbl_liveness.grid(row=2, column=1, padx=20, pady=10)

        self.btn_stop = ctk.CTkButton(self.tab_dashboard, text="Stop Camera", command=self.stop_camera, height=40, fg_color="red")
        self.btn_stop.grid(row=3, column=1, padx=20, pady=10, sticky="ew")

        self.log_box = ctk.CTkTextbox(self.tab_dashboard, height=200)
        self.log_box.grid(row=4, column=1, padx=20, pady=10, sticky="nsew")
        self.log("Welcome! Please start the camera.")

        # --- ADMIN TAB ---
        self.tab_admin.grid_columnconfigure(0, weight=1)
        self.tab_admin.grid_columnconfigure(1, weight=1)

        self.admin_video_frame = ctk.CTkFrame(self.tab_admin)
        self.admin_video_frame.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")
        
        self.admin_video_label = ctk.CTkLabel(self.admin_video_frame, text="Camera Preview")
        self.admin_video_label.pack(fill="both", expand=True)

        self.admin_controls = ctk.CTkFrame(self.tab_admin)
        self.admin_controls.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")

        self.entry_name = ctk.CTkEntry(self.admin_controls, placeholder_text="Enter New Student Name", width=250)
        self.entry_name.pack(pady=(50, 20))

        self.btn_train = ctk.CTkButton(self.admin_controls, text="Start Capturing Face", command=self.start_training, height=50)
        self.btn_train.pack(pady=10)

        self.lbl_progress = ctk.CTkLabel(self.admin_controls, text="Status: Idle", font=("Arial", 14))
        self.lbl_progress.pack(pady=10)

        # --- REPORTS TAB ---
        self.lbl_reports_title = ctk.CTkLabel(self.tab_reports, text="Today's Attendance Records", font=("Arial", 20, "bold"))
        self.lbl_reports_title.pack(pady=10)
        
        self.reports_box = ctk.CTkTextbox(self.tab_reports, width=800, height=500, font=("Consolas", 14))
        self.reports_box.pack(pady=10)

    # --- LOGIC HANDLERS ---
    def on_tab_change(self):
        current = self.tab_view.get()
        if current == "Admin / Add Face":
            if not self.running:
                self.start_camera()
        if current == "Reports":
            self.refresh_reports()

    def log(self, message):
        self.log_box.insert("end", f"{message}\n")
        self.log_box.see("end")

    def speak(self, text):
        def _speak():
            try:
                engine = pyttsx3.init()
                engine.say(text)
                engine.runAndWait()
            except Exception as e:
                print(f"TTS Error: {e}")
        threading.Thread(target=_speak, daemon=True).start()

    def load_and_train(self):
        try:
            with open(self.names_path, 'rb') as f:
                self.LABELS = pickle.load(f)
            with open(self.faces_path, 'rb') as f:
                self.FACES = pickle.load(f)
            if len(self.FACES) > 0:
                self.knn = KNeighborsClassifier(n_neighbors=5)
                self.knn.fit(self.FACES, self.LABELS)
                self.log(f"Model trained with {len(self.LABELS)} records.")
            else:
                self.knn = None
        except FileNotFoundError:
            self.knn = None

    # --- CAMERA LOOP ---
    def start_camera(self):
        if not self.running:
            self.cap = cv2.VideoCapture(0)
            self.running = True
            self.update_frame()

    def stop_camera(self):
        self.running = False
        if self.cap:
            self.cap.release()
        self.dash_video_label.configure(image=None)
        self.admin_video_label.configure(image=None)

    def update_frame(self):
        if not self.running:
            return

        ret, frame = self.cap.read()
        if not ret:
            self.stop_camera()
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        
        current_tab = self.tab_view.get()
        self.current_detected_name = None 

        # --- ADMIN TAB LOGIC ---
        if current_tab == "Admin / Add Face":
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                if self.capture_active:
                    self.capture_face_data(frame, x, y, w, h)
            self.display_image(frame, self.admin_video_label)

        # --- DASHBOARD LOGIC (With Liveness) ---
        else: 
            if len(faces) == 0:
                # Reset if no face found
                self.blink_verified = False
                self.lbl_liveness.configure(text="Liveness: No Face", text_color="gray")

            for (x, y, w, h) in faces:
                # 1. Recognition Logic
                name = "Unknown"
                color = (0, 0, 255) # Red default
                if self.knn:
                    try:
                        crop_img = frame[y:y+h, x:x+w]
                        resized = cv2.resize(crop_img, (50, 50)).flatten().reshape(1, -1)
                        name = self.knn.predict(resized)[0]
                    except Exception:
                        pass
                
                # 2. Liveness Logic (Dlib)
                if self.dlib_predictor:
                    # Dlib needs a rectangle object, not x,y,w,h
                    dlib_rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
                    shape = self.dlib_predictor(gray, dlib_rect)
                    shape = np.array([[p.x, p.y] for p in shape.parts()])

                    # Extract Eye Coordinates (indices for 68-point model)
                    leftEye = shape[42:48]
                    rightEye = shape[36:42]
                    leftEAR = calculate_ear(leftEye)
                    rightEAR = calculate_ear(rightEye)
                    ear = (leftEAR + rightEAR) / 2.0

                    # Check for Blink
                    if ear < self.EYE_AR_THRESH:
                        self.blink_counter += 1
                    else:
                        if self.blink_counter >= self.EYE_AR_CONSEC_FRAMES:
                            self.blink_verified = True
                        self.blink_counter = 0

                    # Draw Eye landmarks (Optional)
                    for (ex, ey) in leftEye: cv2.circle(frame, (ex, ey), 1, (0, 255, 255), -1)
                    for (ex, ey) in rightEye: cv2.circle(frame, (ex, ey), 1, (0, 255, 255), -1)

                # 3. Status Update based on Blink
                if self.blink_verified:
                    color = (0, 255, 0) # Green
                    self.lbl_liveness.configure(text="Liveness: Verified (Real)", text_color="green")
                    self.current_detected_name = name # Only allow marking if verified
                else:
                    color = (0, 165, 255) # Orange
                    self.lbl_liveness.configure(text="Liveness: Please Blink...", text_color="orange")
                    # We do NOT set self.current_detected_name here, effectively blocking attendance

                # Draw Face Box
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
            self.display_image(frame, self.dash_video_label)

        self.after(10, self.update_frame)

    def display_image(self, frame, label_widget):
        rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = PIL.Image.fromarray(rgb_img)
        img_ctk = ctk.CTkImage(light_image=img_pil, dark_image=img_pil, size=(640, 480))
        label_widget.configure(image=img_ctk)
        label_widget.image = img_ctk 

    # --- ADD FACE LOGIC ---
    def start_training(self):
        name = self.entry_name.get().strip()
        if not name:
            self.lbl_progress.configure(text="Error: Enter a name!", text_color="red")
            return
        if not self.running: self.start_camera()
        self.name_buffer = name
        self.faces_buffer = []
        self.capture_count = 0
        self.capture_active = True
        self.lbl_progress.configure(text=f"Capturing...", text_color="yellow")

    def capture_face_data(self, frame, x, y, w, h):
        if self.capture_count < self.capture_limit:
            crop_img = frame[y:y+h, x:x+w]
            resized_img = cv2.resize(crop_img, (50, 50)).flatten()
            self.faces_buffer.append(resized_img)
            self.capture_count += 1
            self.lbl_progress.configure(text=f"Captured: {self.capture_count}/{self.capture_limit}")
        else:
            self.save_new_face()

    def save_new_face(self):
        self.capture_active = False
        self.lbl_progress.configure(text="Saving data...", text_color="blue")
        if len(self.FACES) == 0:
            self.FACES = np.array(self.faces_buffer)
            self.LABELS = np.array([self.name_buffer] * self.capture_limit)
        else:
            self.FACES = np.append(self.FACES, self.faces_buffer, axis=0)
            self.LABELS = np.append(self.LABELS, [self.name_buffer] * self.capture_limit)
        with open(self.names_path, 'wb') as f: pickle.dump(self.LABELS, f)
        with open(self.faces_path, 'wb') as f: pickle.dump(self.FACES, f)
        self.knn = KNeighborsClassifier(n_neighbors=5)
        self.knn.fit(self.FACES, self.LABELS)
        self.lbl_progress.configure(text=f"Success! {self.name_buffer} added.", text_color="green")
        self.entry_name.delete(0, 'end')

    # --- ATTENDANCE MARKING ---
    def mark_attendance(self):
        if not self.blink_verified:
            self.log("⚠️ Liveness Check Failed: Please blink at the camera.")
            self.speak("Please blink to verify you are human")
            return

        name = self.current_detected_name
        if not name or name == "Unknown" or name == "No Data":
            self.log("Cannot mark attendance: Unknown user.")
            self.speak("Face not recognized")
            return

        date_str = datetime.now().strftime("%d-%m-%Y")
        time_str = datetime.now().strftime("%H:%M:%S")
        csv_file = os.path.join(self.attendance_dir, f"Attendance_{date_str}.csv")
        
        already_present = False
        if os.path.exists(csv_file):
            with open(csv_file, "r") as f:
                reader = csv.reader(f)
                for row in reader:
                    if row and row[0] == name:
                        already_present = True
                        break
        
        if already_present:
            self.log(f"{name} is already marked present.")
            self.speak(f"{name} is already present")
        else:
            file_exists = os.path.exists(csv_file)
            with open(csv_file, "a", newline="") as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(["NAME", "TIME"])
                writer.writerow([name, time_str])
            self.log(f"✅ Attendance Marked: {name}")
            self.speak(f"Attendance marked for {name}")
            self.refresh_reports()
            
            # Reset Blink for next user
            self.blink_verified = False
            self.lbl_liveness.configure(text="Liveness: Waiting...", text_color="orange")

    def refresh_reports(self):
        date_str = datetime.now().strftime("%d-%m-%Y")
        csv_file = os.path.join(self.attendance_dir, f"Attendance_{date_str}.csv")
        self.reports_box.configure(state="normal")
        self.reports_box.delete("0.0", "end")
        if os.path.exists(csv_file):
            with open(csv_file, "r") as f: self.reports_box.insert("0.0", f.read())
        else:
            self.reports_box.insert("0.0", "No records found for today.")
        self.reports_box.configure(state="disabled")

if __name__ == "__main__":
    app = FaceAttendanceApp()
    app.mainloop()