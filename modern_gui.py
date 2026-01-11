import customtkinter as ctk
import cv2
import PIL.Image, PIL.ImageTk
import threading
import numpy as np
import face_recognition
import dlib
from scipy.spatial import distance as dist
import pyttsx3
import os
import time

# Import your helper modules
import database
import notifications

# --- CONFIGURATION ---
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

# --- LIVENESS HELPER ---
def calculate_ear(eye):
    # Vertical distances
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    # Horizontal distance
    C = dist.euclidean(eye[0], eye[3])
    # Eye Aspect Ratio
    ear = (A + B) / (2.0 * C)
    return ear

class FaceAttendanceApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Smart Attendance System (optimized)")
        self.geometry("1100x750")

        # --- SYSTEM VARIABLES ---
        self.cap = None
        self.running = False
        self.frame_lock = threading.Lock()
        self.latest_frame = None
        
        # Database Data
        self.known_face_encodings = []
        self.known_face_names = []
        self.known_face_ids = []
        
        # Liveness Variables
        self.blink_counter = 0
        self.liveness_verified = False
        self.EYE_AR_THRESH = 0.22  # Lowered slightly for better detection
        self.EYE_AR_CONSEC_FRAMES = 2
        
        # State
        self.current_name = "Unknown"
        self.current_id = None
        
        # Paths
        self.predictor_path = "data/shape_predictor_68_face_landmarks.dat"
        self.cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'

        # --- INITIALIZE MODELS ---
        # 1. Fast Face Detector (Haar) - Much faster than HOG
        self.face_cascade = cv2.CascadeClassifier(self.cascade_path)
        
        # 2. Dlib for Liveness
        if os.path.exists(self.predictor_path):
            self.predictor = dlib.shape_predictor(self.predictor_path)
        else:
            print("CRITICAL WARNING: shape_predictor_68_face_landmarks.dat not found!")
            self.predictor = None

        # 3. Voice Engine
        self.tts_engine = pyttsx3.init()

        # Load UI and Data
        self.create_layout()
        self.reload_db_data()

    def create_layout(self):
        self.tab_view = ctk.CTkTabview(self, command=self.on_tab_change)
        self.tab_view.pack(fill="both", expand=True, padx=20, pady=20)

        self.tab_dash = self.tab_view.add("Dashboard")
        self.tab_admin = self.tab_view.add("Admin")
        self.tab_reports = self.tab_view.add("Reports")

        # --- DASHBOARD ---
        self.tab_dash.grid_columnconfigure(0, weight=3)
        self.tab_dash.grid_columnconfigure(1, weight=1)

        self.video_frame = ctk.CTkFrame(self.tab_dash)
        self.video_frame.grid(row=0, column=0, rowspan=5, padx=10, pady=10, sticky="nsew")
        self.lbl_video = ctk.CTkLabel(self.video_frame, text="")
        self.lbl_video.pack(fill="both", expand=True)

        # Controls
        self.btn_start = ctk.CTkButton(self.tab_dash, text="Start Camera", command=self.start_camera_thread, fg_color="green")
        self.btn_start.grid(row=0, column=1, padx=20, pady=10, sticky="ew")

        self.btn_mark = ctk.CTkButton(self.tab_dash, text="Mark Attendance", command=self.mark_attendance_action)
        self.btn_mark.grid(row=1, column=1, padx=20, pady=10, sticky="ew")

        self.lbl_liveness = ctk.CTkLabel(self.tab_dash, text="Status: Waiting...", font=("Arial", 16, "bold"), text_color="gray")
        self.lbl_liveness.grid(row=2, column=1, padx=20, pady=10)
        
        self.var_email = ctk.BooleanVar(value=False)
        self.switch_email = ctk.CTkSwitch(self.tab_dash, text="Send Email Alerts", variable=self.var_email)
        self.switch_email.grid(row=3, column=1, padx=20, pady=10)

        self.log_box = ctk.CTkTextbox(self.tab_dash, height=200)
        self.log_box.grid(row=4, column=1, padx=20, pady=10, sticky="nsew")

        # --- ADMIN TAB ---
        self.tab_admin.grid_columnconfigure(0, weight=1)
        self.tab_admin.grid_columnconfigure(1, weight=1)
        
        self.admin_video = ctk.CTkLabel(self.tab_admin, text="Camera Preview", bg_color="black")
        self.admin_video.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")

        self.frame_admin_ctrl = ctk.CTkFrame(self.tab_admin)
        self.frame_admin_ctrl.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")

        self.entry_new_name = ctk.CTkEntry(self.frame_admin_ctrl, placeholder_text="Student Name")
        self.entry_new_name.pack(pady=20)
        
        self.btn_register = ctk.CTkButton(self.frame_admin_ctrl, text="Capture & Register Face", command=self.register_new_face)
        self.btn_register.pack(pady=10)
        self.lbl_msg = ctk.CTkLabel(self.frame_admin_ctrl, text="Look at camera and click Register", text_color="gray")
        self.lbl_msg.pack(pady=5)

        # --- REPORTS TAB ---
        self.btn_refresh = ctk.CTkButton(self.tab_reports, text="Refresh Logs", command=self.load_reports)
        self.btn_refresh.pack(pady=10)
        self.report_box = ctk.CTkTextbox(self.tab_reports, width=800, height=500, font=("Consolas", 12))
        self.report_box.pack(pady=10)

    def reload_db_data(self):
        users = database.get_all_users()
        self.known_face_encodings = [u['encoding'] for u in users]
        self.known_face_names = [u['name'] for u in users]
        self.known_face_ids = [u['id'] for u in users]
        self.log(f"System Loaded: {len(users)} users in database.")

    def start_camera_thread(self):
        if not self.running:
            self.cap = cv2.VideoCapture(0)
            self.running = True
            # Run video processing in a separate thread to prevent GUI freezing
            threading.Thread(target=self.video_processing_loop, daemon=True).start()
            # Run GUI update loop
            self.update_gui_loop()

    def video_processing_loop(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # --- 1. OPTIMIZATION: Resize ---
            # Processing small frames is 4x faster
            small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)

            # --- 2. FAST DETECTION (Haar) ---
            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            if len(faces) == 0:
                self.liveness_verified = False
                # Update UI text safely
                self.lbl_liveness.configure(text="Status: No Face", text_color="gray")

            for (x, y, w, h) in faces:
                # Convert Haar rect (x,y,w,h) to Dlib rect (left, top, right, bottom)
                dlib_rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
                
                # --- 3. LIVENESS CHECK (Dlib) ---
                shape = self.predictor(gray, dlib_rect)
                shape = np.array([[p.x, p.y] for p in shape.parts()])
                
                # Extract Eyes
                leftEye = shape[42:48]
                rightEye = shape[36:42]
                leftEAR = calculate_ear(leftEye)
                rightEAR = calculate_ear(rightEye)
                ear = (leftEAR + rightEAR) / 2.0

                # Blink Logic
                if ear < self.EYE_AR_THRESH:
                    self.blink_counter += 1
                else:
                    if self.blink_counter >= self.EYE_AR_CONSEC_FRAMES:
                        self.liveness_verified = True
                    self.blink_counter = 0

                # --- 4. RECOGNITION (Conditional) ---
                # Only run heavy recognition if verified OR if we need to show the name
                name = "Unknown"
                color = (0, 0, 255) # Red

                if self.liveness_verified:
                    color = (0, 255, 0) # Green
                    self.lbl_liveness.configure(text="Liveness: VERIFIED", text_color="green")
                    
                    # Run Heavy Recognition only when needed
                    # We crop the face from the ORIGINAL frame (high qual) for better accuracy
                    # Scale coordinates back up (x2 because we resized down by 0.5)
                    big_x, big_y, big_w, big_h = x*2, y*2, w*2, h*2
                    rgb_face = cv2.cvtColor(frame[big_y:big_y+big_h, big_x:big_x+big_w], cv2.COLOR_BGR2RGB)
                    
                    # Check if face exists in crop
                    if rgb_face.size > 0:
                        try:
                            # Encode just this face patch (Much faster than full image search)
                            encodings = face_recognition.face_encodings(rgb_face)
                            if len(encodings) > 0:
                                encoding = encodings[0]
                                matches = face_recognition.compare_faces(self.known_face_encodings, encoding, tolerance=0.5)
                                if True in matches:
                                    first_match_index = matches.index(True)
                                    name = self.known_face_names[first_match_index]
                                    self.current_name = name
                                    self.current_id = self.known_face_ids[first_match_index]
                        except Exception as e:
                            print("Recog Error:", e)

                else:
                    self.lbl_liveness.configure(text="Liveness: Please Blink", text_color="orange")

                # --- DRAWING ON FRAME ---
                # Scale up for display
                x, y, w, h = x*2, y*2, w*2, h*2
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_DUPLEX, 0.8, color, 1)

                # Draw Eyes (for feedback)
                for (ex, ey) in leftEye: cv2.circle(frame, (ex*2, ey*2), 2, (0, 255, 255), -1)
                for (ex, ey) in rightEye: cv2.circle(frame, (ex*2, ey*2), 2, (0, 255, 255), -1)

            # Update the latest frame safely
            with self.frame_lock:
                self.latest_frame = frame
            
            # Small sleep to prevent CPU hogging
            time.sleep(0.01)

    def update_gui_loop(self):
        if not self.running: return

        # Get the latest frame from the thread
        with self.frame_lock:
            frame = self.latest_frame

        if frame is not None:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = PIL.Image.fromarray(rgb)
            ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=(640, 480))
            
            self.lbl_video.configure(image=ctk_img)
            if self.tab_view.get() == "Admin":
                self.admin_video.configure(image=ctk_img)

        self.after(20, self.update_gui_loop)

    def register_new_face(self):
        name = self.entry_new_name.get()
        if not name:
            self.lbl_msg.configure(text="Error: Enter Name", text_color="red")
            return
        
        # Grab frame safely
        with self.frame_lock:
            if self.latest_frame is None: return
            frame = self.latest_frame.copy()

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes = face_recognition.face_locations(rgb)
        
        if not boxes:
            self.lbl_msg.configure(text="No face detected!", text_color="red")
            return
        
        encodings = face_recognition.face_encodings(rgb, boxes)
        if encodings:
            database.add_user(name, encodings[0])
            self.lbl_msg.configure(text=f"Success! {name} Added.", text_color="green")
            self.reload_db_data()
            self.entry_new_name.delete(0, 'end')
        else:
            self.lbl_msg.configure(text="Encoding failed. Try again.", text_color="red")

    def mark_attendance_action(self):
        if not self.liveness_verified:
            self.log("⚠️ Liveness Failed: Please blink first.")
            threading.Thread(target=lambda: self.tts_engine.say("Please blink to verify you are human"), daemon=True).start()
            return
            
        if self.current_name == "Unknown" or self.current_id is None:
            self.log("❌ Unknown Face. Register first.")
            threading.Thread(target=lambda: self.tts_engine.say("Face not recognized"), daemon=True).start()
            return

        result = database.mark_attendance(self.current_id, self.current_name)
        self.log(f"ACTION: {self.current_name} -> {result}")
        
        msg = f"Welcome {self.current_name}" if "Marked" in result else f"{self.current_name} already present"
        threading.Thread(target=lambda: self.tts_engine.say(msg), daemon=True).start()

        if self.var_email.get() and "Marked" in result:
            self.log("📧 Sending Email...")
            notifications.send_attendance_email("parent_dummy@example.com", self.current_name, "Now", result)
            
        # Reset Liveness
        self.liveness_verified = False
        self.lbl_liveness.configure(text="Liveness: Reset. Next user.", text_color="gray")

    def load_reports(self):
        rows = database.get_today_report()
        self.report_box.configure(state="normal")
        self.report_box.delete("0.0", "end")
        self.report_box.insert("end", f"{'NAME':<20} | {'TIME':<15} | {'STATUS':<10}\n")
        self.report_box.insert("end", "="*55 + "\n")
        for r in rows:
            self.report_box.insert("end", f"{r[0]:<20} | {r[1]:<15} | {r[2]:<10}\n")
        self.report_box.configure(state="disabled")

    def on_tab_change(self):
        if self.tab_view.get() == "Reports":
            self.load_reports()
        if self.tab_view.get() == "Admin" and not self.running:
            self.start_camera_thread()

    def log(self, msg):
        self.log_box.insert("end", f"• {msg}\n")
        self.log_box.see("end")

if __name__ == "__main__":
    app = FaceAttendanceApp()
    app.mainloop()