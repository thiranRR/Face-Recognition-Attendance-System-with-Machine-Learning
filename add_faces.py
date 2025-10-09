import cv2
import pickle
import numpy as np
import os

os.makedirs("data", exist_ok=True)

video = cv2.VideoCapture(0)
facedetect = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

faces_data = []
i = 0
name = input("Enter Your Name: ").strip()
if not name:
    print("Name cannot be empty.")
    exit()

print(f"\n[INFO] Capturing face data for {name}. Look at the camera...")
while True:
    ret, frame = video.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        crop_img = frame[y:y+h, x:x+w, :]
        resized_img = cv2.resize(crop_img, (50, 50))
        if len(faces_data) < 100 and i % 10 == 0:
            faces_data.append(resized_img)
        i += 1
        cv2.putText(frame, f"Samples: {len(faces_data)}/100", (30, 30),
                    cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 0), 2)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow("Collecting Faces", frame)
    k = cv2.waitKey(1)
    if k == ord('q') or len(faces_data) == 100:
        break

video.release()
cv2.destroyAllWindows()

faces_data = np.asarray(faces_data).reshape(100, -1)

# Save to pickle
if not os.path.exists('data/names.pkl'):
    names = [name] * 100
else:
    with open('data/names.pkl', 'rb') as f:
        names = pickle.load(f)
    names += [name] * 100

with open('data/names.pkl', 'wb') as f:
    pickle.dump(names, f)

if not os.path.exists('data/faces_data.pkl'):
    with open('data/faces_data.pkl', 'wb') as f:
        pickle.dump(faces_data, f)
else:
    with open('data/faces_data.pkl', 'rb') as f:
        faces = pickle.load(f)
    faces = np.append(faces, faces_data, axis=0)
    with open('data/faces_data.pkl', 'wb') as f:
        pickle.dump(faces, f)

print(f"[INFO] Data for {name} successfully saved.")
