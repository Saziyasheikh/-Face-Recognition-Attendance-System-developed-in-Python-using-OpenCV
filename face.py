import cv2
import os
import datetime
import numpy as np

# 1. OpenCV ka Inbuilt Face Detector Load Karna
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 2. Images folder se saare faces load karna
path = 'images'
known_images = []
known_names = []

if not os.path.exists(path):
    os.makedirs(path)
    print(f"'{path}' folder bana diya hai. Usme photos daalein!")

# Folder ke andar ki saari photos read karna
for cl in os.listdir(path):
    img_path = os.path.join(path, cl)
    curImg = cv2.imread(img_path)
    if curImg is not None:
        # Image ko gray scale mein convert karna recognition ke liye
        curImg_gray = cv2.cvtColor(curImg, cv2.COLOR_BGR2GRAY)
        # Image ko ek standard size dena (200x200)
        curImg_resized = cv2.resize(curImg_gray, (200, 200))
        
        known_images.append(curImg_resized)
        known_names.append(os.path.splitext(cl)[0].upper())

# LBPH Face Recognizer model initialize aur train karna
if len(known_images) > 0:
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    # Labels generate karna (0, 1, 2...)
    labels = np.array(list(range(len(known_names))))
    recognizer.train(known_images, labels)
    print(f"Successfully trained on names: {known_names}")
else:
    print("WARNING: 'images' folder mein koi photo nahi mili!")
    recognizer = None

# 3. Attendance File Create Karna
attendance_file = "Attendance.csv"
if not os.path.exists(attendance_file):
    with open(attendance_file, "w") as f:
        f.write("Name,Time,Date\n")

def markAttendance(name):
    with open(attendance_file, "r+") as f:
        myDataList = f.readlines()
        nameList = [line.split(',')[0] for line in myDataList]
        if name not in nameList:
            now = datetime.datetime.now()
            f.writelines(f'\n{name},{now.strftime("%H:%M:%S")},{now.strftime("%d-%b-%Y")}')
            print(f"Attendance Marked for {name}")

# 4. Web-Camera Open Karna
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:
    success, img = cap.read()
    if not success:
        break

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        name = "UNKNOWN"
        
        # Agar models trained hain toh match karne ki koshish karna
        if recognizer is not None:
            face_roi = gray[y:y+h, x:x+w]
            face_roi_resized = cv2.resize(face_roi, (200, 200))
            
            # Predict karna
            label, confidence = recognizer.predict(face_roi_resized)
            
            # Confidence jitna kam hoga, match utna perfect hoga (threshold = 80)
            if confidence < 80:
                name = known_names[label]
        
        cv2.putText(img, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        if name != "UNKNOWN":
            markAttendance(name)

    cv2.imshow('Attendance System', img)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()