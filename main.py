from keras.models import load_model
from time import sleep
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import cv2
import numpy as np
import tkinter as tk
from tkinter import messagebox

# Load pre-trained models
face_classifier = cv2.CascadeClassifier(r'C:\emotion_detection_cnn\haarcascade_frontalface_default.xml')
classifier = load_model(r'C:\emotion_detection_cnn\model.h5')

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Global variable to control the video loop
video_loop = False

def start_emotion_detection():
    global video_loop
    video_loop = True
    cap = cv2.VideoCapture(0)
    
    while video_loop:
        _, frame = cap.read()
        labels = []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray)
        
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
            roi_gray = gray[y:y + h, x:x + w]
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
            
            if np.sum([roi_gray]) != 0:
                roi = roi_gray.astype('float') / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)
                
                prediction = classifier.predict(roi)[0]
                label = emotion_labels[prediction.argmax()]
                label_position = (x, y)
                cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(frame, 'No Faces', (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow('Emotion Detector', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

def stop_emotion_detection():
    global video_loop
    video_loop = False

def minimize_window():
    root.attributes('-fullscreen', False)
    root.iconify()

def restore_window():
    root.deiconify()
    root.attributes('-fullscreen', True)

def on_closing():
    if messagebox.askokcancel("Quit", "Do you want to quit?"):
        stop_emotion_detection()
        root.destroy()

# Setup Tkinter UI
root = tk.Tk()
root.title("Emotion Detection")
root.attributes('-fullscreen', True)  # Make the window full screen

# Create a frame for the top buttons
top_frame = tk.Frame(root, bg='grey')
top_frame.pack(side=tk.TOP, fill=tk.X)

minimize_button = tk.Button(top_frame, text="_", command=minimize_window, font=('Helvetica', 12))
minimize_button.pack(side=tk.RIGHT, padx=5, pady=5)

restore_button = tk.Button(top_frame, text="◻", command=restore_window, font=('Helvetica', 12))
restore_button.pack(side=tk.RIGHT, padx=5, pady=5)

exit_button = tk.Button(top_frame, text="X", command=on_closing, font=('Helvetica', 12))
exit_button.pack(side=tk.RIGHT, padx=5, pady=5)

# Create a frame for the action buttons
frame = tk.Frame(root)
frame.pack(pady=20)

start_button = tk.Button(frame, text="Start", command=start_emotion_detection, font=('Helvetica', 20))
start_button.pack(side=tk.LEFT, padx=20)

stop_button = tk.Button(frame, text="Stop", command=stop_emotion_detection, font=('Helvetica', 20))
stop_button.pack(side=tk.LEFT, padx=20)

root.protocol("WM_DELETE_WINDOW", on_closing)

root.mainloop()
