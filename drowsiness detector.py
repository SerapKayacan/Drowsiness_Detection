import os
from sympy import true
from ultralytics import YOLO

import torch
import matplotlib
import numpy as np
import cv2
import tkinter as tk
from PIL import ImageTk, Image
import winsound

#Load custom YOLOv8 model
model = YOLO('best.pt')

cap = cv2.VideoCapture(0)
drowsy_detected = False
counter=0

def reset_counter():
    global counter
    counter = 0

def play_sound():
    sound_file = "C:/Users/SAAD SYED/Desktop/Graduation Thesis/AlarmInterface.wav"
    winsound.PlaySound(sound_file, winsound.SND_ASYNC)

def update_frame():
    global drowsy_detected, counter

    ret, frame = cap.read()
    #Adding the counter text to the frame
    cv2.putText(frame, f"Counter: {counter}", (10, 30), cv2.FONT_HERSHEY_COMPLEX, 1,(0, 255, 0), 2)

    results = model.predict(source=frame, show=True)
    for r in results:
        if len(r.boxes.cls)> 0:
            dclass = r.boxes.cls[0].item()
            print(dclass)

            if dclass == 1.0:
                if not drowsy_detected:
                    play_sound()
                    drowsy_detected = True
                    counter += 1
                else:
                    drowsy_detected = False
    
    #Convert the frame to a format compatible with Tkinter
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_pil = Image.fromarray(frame_rgb)
    frame_tk = ImageTk.PhotoImage(frame_pil)

    
    #Calling the update_frame function again after delay
    root.after(1, update_frame)

#Create Tkinter window
root = tk.Tk()
root.title("Drowsiness Detector")


#Create Frame to hold the widgets
frame = tk.Frame(root)
frame.pack()

video_label = tk.Label(frame)
video_label.pack()

cap = cv2.VideoCapture(0)

#Create Label widget for video frame
frame_label = tk.Label(frame)
frame_label.pack()
#Create Button widget for reset
reset_button = tk.Button(frame, text="Reset Counter", command=reset_counter)
reset_button.pack()
quit_button = tk.Button(frame, text="Quit", command=quit)
quit_button.pack()

#Calling update_frame function to start live detection
update_frame()

#Run Tkinter event loop
root.mainloop()