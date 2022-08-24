from gpiozero import AngularServo,Device,Servo
from gpiozero.pins.pigpio import PiGPIOFactory
from time import sleep
import os
import socket
import subprocess
from picamera import PiCamera
from picamera.array import PiRGBArray
import mediapipe as mp
import cv2
import numpy as np
import pandas as pd
import os
from tensorflow.keras.models import load_model


#definisi kamera
camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 30
rawCapture = PiRGBArray(camera, size=(640,480))


mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

os.system('sudo pigpiod')
os.system('sudo pigpiod')
    
def wlan_ip():
    result=subprocess.run('ifconfig',stdout=subprocess.PIPE,text=True).stdout.lower()
    wlan_found = None
    for i in range(len(result.split('\n'))):
        if 'wlan0' in result.split('\n')[i]:
            wlan_found = result.split('\n')[i+1]
            for i in range(len(wlan_found.split(' '))):
                if 'inet' == wlan_found.split(' ')[i]:
                    target_ip = wlan_found.split(' ')[i+1]
                    return target_ip
    return '192.168.0.1'

print(wlan_ip())

factory = PiGPIOFactory(host='0')

s = AngularServo(17,
                 min_angle=0, max_angle=180,
                 min_pulse_width=0.5/1000,
                 max_pulse_width=2.5/1000,
                 frame_width=20/1000,
                 pin_factory = factory)

s1 = AngularServo(18,
                 min_angle=0, max_angle=180,
                 min_pulse_width=0.5/1000,
                 max_pulse_width=2.5/1000,
                 frame_width=20/1000,
                 pin_factory = factory)

s.mid()
s1.mid()
angle_s = s.angle
angle_s1 = s1.angle
sleep(0.1)

Classifier = load_model("fivo/best_model.h5")

with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5, max_num_hands=1) as hands:
    for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
        image = frame.array

        # BGR 2 RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Flip on horizontal
        image = cv2.flip(image, 1)
        
        # Set flag
        image.flags.writeable = False
        
        # Detections
        results = hands.process(image)
        
        # Set flag to true
        image.flags.writeable = True
        
        # RGB 2 BGR
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Detections
        inputs = []
        predicted = 'Unknown'
        # Rendering results
        if results.multi_hand_landmarks:
            for num, hand in enumerate(results.multi_hand_landmarks):
                mp_drawing.draw_landmarks(image, hand, mp_hands.HAND_CONNECTIONS, 
                                        mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                                        mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2),
                                         )
                
                for lm in hand.landmark:
                    inputs.append(lm.x)
                    inputs.append(lm.y)
                    inputs.append(lm.z)
            inputs = pd.DataFrame(inputs).T
            inputs = np.expand_dims(inputs,-1)
            predicted = Classifier.predict(inputs)
            predicted = pd.DataFrame(predicted)
            predicted.columns = ['belakang','depan','kanan','kiri','stop']
            predicted = predicted.idxmax(axis=1).values[0]
            print(predicted)
            
            if predicted =='belakang':
                if angle_s1 >= 180:
                    angle_s1 = 180
                else:
                    angle_s1 += 5
                    s1.angle = angle_s1
            if predicted =='depan':
                if angle_s1 <= 0:
                    angle_s1 = 0
                else:
                    angle_s1 -= 5
                    s1.angle = angle_s1
                    
            if predicted =='kiri':
                if angle_s >= 180:
                    angle_s = 180
                else:
                    angle_s += 5
                    s.angle = angle_s
            if predicted =='kanan':
                if angle_s <= 0:
                    angle_s = 0
                else:
                    angle_s -= 5
                    s.angle = angle_s
                    
            image = cv2.putText(image, predicted, (0, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, 
                     (0, 0, 255), 2, cv2.LINE_AA, False)
        #cv2.imshow('tracking',image)
        key = cv2.waitKey(1) & 0xFF
                
        rawCapture.truncate(0)
        
        if key == ord("q"):
            s.stop()
            s1.stop()
            break

cv2.destroyAllWindows()