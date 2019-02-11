


####################################################################
#    QuiVision for surveillance detection Developed by Akash Munshi Dated 12th February 2019     #
#    QuiVision-Beta                                                   #
#    Copyright 2019 Akash Munshi and Quinch Systems Pvt. Ltd.      #
####################################################################

"""
    A full fledged human detection system that enables the concerned organisation to
    monitor their areas of concern.
    ->It includes Full body detection along with upper body, lower body and face detection
    ->This beta phase model will gather more updates for example, sending alerts on observing suspicous activities
"""
####################################################################



import cv2
import time
import os
#from package import alert_subsystem
person_cascade = cv2.CascadeClassifier(
    os.path.join('haarcascade_fullbody.xml'))
upper_body_cascade = cv2.CascadeClassifier(os.path.join('haarcascade_upperbody.xml'))
lower_body_cascade = cv2.CascadeClassifier(os.path.join('haarcascade_lowerbody.xml'))
face_cascade = cv2.CascadeClassifier(os.path.join('haarcascade_frontalface_default.xml'))

vid = cv2.VideoCapture('crowd.mp4')
#vid = cv2.VideoCapture(0)
#vid.set(cv2.CAP_PROP_FPS, 12)
while True:
    ret, frame = vid.read()
    if ret:
        time_start = time.time()
        #resizing the frame
        frame = cv2.resize(frame, (500, 500))
        gray_scale = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        human = person_cascade.detectMultiScale(gray_scale)

        time_end = time.time()
        print("Time difference", time_end - time_start)

    #let's observe some video
        for (x, y, w, h) in human:
            #for whole body detetction
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
            roi_gray = gray_scale[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
            #for upper body detection
            upper_body = upper_body_cascade.detectMultiScale(roi_gray)
            for(ux, uy, uw, uh) in upper_body:
                cv2.rectangle(roi_color, (ux, uy), (ux+uw, uy+uh), (0 , 0, 255), 2)
            #for lower body detection
            lower_body = lower_body_cascade.detectMultiScale(roi_gray)
            for(lx, ly, lw, lh) in lower_body:
                cv2.rectangle(roi_color, (lx,ly), (lx+lw, ly+lh), (255, 0, 0), 2)
            #for face detection
            face = face_cascade.detectMultiScale(roi_gray)
            for (fx, fy, fw, fh) in face:
                cv2.rectangle(roi_color, (fx, fy), (fx + fw, fy + fh), (120, 230, 0), 4)

        cv2.imshow("human_detection", frame)
        #press e to exit
    if cv2.waitKey(1) & 0xFF == ord("e"):
        break