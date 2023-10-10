import cv2 as cv
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

#resize frame
# def rescaleframe(frame,scale=0.5):
#     width = int(frame.shape[1]*0.5)
#     height = int(frame.shape[0]*0.5)
#     dimensions =(width,height)

#     cv.resize(frame,dimensions,interpolation=cv.INTER_AREA)


counter  = 0
stage = None 


capture = cv.VideoCapture(0)
with mp_pose.Pose(min_detection_confidence =0.5,min_tracking_confidence =0.5) as pose:
    while True:
        isTrue,frame =capture.read()
        
        img = cv.cvtColor(frame,cv.COLOR_BGR2RGB) 
        img.flags.writeable=False

        results =  pose.process(img)
        # print(results.pose_landmarks)

        img.flags.writeable= True
        img =cv.cvtColor(img,cv.COLOR_RGB2BGR)

        #calculating anlge
        def calculate_angle(a,b,c):
            a = np.array(a) # fitst(shoulder point)
            b = np.array(b) # mid point 
            c = np.array(c) # end 

            radians = np.arctan2(c[1]-b[1],c[0]-b[0]) - np.arctan2(a[1]-b[1],a[0]-b[0])

            angle = np.abs(radians*180.0/np.pi)
            

            if angle > 180.0:
                angle = 360 -angle 

                return angle 

        try:
            # Extract landmarks 
            landmarks = results.pose_landmarks.landmark
            
            # Get coordinates
            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

            
            #calculate angle 
            angle = calculate_angle(shoulder,elbow,wrist)

            # visualize angle 
            cv.putText(img,str(angle),tuple(np.multiply(elbow,[640,480]).astype(int)),cv.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),2,cv.LINE_AA)
            
            #curler counter logic
            if angle > 160:
                stage = "down"
            if angle < 80 and stage =="down":
                stage = "up"
                counter += 1
                print(counter)
                  
        except:
            pass


        # Render curl counter 
        #Setup status bar 
        cv.rectangle(img,(0,0),(230,73),(245,117,16),-1)

        # rep data
        cv.putText(img,"REPS",(15,13),cv.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),1,cv.LINE_AA)
        cv.putText(img,str(counter),(10,60),cv.FONT_HERSHEY_SIMPLEX,2,(255,255,255),2,cv.LINE_AA)
        # rep data
        cv.putText(img,"STAGE",(95,12),cv.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),1,cv.LINE_AA)
        cv.putText(img,stage,(80,60),cv.FONT_HERSHEY_SIMPLEX,2,(255,255,255),2,cv.LINE_AA)
        
        mp_drawing.draw_landmarks(img,results.pose_landmarks,mp_pose.POSE_CONNECTIONS,
                                   mp_drawing.DrawingSpec(color=(0,255,0),thickness= 2,circle_radius=2),
                                   mp_drawing.DrawingSpec(color=(255,0,255),thickness= 2,circle_radius=2))

        # for lndmrk in mp_pose.PoseLandmark:
        #     print(lndmrk)
        
        

        cv.imshow('gym tracker',img)

        if cv.waitKey(20) & 0xFF ==ord("q"):
            break
    capture.release()
    cv.destroyAllWindows()