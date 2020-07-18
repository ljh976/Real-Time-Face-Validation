import numpy as np
import cv2
import time

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

focus = 0  # min: 0, max: 255, increment:5
cap.set(28, focus) 
                                          
#bring cascade 
face_cascade = cv2.CascadeClassifier('xmls/haarcascade_frontalface_default.xml') 
eyes_cascade = cv2.CascadeClassifier('xmls/haarcascade_eye.xml')

nose_cascade = cv2.CascadeClassifier('xmls/Nariz.xml')
mouth_cascade = cv2.CascadeClassifier('xmls/Mouth.xml')



#font info
# font 
font = cv2.FONT_HERSHEY_SIMPLEX 
   
# fontScale 
fontScale = 0.75
   
# Blue color in BGR 
color = (150, 255, 0) 
  
# Line thickness of 2 px 
thickness = 2

#for photo index
count = 0

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    # Convert the image to gray scale 
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
      
    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (18, 18))   
    # Performing OTSU threshold 
    ret, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV) 
    # Appplying dilation on the threshold image 
    dilation = cv2.dilate(thresh1, rect_kernel, iterations = 1) 
    # Finding contours 
    contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL,  
                                                     cv2.CHAIN_APPROX_NONE) 
    
    # Our operations on the frame come here
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 

    #face set
    face = face_cascade.detectMultiScale(frame, 1.3, 5)
    eyes = eyes_cascade.detectMultiScale(frame, 1.3, 5, minSize = (40, 40), maxSize = (45, 45))
    nose = nose_cascade.detectMultiScale(frame, 1.3, 5, minSize = (40, 40), maxSize = (45, 45))
    mouth = mouth_cascade.detectMultiScale(frame, 1.3, 5, minSize = (40, 40), maxSize = (70, 70))
    
    #area of face picture
    picture_area_dict = {
        'x' : 210,
        'y' : 100,
        'w' : 400,
        'h' : 400,
    }
    
    cv2.rectangle(frame,(picture_area_dict['x'], picture_area_dict['y']),
                    (picture_area_dict['x']+int(picture_area_dict['w']/2), picture_area_dict['y']+
                    int(picture_area_dict['h']/2)),(255,255,255),2)
    
    cv2.putText(frame, str(count), (50, 20), font, fontScale, (150, 0, 200), thickness, cv2.LINE_AA)
    
    #detection part 
    #face part
    if np.any(face):
        print("Found a face")
        for (x, y, w, h) in face:
            arr = [x, y, x+w, y+h]
            print(arr)
           # arr = []
            
            mercy = 20
            
            #text org (x, y) 
            org = (x, y-20)
            #draw circle on face
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
            #cv2.putText(frame, '', org, font,  
                   #fontScale, color, thickness, cv2.LINE_AA)
    #eyes part
    if np.any(eyes):# and np.any(mouth) and np.any(nose):
        print("Found eyes")
        for (ex, ey, ew, eh) in eyes:
            arr = [ex, ey, ew, eh]
            print(arr)
            arr = []
            #center: (x+w/2, y+h/2)
        
            #text org (x, y) 
            org = (ex, ey-20)
            #draw circle on face
            cv2.rectangle(frame,(ex,ey),(ex+ew,ey+eh),(0,255,255),2)
            #cv2.putText(frame, str(len(eyes)), org, font,  
                #   fontScale, color, thickness, cv2.LINE_AA)
                
    #nose part
    if np.any(nose):# and np.any(mouth) and np.any(nose):
        print("Found nose")
        for (x, y, w, h) in nose:
            arr = [x, y, w, h]
            print(arr)
            arr = []
            org = (x, y-20)
            #draw circle on face
            cv2.rectangle(frame,(x,y),(x+w,y+h),(100,150,100),2)
            mercy = 0
            if ((x) > picture_area_dict['x']-mercy and (y) > picture_area_dict['y']-mercy and 
                x+int(w/2) < picture_area_dict['x']+int(picture_area_dict['w']/2)+mercy and 
                y+int(h/2) < picture_area_dict['y']+int(picture_area_dict['h']/2)+mercy and 
                len(eyes) > 1 and len(mouth) != 0 and len(face) != 0):
                count += 1
                if count < 10:
                    #time.sleep(0.005)
                    img_name = "mugshot_{}.png".format(count)
                    cv2.imwrite(img_name, frame)
                
            
            
    #mouth 
    if np.any(mouth):# and np.any(mouth) and np.any(nose):
        print("Found nose")
        for (ex, ey, ew, eh) in mouth:
            arr = [ex, ey, ew, eh]
            print(arr)
            arr = []
            org = (ex, ey-20)
            #draw circle on face
            cv2.rectangle(frame,(ex,ey),(ex+ew,ey+eh),(150,150,100),2)


    # Display the resulting frame
    cv2.namedWindow("frame", 0);
    cv2.resizeWindow("frame", 800, 680);
    cv2.imshow('frame',frame) #imshow('window name', color)
    if cv2.waitKey(1) & 0xFF == ord('q'):   
        break
   
    
# When everything done, release the capture

cap.release()
cv2.destroyAllWindows()
