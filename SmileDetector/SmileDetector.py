import cv2

#classifier for face
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
smile_detector = cv2.CascadeClassifier('haarcascade_smile.xml')

# webcam input
webcam = cv2.VideoCapture(0)

#window properties
windowName = 'Smile Detector'
cv2.namedWindow(windowName,cv2.WINDOW_NORMAL)
cv2.setWindowProperty(windowName,cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
cv2.setWindowProperty(windowName,cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_NORMAL)

while True:
    #read frame
    successful_read, frame = webcam.read()

    #exit if error with frame
    if not successful_read:
        break

    #grayscale conversion    
    frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #detect face locations first
    faces = face_detector.detectMultiScale(frame_grayscale)
    

    #draw rectangles around all faces found
    for (x, y, w, h) in faces:
        #rectangle around face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (100, 250, 50), 2)

        #get "sub-frame" (using numpy N-dimensional array slicing)
        face = frame[y:(y+h), x:(x+w)]               # y-direction first, then x... it's just how the frame is
        #grayscale conversion
        face_grayscale = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

        '''
        then detect smile locations:
            scale factor is blur for the frame so we can easily detect facial features
            minNeighbors means the minimum amount of neighbors of rectangles needed to be counted as a smile. very dense detection zones to become counted as a smile
        '''
        smiles = smile_detector.detectMultiScale(face_grayscale, scaleFactor=1.7, minNeighbors=20) 

        #for each face, draw all the smiles
        # for (x2, y2, w2, h2) in smiles:
        #     cv2.rectangle(face, (x2, y2), (x2+w2, y2+h2), (255, 0, 0), 2)

        #label the face as smiling
        if (len(smiles) > 0):
            cv2.putText(frame, 'smiling!', (x,y+h+40), fontScale=3, fontFace=cv2.FONT_HERSHEY_PLAIN, color=(255,255,255))

    #display the frame
    cv2.imshow(windowName, frame)

    #wait for q press for quit
    if cv2.waitKey(33) == ord('q'):
        break

#cleanup
webcam.release()
cv2.destroyAllWindows() 


print('Thanks for using the smile detector')