import cv2

# pre-trained classifier
car_classifier_file = 'cars.xml'
ped_classifier_file = 'haarcascade_fullbody.xml'

# create claasifier
car_tracker = cv2.CascadeClassifier(car_classifier_file)
pedestrian_tracker = cv2.CascadeClassifier(ped_classifier_file)

windowName = 'Car Detector'
cv2.namedWindow(windowName,cv2.WINDOW_NORMAL)
cv2.setWindowProperty(windowName,cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
cv2.setWindowProperty(windowName,cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_NORMAL)

########################FILE ANALYSIS NAMES#####################################################
# image file
img_file = 'traffic_jam.jpg'
# video file
video = cv2.VideoCapture('car-pedestrian.mp4')



### VIDEO ###
while True:
    #read the current frame
    read_successful, frame = video.read()

    if read_successful:
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        break
        
    #detect cars    
    cars = car_tracker.detectMultiScale(gray_frame)

    # detect pedestrians
    pedestrians = pedestrian_tracker.detectMultiScale(gray_frame)

    #draw rectangles for cars
    for (x, y, w, h) in cars:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 0), 2)
    
    # #draw rectangles for pedestrians
    for (x, y, w, h) in pedestrians:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 0, 255), 2)

    cv2.imshow(windowName, frame) 

    if cv2.waitKey(33) == ord('q'):
        break

video.release()
cv2.destroyAllWindows()

### VIDEO ###

'''
### IMAGE ###
# create opencv image
image = cv2.imread(img_file)

# convert to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#detect cars
cars = car_tracker.detectMultiScale(gray_image)

#draw rectangles
for (x, y, w, h) in cars:
    cv2.rectangle(gray_image, (x,y), (x+w, y+h), (0, 255, 0), 2)


#display
windowName = 'Face Detector'
cv2.namedWindow(windowName,cv2.WINDOW_NORMAL)
cv2.setWindowProperty(windowName,cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
cv2.setWindowProperty(windowName,cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_NORMAL)
cv2.imshow('Car Detector', gray_image)


#wait for a key
cv2.waitKey()

### IMAGE ###
'''
print("Thanks for using this app")