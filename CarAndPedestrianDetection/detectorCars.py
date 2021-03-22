import cv2

car_classifier_file = 'cars.xml'
ped_classifier_file = 'haarcascade_fullbody.xml'

car_tracker = cv2.CascadeClassifier(car_classifier_file)
pedestrian_tracker = cv2.CascadeClassifier(ped_classifier_file)

#format stuff
windowName = 'Car Detector'
cv2.namedWindow(windowName,cv2.WINDOW_NORMAL)
cv2.setWindowProperty(windowName,cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
cv2.setWindowProperty(windowName,cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_NORMAL)

#image file
img_file = 'traffic_jam.jpg'



### IMAGE STUFF

image = cv2.imread(img_file)

grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

cars = car_tracker.detectMultiScale(grayscale)

#draw rectangles
for (x, y, w, h) in cars:
    cv2.rectangle(image, (x,y), (x+w, y+h), (0, 255, 0), 2)

#display
cv2.imshow(windowName, image)

#wait key
cv2.waitKey()

print("thanks for using this") 