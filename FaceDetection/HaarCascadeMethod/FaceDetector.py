import cv2
from random import randrange

# Load pre-trained data on face frontals (haar cascade algorithm)
trained_face_data = cv2.CascadeClassifier(
    'haarcascade_frontalface_default.xml')

# # choose your video cam
# webcam = cv2.VideoCapture(0)
# while True:
#     # Read current frame
#     successful_frame_read, frame = webcam.read()

#     grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     # face detection
#     face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)
#     # draw rectangles
#     for (x, y, w, h) in face_coordinates:
#         cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

#     cv2.imshow('Face Detector', frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# webcam.release()
# cv2.destroyAllWindows()

# choose an image
img = cv2.imread('rock1.jpeg')
grayscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #BGR not RGB

# #face detection
face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)
# face_coordinates = trained_face_data.detectMultiScale(grayscaled_img, minNeighbors=5, outputRejectLevels = True)
# faces = face_coordinates[0]
# neighbours = face_coordinates[1]
# weights = face_coordinates[2]
print(face_coordinates)

# # loop thru faces
for (x,y,w,h) in face_coordinates:
    cv2.rectangle(img, (x,y), (x+w, y+h), (randrange(128,256), randrange(128,256), randrange(128,256)), 2)

cv2.rectangle(img, (420, 69), (69, 420), (191, 64, 191), 8)
# #Display
windowName = 'Face Detector'
cv2.namedWindow(windowName,cv2.WINDOW_NORMAL)
cv2.setWindowProperty(windowName,cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
cv2.setWindowProperty(windowName,cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_NORMAL)
cv2.imshow(windowName, img)
cv2.waitKey() # wait for any key stroke

print("Code Completed")
