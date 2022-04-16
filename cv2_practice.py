import cv2
import os
import numpy as np
import dlib
from imutils import face_utils


img = cv2.imread('resources/lenna.png')
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(
    '/Users/spkaikai/Documents/Tsukuba/resources/shape_predictor_68_face_landmarks.dat')  # http://dlib.net/files/
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
rects = detector(gray, 0)
for rect in rects:
    shape0 = predictor(gray, rect)
    shape0 = np.array(face_utils.shape_to_np(shape0))


# https://learnopencv.com/head-pose-estimation-using-opencv-and-dlib/

i = 0

image_points = np.array([
    (shape0[30, :]),  # nose tip
    (shape0[8, :]),  # Chin
    (shape0[36, :]),  # Left eye left corner
    (shape0[45, :]),  # right eye right corner
    (shape0[48, :]),  # left mouth corner
    (shape0[54, :])  # right mouth corner

], dtype='double')
print(image_points)

# for item in list(image_points):
#     item = list(item)
#     cv2.putText(img, str(i), (item[0], item[1]), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 0, 0), 1)
#     cv2.circle(img, (item[0], item[1]), 5, (255, 0, 0), 1)
#     i += 1


model_points = np.array([
    (0.0, 0.0, 0.0),             # Nose tip
    (0.0, -330.0, -65.0),        # Chin
    (-225.0, 170.0, -135.0),     # Left eye left corner
    (225.0, 170.0, -135.0),     # Right eye right corne
    (-150.0, -150.0, -125.0),    # Left Mouth corner
    (150.0, -150.0, -125.0)      # Right mouth corner
])

size = img.shape
focal_length = size[1]
center = (size[1]/2, size[0]/2)
camera_matrix = np.array(
    [[focal_length, 0, center[0]],
     [0, focal_length, center[1]],
        [0, 0, 1]], dtype="double"
)

print("Camera Matrix :\n {}".format(camera_matrix))

dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
(success, rotation_vector, translation_vector) = cv2.solvePnP(
    model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
print("Rotation Vector:\n {0}".format(rotation_vector))
print("Translation Vector:\n {0}".format(translation_vector))

# Project a 3D point (0, 0, 1000.0) onto the image plane.
# We use this to draw a line sticking out of the nose
(nose_end_point2D, jacobian) = cv2.projectPoints(
    np.array([(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)

for p in image_points:
    cv2.circle(img, (int(p[0]), int(p[1])), 3, (0, 0, 255), -1)

p1 = (int(image_points[0][0]), int(image_points[0][1]))
p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
cv2.line(img, p1, p2, (255, 0, 0), 2)

print('p1: ', p1)
print('p2: ', p2)

cv2.imshow('output', img)
k = cv2.waitKey(10000) & 0xff
if k == 27:         # wait for ESC key to exit
    cv2.destroyWindow('image')
#
# imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# imgBlur = cv2.GaussianBlur(imgGray, (7, 7), 0)
# imgCanny = cv2.Canny(img, 150, 200)
# imgDilation = cv2.dilate(imgCanny, kernel, iterations=5)

# cv2.imshow('gray image', imgGray)
# cv2.imshow('blur image', imgBlur)
# cv2.imshow('canny image', imgCanny)
# cv2.imshow('dilate image', imgDilation)
# cv2.waitKey(0)

# fpath = os.path.dirname(cv2.__file__) + '/data/haarcascade_upperbody.xml'
# print(fpath)
# cascade = cv2.CascadeClassifier(fpath)

#
# SCALE = 2
#
# # img = cv2.imread('resources/lenna.png')
# cap = cv2.VideoCapture('resources/IMG_4988.mp4')
#
# while True:
#     _, img = cap.read()
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     height, width = gray.shape[:2]
#     img = cv2.resize(gray, (int(width/SCALE), int(height/SCALE)))
#
#     faces = cascade.detectMultiScale(img)
#     for (x, y, w, h) in faces:
#         cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 1)
#     cv2.imshow('img', img)
#     # cv2.waitKey()
#
#     k = cv2.waitKey(30) & 0xff
#     if k == 27:
#         break
#
# cap.release()
