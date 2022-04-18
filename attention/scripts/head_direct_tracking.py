#!/usr/bin/env python
# coding: utf-8

import rospy
import math
import sys
import dlib
from imutils import face_utils
import time

# for ObjectTracker
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Point

# for NeckYawPitch
import actionlib
from control_msgs.msg import (
    FollowJointTrajectoryAction,
    FollowJointTrajectoryGoal,
    JointTrajectoryControllerState
)
from trajectory_msgs.msg import JointTrajectoryPoint


class ObjectTracker:
    def __init__(self):
        self._bridge = CvBridge()
        self._image_sub = rospy.Subscriber(
            "/sciurus17/camera/color/image_raw", Image, self._image_callback, queue_size=1)
        self._image_pub = rospy.Publisher("~output_image", Image, queue_size=1)
        self._object_rect = [0, 0, 0, 0]
        self._object_target = [0, 0]
        self._image_shape = Point()
        self._object_detected = False

        self._CV_MAJOR_VERSION, _, _ = cv2.__version__.split('.')

        # カスケードファイルの読み込み
        # 例
        self._face_cascade = cv2.CascadeClassifier(
            "/home/sciurus/Documents/data/haarcascades/haarcascade_frontalface_default.xml")
        self._eyes_cascade = cv2.CascadeClassifier(
            "/home/sciurus/Documents/data/haarcascades/haarcascade_eye.xml")
        # self._face_cascade = ""
        # self._eyes_cascade = ""

    def _image_callback(self, ros_image):
        try:
            input_image = self._bridge.imgmsg_to_cv2(ros_image, "bgr8")
        except CvBridgeError as e:
            rospy.logerr(e)

        # 画像のwidth, heightを取得
        self._image_shape.x = input_image.shape[1]
        self._image_shape.y = input_image.shape[0]

        # オブジェクト(特定色 or 顔) の検出
        # output_image = self._detect_orange_object(input_image)
        # output_image = self._detect_blue_object(input_image)
        # output_image = self._detect_face(input_image)
        output_image = self._detect_head_direct(input_image)

        self._image_pub.publish(self._bridge.cv2_to_imgmsg(output_image, "bgr8"))

    def get_object_position(self):
        # 画像中心を0, 0とした座標系におけるオブジェクトの座標を出力
        # オブジェクトの座標は-1.0 ~ 1.0に正規化される

        # object_center = Point(
        #        self._object_rect[0] + self._object_rect[2] * 0.5,
        #        self._object_rect[1] + self._object_rect[3] * 0.5,
        #        0)
        object_center = Point(self._object_target[0], self._object_target[0], 0)

        # 画像の中心を0, 0とした座標系に変換
        translated_point = Point()
        translated_point.x = object_center.x - self._image_shape.x * 0.5
        translated_point.y = -(object_center.y - self._image_shape.y * 0.5)

        # 正規化
        normalized_point = Point()
        if self._image_shape.x != 0 and self._image_shape.y != 0:
            normalized_point.x = translated_point.x / (self._image_shape.x * 0.5)
            normalized_point.y = translated_point.y / (self._image_shape.y * 0.5)

        return normalized_point

    def object_detected(self):
        return self._object_detected

    def _detect_color_object(self, bgr_image, lower_color, upper_color):
        # 画像から指定された色の物体を検出する

        MIN_OBJECT_SIZE = 1000  # px * px

        # BGR画像をHSV色空間に変換
        hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)

        # 色を抽出するマスクを生成
        mask = cv2.inRange(hsv, lower_color, upper_color)

        # マスクから輪郭を抽出
        if self._CV_MAJOR_VERSION == '4':
            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        else:
            _, contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # 輪郭を長方形に変換し、配列に格納
        rects = []
        for contour in contours:
            approx = cv2.convexHull(contour)
            rect = cv2.boundingRect(approx)
            rects.append(rect)

        self._object_detected = False
        if len(rects) > 0:
            # 最も大きい長方形を抽出
            rect = max(rects, key=(lambda x: x[2] * x[3]))

            # 長方形が小さければ検出判定にしない
            if rect[2] * rect[3] > MIN_OBJECT_SIZE:
                # 抽出した長方形を画像に描画する
                cv2.rectangle(bgr_image,
                              (rect[0], rect[1]),
                              (rect[0] + rect[2], rect[1] + rect[3]),
                              (0, 0, 255), thickness=2)

                self._object_rect = rect
                self._object_detected = True

        return bgr_image

    def _detect_orange_object(self, bgr_image):
        # H: 0 ~ 179 (0 ~ 360°)
        # S: 0 ~ 255 (0 ~ 100%)
        # V: 0 ~ 255 (0 ~ 100%)
        lower_orange = np.array([5, 127, 127])
        upper_orange = np.array([20, 255, 255])

        return self._detect_color_object(bgr_image, lower_orange, upper_orange)

    def _detect_blue_object(self, bgr_image):
        # H: 0 ~ 179 (0 ~ 360°)
        # S: 0 ~ 255 (0 ~ 100%)
        # V: 0 ~ 255 (0 ~ 100%)
        lower_blue = np.array([100, 127, 127])
        upper_blue = np.array([110, 255, 255])

        return self._detect_color_object(bgr_image, lower_blue, upper_blue)

    def _detect_face(self, bgr_image):
        # 画像から顔(正面)を検出する

        SCALE = 1

        # カスケードファイルがセットされているかを確認
        if self._face_cascade == "" or self._eyes_cascade == "":
            rospy.logerr("cascade file does not set")
            return bgr_image

        # BGR画像をグレー画像に変換
        gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)

        # 処理時間短縮のため画像を縮小
        height, width = gray.shape[:2]
        small_gray = cv2.resize(gray, (int(width/SCALE), int(height/SCALE)))

        # カスケードファイルを使って顔認識
        small_faces = self._face_cascade.detectMultiScale(small_gray)

        self._object_detected = False
        for small_face in small_faces:
            # 顔の領域を元のサイズに戻す
            face = small_face*SCALE

            # グレー画像から顔部分を抽出
            roi_gray = gray[
                face[1]:face[1]+face[3],
                face[0]:face[0]+face[2]]

            # 顔の中から目を検知
            eyes = self._eyes_cascade.detectMultiScale(roi_gray)

            # 目を検出したら、対象のrect(座標と大きさ)を記録する
            if len(eyes) > 0:
                cv2.rectangle(bgr_image,
                              (face[0], face[1]),
                              (face[0]+face[2], face[1]+face[3]),
                              (0, 0, 255), 2)

                self._object_rect = face
                self._object_detected = True
                break

        return bgr_image

    def _detect_head_direct(self, bgr_image):

        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(
            '/home/sciurus/Documents/data/shape_predictor_68_face_landmarks.dat')

        # BGR画像をグレー画像に変換
        gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 0)

        model_points = np.array([
            (0.0, 0.0, 0.0),             # Nose tip
            (0.0, -330.0, -65.0),        # Chin
            (-225.0, 170.0, -135.0),     # Left eye left corner
            (225.0, 170.0, -135.0),     # Right eye right corne
            (-150.0, -150.0, -125.0),    # Left Mouth corner
            (150.0, -150.0, -125.0)      # Right mouth corner
        ])

        size = bgr_image.shape
        focal_length = size[1]
        center = (size[1]/2, size[0]/2)
        camera_matrix = np.array([[focal_length, 0, center[0]], [
                                 0, focal_length, center[1]], [0, 0, 1]], dtype="double")

        self._object_detected = False

        for rect in rects:
            shape0 = predictor(gray, rect)
            shape0 = np.array(face_utils.shape_to_np(shape0))

        if len(rects) > 0:
            image_points = np.array([
                (shape0[30, :]),  # nose tip
                (shape0[8, :]),  # Chin
                (shape0[36, :]),  # Left eye left corner
                (shape0[45, :]),  # right eye right corner
                (shape0[48, :]),  # left mouth corner
                (shape0[54, :])  # right mouth corner
            ], dtype='double')

            dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
            (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points,
                                                                          image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

            (nose_end_point2D, jacobian) = cv2.projectPoints(
                np.array([(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)

            p1 = (int(image_points[0][0]), int(image_points[0][1]))
            p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
            cv2.line(gray, p1, p2, (255, 0, 0), 2)
            try:
                cv2.imshow('output', gray)
                cv2.waitKey(1)
            except Exception as err:
                print(err)

            self._object_target = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))

            self._object_detected = True

        return bgr_image


class NeckYawPitch(object):
    def __init__(self):
        self.__client = actionlib.SimpleActionClient("/sciurus17/controller3/neck_controller/follow_joint_trajectory",
                                                     FollowJointTrajectoryAction)
        self.__client.wait_for_server(rospy.Duration(5.0))
        if not self.__client.wait_for_server(rospy.Duration(5.0)):
            rospy.logerr("Action Server Not Found")
            rospy.signal_shutdown("Action Server not found")
            sys.exit(1)

        self._state_sub = rospy.Subscriber("/sciurus17/controller3/neck_controller/state",
                                           JointTrajectoryControllerState, self._state_callback, queue_size=1)

        self._state_received = False
        self._current_yaw = 0.0  # Degree
        self._current_pitch = 0.0  # Degree

    def _state_callback(self, state):
        # 首の現在角度を取得

        self._state_received = True
        yaw_radian = state.actual.positions[0]
        pitch_radian = state.actual.positions[1]

        self._current_yaw = math.degrees(yaw_radian)
        self._current_pitch = math.degrees(pitch_radian)

    def state_received(self):
        return self._state_received

    def get_current_yaw(self):
        return self._current_yaw

    def get_current_pitch(self):
        return self._current_pitch

    def set_angle(self, yaw_angle, pitch_angle, goal_secs=1.0e-9):
        # 首を指定角度に動かす
        goal = FollowJointTrajectoryGoal()
        goal.trajectory.joint_names = ["neck_yaw_joint", "neck_pitch_joint"]

        yawpoint = JointTrajectoryPoint()
        yawpoint.positions.append(yaw_angle)
        yawpoint.positions.append(pitch_angle)
        yawpoint.time_from_start = rospy.Duration(goal_secs)
        goal.trajectory.points.append(yawpoint)

        self.__client.send_goal(goal)
        self.__client.wait_for_result(rospy.Duration(0.1))
        return self.__client.get_result()


def hook_shutdown():
    # shutdown時に0度へ戻る
    neck.set_angle(math.radians(0), math.radians(0), 3.0)


def main():
    r = rospy.Rate(60)

    rospy.on_shutdown(hook_shutdown)

    # オブジェクト追跡のしきい値
    # 正規化された座標系(px, px)
    THRESH_X = 0.05
    THRESH_Y = 0.05

    # 首の初期角度 Degree
    INITIAL_YAW_ANGLE = 0
    INITIAL_PITCH_ANGLE = 0

    # 首の制御角度リミット値 Degree
    MAX_YAW_ANGLE = 120
    MIN_YAW_ANGLE = -120
    MAX_PITCH_ANGLE = 50
    MIN_PITCH_ANGLE = -70

    # 首の制御量
    # 値が大きいほど首を大きく動かす
    OPERATION_GAIN_X = 8.0  # 5
    OPERATION_GAIN_Y = 8.0

    # 初期角度に戻る時の制御角度 Degree
    RESET_OPERATION_ANGLE = 3

    # Horizontal and vertical visual field
    h_view = 65
    v_view = 40

    # 現在の首角度を取得する
    # ここで現在の首角度を取得することで、ゆっくり初期角度へ戻る
    while not neck.state_received():
        pass
    yaw_angle = neck.get_current_yaw()
    pitch_angle = neck.get_current_pitch()

    look_object = False
    detection_timestamp = rospy.Time.now()

    while not rospy.is_shutdown():
        # 正規化されたオブジェクトの座標を取得

        pos_x, pos_y = 0, 0
        n = 5
        for i in range(n):
            object_position = object_tracker.get_object_position()
            pos_x += object_position.x
            pos_y += object_position.y
            time.sleep(1)

        object_position.x = pos_x/n
        object_position.y = pos_y/n

        if object_tracker.object_detected():
            detection_timestamp = rospy.Time.now()
            look_object = True
        else:
            lost_time = rospy.Time.now() - detection_timestamp
            # 一定時間オブジェクトが見つからない場合は初期角度に戻る
            if lost_time.to_sec() > 1.0:
                look_object = False

        if look_object:
            # オブジェクトが画像中心にくるように首を動かす		＜ー首ガクガク問題
            if math.fabs(object_position.x) > THRESH_X:
                # quadlatic is better than linear
                yaw_angle += -object_position.x * math.fabs(object_position.x) * OPERATION_GAIN_X
                # yaw_angle += - 0.02 * math.degrees(math.atan(object_position.x * 2 * math.tan(math.radians(h_view/2))))

            if math.fabs(object_position.y) > THRESH_Y:
                pitch_angle += object_position.y * math.fabs(object_position.y) * OPERATION_GAIN_Y
                # pitch_angle += 0.02 * math.degrees(math.atan(object_position.y * 2 * math.tan(math.radians(v_view/2))))

            # 首の制御角度を制限する
            if yaw_angle > MAX_YAW_ANGLE:
                yaw_angle = MAX_YAW_ANGLE
            if yaw_angle < MIN_YAW_ANGLE:
                yaw_angle = MIN_YAW_ANGLE

            if pitch_angle > MAX_PITCH_ANGLE:
                pitch_angle = MAX_PITCH_ANGLE
            if pitch_angle < MIN_PITCH_ANGLE:
                pitch_angle = MIN_PITCH_ANGLE

        else:
            # ゆっくり初期角度へ戻る
            diff_yaw_angle = yaw_angle - INITIAL_YAW_ANGLE
            if math.fabs(diff_yaw_angle) > RESET_OPERATION_ANGLE:
                yaw_angle -= math.copysign(RESET_OPERATION_ANGLE, diff_yaw_angle)
            else:
                yaw_angle = INITIAL_YAW_ANGLE

            diff_pitch_angle = pitch_angle - INITIAL_PITCH_ANGLE
            if math.fabs(diff_pitch_angle) > RESET_OPERATION_ANGLE:
                pitch_angle -= math.copysign(RESET_OPERATION_ANGLE, diff_pitch_angle)
            else:
                pitch_angle = INITIAL_PITCH_ANGLE

        neck.set_angle(math.radians(yaw_angle), math.radians(pitch_angle))

        r.sleep()


if __name__ == '__main__':
    rospy.init_node("head_camera_tracking")

    neck = NeckYawPitch()
    object_tracker = ObjectTracker()

    main()
