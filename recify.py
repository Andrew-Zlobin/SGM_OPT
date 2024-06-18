import cv2
import numpy as np

def undistort_image(image, camera_matrix, dist_coeffs):
    """Коррекция дисторсии изображения"""
    undistorted_image = cv2.undistort(image, camera_matrix, dist_coeffs)
    return undistorted_image

def rectify_images(img1, img2, camera_matrix1, dist_coeffs1, camera_matrix2, dist_coeffs2, R, T):
    """Ректификация изображений стереокамеры"""
    # Вычисление ректификационных преобразований и матриц проекции
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(camera_matrix1, dist_coeffs1,
                                                 camera_matrix2, dist_coeffs2,
                                                 img1.shape[:2], R, T)

    # Вычисление матриц для преобразования изображений
    map1x, map1y = cv2.initUndistortRectifyMap(camera_matrix1, dist_coeffs1, R1, P1, img1.shape[:2], cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(camera_matrix2, dist_coeffs2, R2, P2, img2.shape[:2], cv2.CV_32FC1)

    # Преобразование изображений
    rectified_img1 = cv2.remap(img1, map1x, map1y, cv2.INTER_LINEAR)
    rectified_img2 = cv2.remap(img2, map2x, map2y, cv2.INTER_LINEAR)

    return rectified_img1, rectified_img2

def calibrate_camera(images, chessboard_size, square_size):

    # Подготовка объектов (углов) и изображений
    obj_points = []
    img_points = []

    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
    objp *= square_size

    for img_path in images:
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Найти углы шахматной доски
        ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

        if ret:
            img_points.append(corners)
            obj_points.append(objp)

            # Отобразить углы
            cv2.drawChessboardCorners(img, chessboard_size, corners, ret)
    #         cv2.imshow('img', img)
    #         cv2.waitKey(500)

    # cv2.destroyAllWindows()

    # Калибровка камеры
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None)

    return camera_matrix, dist_coeffs

def stereo_calibrate(left_images, right_images, chessboard_size, square_size, camera_matrix1, dist_coeffs1, camera_matrix2, dist_coeffs2):

    # Подготовка объектов (углов) и изображений
    obj_points = []
    img_points1 = []
    img_points2 = []

    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
    objp *= square_size

    for left_img_path, right_img_path in zip(left_images, right_images):
        img1 = cv2.imread(left_img_path)
        img2 = cv2.imread(right_img_path)
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        # Найти углы шахматной доски
        ret1, corners1 = cv2.findChessboardCorners(gray1, chessboard_size, None)
        ret2, corners2 = cv2.findChessboardCorners(gray2, chessboard_size, None)

        if ret1 and ret2:
            img_points1.append(corners1)
            img_points2.append(corners2)
            obj_points.append(objp)

            # Отобразить углы
            cv2.drawChessboardCorners(img1, chessboard_size, corners1, ret1)
            cv2.drawChessboardCorners(img2, chessboard_size, corners2, ret2)
    #         cv2.imshow('left_img', img1)
    #         cv2.imshow('right_img', img2)
    #         cv2.waitKey(500)

    # cv2.destroyAllWindows()

    # Стереокалибровка
    flags = cv2.CALIB_FIX_INTRINSIC
    ret, camera_matrix1, dist_coeffs1, camera_matrix2, dist_coeffs2, R, T, E, F = cv2.stereoCalibrate(
        obj_points, img_points1, img_points2, camera_matrix1, dist_coeffs1, camera_matrix2, dist_coeffs2, gray1.shape[::-1], flags=flags)

    return R, T