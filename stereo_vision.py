"""
Author: smilealvin92
Time: 11/21/2019
Version: 1.0
Reference: Stephane Vujasinovic and Frederic Uhrweiller
"""
import numpy as np
import os
import cv2

# 应该要降低图片的分辨率，才能符合原项目的参数
# Filtering对于分辨率比较大的图像，是否要增大滤波器？
kernel = np.ones((3, 3), np.uint8)


class StereoVision:
    def __init__(self, corner_num_w, corner_num_h, frame_w,
                 frame_h, mouse_event, chess_img_dir, chess_img_num, camera_id):
        # Termination criteria
        # 设置寻找亚像素角点的参数，采用的停止准则是最大循环次数30和最大误差限度0.001
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        self.criteria_stereo = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        self.img_channel = 3
        self.corner_num_h = corner_num_h
        self.corner_num_w = corner_num_w
        self.frame_w = frame_w
        self.frame_h = frame_h
        self.mouse_event = mouse_event
        self.chess_img_dir = chess_img_dir
        self.chess_img_num = chess_img_num
        self.filter_disp = None
        self.camera_id = camera_id

    def prepare_chess_imgs(self):
        print("先准备合格的棋盘格标定图片")
        os.makedirs(self.chess_img_dir)
        capture = cv2.VideoCapture(self.camera_id)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_h)
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_w)
        if not capture.isOpened():
            exit(1)
        img_num = 0
        while img_num < self.chess_img_num:
            # Start Reading Camera images
            ret, frame = capture.read()
            if not ret:
                print("图像获取失败，请按照说明进行问题排查！")
                break
            key = cv2.waitKey(1)
            cv2.imshow('3DUsbCamera', frame)
            if key & 0xFF == ord('q'):
                print("程序正常退出，BYE！不要想我哦！")
                break
            elif key & 0xFF == ord('s'):
                frameL = frame[0:self.frame_h, 0:int(self.frame_w / 2)]
                frameR = frame[0:self.frame_h, int(self.frame_w / 2):self.frame_w]
            else:
                continue
            cv2.imshow('imgR', frameR)
            cv2.imshow('imgL', frameL)
            grayR = cv2.cvtColor(frameR, cv2.COLOR_BGR2GRAY)
            grayL = cv2.cvtColor(frameL, cv2.COLOR_BGR2GRAY)
            # Find the chess board corners
            # Define the number of chess corners (here 9 by 6),we are looking for with the right Camera
            # 第二个size是内部角点的size，不是棋盘格子的size
            # 棋盘格子的尺寸是10*7，但是，每个格子的具体长度，多少毫米，如何确定？
            retR, cornersR = cv2.findChessboardCorners(grayR, (self.corner_num_w, self.corner_num_h), None)
            retL, cornersL = cv2.findChessboardCorners(grayL, (self.corner_num_w, self.corner_num_h),
                                                       None)  # Same with the left camera
            # print(retL, retR)

            # 如果找到了指定size的角点，并且顺序是按照从左往右、从上往下逐行的排列的，则返回值ret非零，具体坐标存在cornersR中
            if retR & retL:
                # termination criteria
                # cornersR既作为输入也作为输出
                corners2R = cv2.cornerSubPix(grayR, cornersR, (11, 11), (-1, -1), self.criteria)  # Refining the Position
                corners2L = cv2.cornerSubPix(grayL, cornersL, (11, 11), (-1, -1), self.criteria)

                # Draw and display the corners
                cv2.drawChessboardCorners(grayR, (self.corner_num_w, self.corner_num_h), corners2R, retR)
                cv2.drawChessboardCorners(grayL, (self.corner_num_w, self.corner_num_h), corners2L, retL)
                cv2.imshow('VideoR', grayR)
                cv2.imshow('VideoL', grayL)
                # 每抓拍一次棋盘图像，整个过程要按两下s键
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                # Save the image in the file where this Programm is located
                cv2.imwrite(self.chess_img_dir + 'chessboard-R' + str(img_num) + '.png', frameR)
                cv2.imwrite(self.chess_img_dir + 'chessboard-L' + str(img_num) + '.png', frameL)
                print('Images' + str(img_num) + 'is saved')
                img_num += 1
        print("棋盘格标定图片准备完毕")

    def compute_stereo_map(self):
        """
        Prepare Parameters for Distortion Calibration
        :return: None
        """
        # Prepare object points
        # 获取标定板角点的位置
        objp = np.zeros((self.corner_num_w * self.corner_num_h, self.img_channel), np.float32)
        # 将世界坐标系建在标定板上，所有点的Z坐标全部为0，所以只需要赋值x和y
        objp[:, :2] = np.mgrid[0:self.corner_num_w, 0:self.corner_num_h].T.reshape(-1, 2)

        # Arrays to store object points and image points from all images
        objpoints = []  # 3d points in real world space
        imgpointsR = []  # 2d points in image plane
        imgpointsL = []

        # Start calibration from the camera
        print('Starting calibration for the 2 cameras... ')
        # 用的棋盘格图片对棋盘格子的具体尺寸没有要求
        # Call all saved images
        # Put the amount of pictures you have taken for the calibration in between range(0,?)
        # wenn starting from the image number 0
        if not os.path.exists(self.chess_img_dir):
            print("未准备好合格的、能够用来计算校正映射矩阵的棋盘格标定图片")
            self.prepare_chess_imgs()
        ChessImaR = cv2.imread(self.chess_img_dir + 'chessboard-R' + '0' + '.png', 0)  # Right side
        ChessImaL = cv2.imread(self.chess_img_dir + 'chessboard-L' + '0' + '.png', 0)  # Left side
        for i in range(0, self.chess_img_num):
            # 后面的参数0，表示是读取为灰度图
            ChessImaR = cv2.imread(self.chess_img_dir + 'chessboard-R' + str(i) + '.png', 0)  # Right side
            ChessImaL = cv2.imread(self.chess_img_dir + 'chessboard-L' + str(i) + '.png', 0)  # Left side
            # Define the number of chess corners we are looking for
            retR, cornersR = cv2.findChessboardCorners(ChessImaR,
                                                       (self.corner_num_w, self.corner_num_h),
                                                       None)
            # Left side
            retL, cornersL = cv2.findChessboardCorners(ChessImaL,
                                                       (self.corner_num_w, self.corner_num_h),
                                                       None)
            if retR & retL:
                # 添加67个objp
                objpoints.append(objp)
                # 亚像素精确化，在原角点的基础上寻找亚像素角点
                cv2.cornerSubPix(ChessImaR, cornersR, (11, 11), (-1, -1), self.criteria)
                cv2.cornerSubPix(ChessImaL, cornersL, (11, 11), (-1, -1), self.criteria)
                imgpointsR.append(cornersR)
                imgpointsL.append(cornersL)

        # Determine the new values for different parameters
        #   Right Side
        # 重新在图片中提取角点，再拿这些新的角点坐标去校正摄像头
        # shape那里是为了倒序输出，才写成[::-1]
        # objpoints是世界坐标系中的点坐标，imgpointsR是对应的图像特征点
        # 后面的两个None指代的分别是cameraMatrix（内参数矩阵）和distCoeffs（畸变矩阵）
        # 拍很多张图片就是等于建立很多个方程组，最后才能解出相机的这些未知参数
        # 返回的分别是标定结果，相机内参数矩阵，畸变系数，旋转矩阵，平移向量
        retR, mtxR, distR, rvecsR, tvecsR = cv2.calibrateCamera(objpoints,
                                                                imgpointsR,
                                                                ChessImaR.shape[::-1], None, None)
        # print("retR: ", retR)
        # 内参数矩阵就是相机矩阵，包括焦距(fx, fy)以及光学中心(cx, cy)，
        # 它应该是一个3*3的矩阵
        # 下面这些数据的单位是像素点
        # 初步计算得出，R的焦距(fx, fy)=(836.34566877, 834.75240275)
        # R的光学中心(704.87616892, 497.31647786)一旦计算出来，就可以保存起来，以备以后使用
        #
        # print("mtxR: ", mtxR)  # 内参数矩阵
        # 畸变系数 distortion cofficients = (k_1, k_2, p_1, p_2, k_3)
        # 包括径向畸变和切向畸变
        # print("distR: ", distR)
        # 下面两个都是外部参数，反映的是一个三维的点映射到二维的系统里
        # print("rvecsR: ", rvecsR)  # 旋转向量，外参数
        # print("tvecsR: ", tvecsR)  # 平移向量，外参数
        # hR, wR = ChessImaR.shape[:2]
        # 该函数能优化内参数矩阵和畸变系数，返回包含额外黑色像素点的内参数和畸变系数，还有一个ROI用于将其剪裁掉
        # 但是后续又没有使用这个OmtxR参数，先注释掉看看
        # OmtxR, roiR = cv2.getOptimalNewCameraMatrix(mtxR, distR,
        #                                             (wR, hR), 1, (wR, hR))

        #   Left Side，标定
        retL, mtxL, distL, rvecsL, tvecsL = cv2.calibrateCamera(objpoints,
                                                                imgpointsL,
                                                                ChessImaL.shape[::-1], None, None)
        # hL, wL = ChessImaL.shape[:2]
        # OmtxL, roiL = cv2.getOptimalNewCameraMatrix(mtxL, distL, (wL, hL), 1, (wL, hL))

        # print("retL: ", retL)
        # 内参数矩阵就是相机矩阵，包括焦距(fx, fy)以及光学中心(cx, cy)，
        # 它应该是一个3*3的矩阵
        # 初步计算得出，L的焦距(fx, fy)=(842.90630298, 841.86857972)
        # L的光学中心(718.04585971, 518.83188702)
        # print("mtxL: ", mtxL)  # 内参数矩阵
        # 畸变系数 distortion cofficients = (k_1, k_2, p_1, p_2, k_3)
        # 包括径向畸变和切向畸变
        # print("distL: ", distL)
        # 下面两个都是外部参数，反映的是一个三维的点映射到二维的系统里
        # print("rvecsL: ", rvecsL)  # 旋转向量，外参数
        # print("tvecsL: ", tvecsL)  # 平移向量，外参数
        print('Cameras Ready to use')
        # 上面是计算相机的各种参数，下面用这些参数来校正将要测量的图片，图片校正好之后
        # 才能送到立体视觉系统，进行视差的计算，视差图计算出来了，最后就能计算距离了

        # ********************************************
        # ***** Calibrate the Cameras for Stereo *****
        # ********************************************

        # StereoCalibrate function
        flags = 0
        flags |= cv2.CALIB_FIX_INTRINSIC
        # flags |= cv2.CALIB_FIX_PRINCIPAL_POINT
        # flags |= cv2.CALIB_USE_INTRINSIC_GUESS
        # flags |= cv2.CALIB_FIX_FOCAL_LENGTH
        # flags |= cv2.CALIB_FIX_ASPECT_RATIO
        # flags |= cv2.CALIB_ZERO_TANGENT_DIST
        # flags |= cv2.CALIB_RATIONAL_MODEL
        # flags |= cv2.CALIB_SAME_FOCAL_LENGTH
        # flags |= cv2.CALIB_FIX_K3
        # flags |= cv2.CALIB_FIX_K4
        # flags |= cv2.CALIB_FIX_K5
        # mtxL是cameraMatrix1，distL是distCoeffs1，
        # mtxR是cameraMatrix2，distR是distCoeffs2，
        # 得到的R是两个摄像头之间的旋转矩阵，T是两个摄像头之间的平移矩阵
        retS, MLS, dLS, MRS, dRS, R, T, E, F = cv2.stereoCalibrate(objpoints,
                                                                   imgpointsL,
                                                                   imgpointsR,
                                                                   mtxL,
                                                                   distL,
                                                                   mtxR,
                                                                   distR,
                                                                   ChessImaR.shape[::-1],
                                                                   self.criteria_stereo,
                                                                   flags)

        # StereoRectify function
        rectify_scale = 0  # if 0 image croped, if 1 image nor croped
        # 该函数的作用是为每个摄像头计算立体校正的映射矩阵，所以其运行结果并不是直接将图片进行立体矫正，
        # 而是得出进行立体矫正所需要的映射矩阵
        # 立体极线校正
        RL, RR, PL, PR, Q, roiL, roiR = cv2.stereoRectify(MLS, dLS, MRS, dRS,
                                                          ChessImaR.shape[::-1], R, T,
                                                          rectify_scale,
                                                          (0,
                                                           0))  # last paramater is alpha, if 0= croped, if 1= not croped
        # initUndistortRectifyMap function
        # cv2.CV_16SC2 this format enables us the programme to work faster
        # 对整幅图像去除畸变，生成映射矩阵
        Left_Map = cv2.initUndistortRectifyMap(MLS, dLS, RL, PL,
                                               ChessImaR.shape[::-1],
                                               cv2.CV_16SC2)
        Right_Map = cv2.initUndistortRectifyMap(MRS, dRS, RR, PR,
                                                ChessImaR.shape[::-1], cv2.CV_16SC2)
        np.savez("left_map.npz", left_map_0=Left_Map[0], left_map_1=Left_Map[1])
        np.savez("right_map.npz", right_map_0=Right_Map[0], right_map_1=Right_Map[1])
        print("校正映射矩阵计算完毕，已保存")

    def compute_distance(self):
        if (not os.path.exists("left_map.npz")) or (not os.path.exists("right_map.npz")):
            print("无法进行畸变校正，先计算校正映射矩阵")
            self.compute_stereo_map()
        left_map = np.load("left_map.npz")
        right_map = np.load("right_map.npz")
        Left_Stereo_Map = (left_map['left_map_0'], left_map['left_map_1'])
        Right_Stereo_Map = (right_map['right_map_0'], right_map['right_map_1'])

        # Create StereoSGBM and prepare all parameters
        window_size = 3
        # 正常来说，min_disp应该是0
        min_disp = 0
        num_disp = 128 - min_disp
        # 用SGBM算法获取视差图，即景深图
        # StereoSGBM的速度比StereoBM慢，但是精度更高，准确性更好
        # 下面的这些参数都是可以调节的，都是超参数，要做实验，以便确定最佳参数，根据具体的摄像机来确定
        # numDisparities必须要能被16整除
        # blockSize是matched block size，它应该为一个奇数，大部分情况下，它在3到11之间
        # P1和P2控制disparity smoothness
        # speckleRange一般来说，1或者2就足够好了
        stereo = cv2.StereoSGBM_create(minDisparity=min_disp,
                                       numDisparities=num_disp,
                                       blockSize=window_size,
                                       uniquenessRatio=10,
                                       speckleWindowSize=100,
                                       speckleRange=1,
                                       disp12MaxDiff=10,
                                       P1=8 * 3 * window_size ** 2,
                                       P2=32 * 3 * window_size ** 2)

        # Used for the filtered image
        stereoR = cv2.ximgproc.createRightMatcher(stereo)  # Create another stereo for right this time

        # WLS FILTER Parameters
        lmbda = 8000
        sigma = 2.0
        # visual_multiplier = 1.0

        wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=stereo)
        # 较大的lambda使得filter_img和原图的轮廓更为一致，通常值为8000
        # 较小的sigma使得视差对图片的纹理、噪音更为敏感，通常在0.8到2.0之间
        wls_filter.setLambda(lmbda)
        wls_filter.setSigmaColor(sigma)

        # *************************************
        # ***** Starting the StereoVision *****
        # *************************************

        capture = cv2.VideoCapture(self.camera_id)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_h)
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_w)
        if not capture.isOpened():
            exit(1)
        while True:
            # Start Reading Camera images
            ret, frame = capture.read()
            if not ret:
                print("图像获取失败，请按照说明进行问题排查！")
                break
            key = cv2.waitKey(1)
            cv2.imshow('3DUsbCamera', frame)
            if key & 0xFF == ord('q'):
                print("程序正常退出，BYE！不要想我哦！")
                break
            elif key & 0xFF == ord('s'):
                frameL = frame[0:self.frame_h, 0:int(self.frame_w/2)]
                frameR = frame[0:self.frame_h, int(self.frame_w/2):self.frame_w]
            else:
                continue
                # frameL = frame[0:480, 0:640]
                # frameR = frame[0:480, 640:1280]

            # Rectify the images on rotation and alignement
            # 重映射，就是把一幅图像中某位置的像素放置到另一个图片指定位置的过程
            # Rectify the image using the calibration parameters founds during the initialisation
            # Left_Stereo_Map[0]表示新图像该处的来源图像像素点的X坐标是什么
            # Left_Stereo_Map[1]表示新图像该处的来源图像像素点的Y坐标是什么
            Left_nice = cv2.remap(frameL, Left_Stereo_Map[0], Left_Stereo_Map[1], cv2.INTER_LANCZOS4,
                                  cv2.BORDER_CONSTANT,
                                  0)
            Right_nice = cv2.remap(frameR, Right_Stereo_Map[0], Right_Stereo_Map[1], cv2.INTER_LANCZOS4,
                                   cv2.BORDER_CONSTANT, 0)

            ##    # Draw Red lines
            ##    for line in range(0, int(Right_nice.shape[0]/20)): # Draw the Lines on the images Then numer of line is defines by the image Size/20
            ##        Left_nice[line*20,:]= (0,0,255)
            ##        Right_nice[line*20,:]= (0,0,255)
            ##
            ##    for line in range(0, int(frameR.shape[0]/20)): # Draw the Lines on the images Then numer of line is defines by the image Size/20
            ##        frameL[line*20,:]= (0,255,0)
            ##        frameR[line*20,:]= (0,255,0)

            # Show the Undistorted images
            # cv2.imshow('Both Images', np.hstack([Left_nice, Right_nice]))
            # cv2.imshow('Normal', np.hstack([frameL, frameR]))

            # Convert from color(BGR) to gray
            grayR = cv2.cvtColor(Right_nice, cv2.COLOR_BGR2GRAY)
            grayL = cv2.cvtColor(Left_nice, cv2.COLOR_BGR2GRAY)

            # Compute the 2 images for the Depth_image
            # 上面校正完之后，再计算视差
            # 计算视差？
            disp = stereo.compute(grayL, grayR)  # .astype(np.float32)/ 16
            dispL = np.int16(disp)
            # dispL = disp

            dispR = stereoR.compute(grayR, grayL)
            # cv2.imshow('dispR', dispR)
            dispR = np.int16(dispR)

            # Using the WLS filter
            filteredImg = wls_filter.filter(dispL, grayL, None, dispR)
            # cv2.imshow('filteredImg', filteredImg)

            filteredImg = cv2.normalize(src=filteredImg, dst=filteredImg,
                                        beta=0, alpha=255, norm_type=cv2.NORM_MINMAX)
            filteredImg = np.uint8(filteredImg)
            # for y_f in range(0, 480):
            #     for x_f in range(0, 640):
            #         if filteredImg[y_f, x_f]:
            #             print(x_f)
            #             break
            self.filter_disp = ((filteredImg.astype(np.float32) / 16) - min_disp) / num_disp
            filt_Color = cv2.applyColorMap(filteredImg, cv2.COLORMAP_OCEAN)
            # cv2.imshow('Disparity Map', filteredImg)
            # Calculation allowing us to have 0 for the most distant object able to detect
            # disp = ((disp.astype(np.float32) / 16) - min_disp) / num_disp

            ##    # Resize the image for faster executions
            ##    dispR= cv2.resize(disp,None,fx=0.7, fy=0.7, interpolation = cv2.INTER_AREA)
            # Ex是闭运算
            # closed = cv2.morphologyEx(disp, cv2.MORPH_CLOSE, kernel)
            # Colors map
            # dispc = (closed - closed.min()) * 255
            # Convert the type of the matrix from float32 to uint8,
            # this way you can show the results with the function cv2.imshow()
            # dispC = dispc.astype(np.uint8)
            # Change the Color of the Picture into an Ocean Color_Map
            # 伪彩色
            # disp_color后面未使用
            # disp_Color = cv2.applyColorMap(dispC, cv2.COLORMAP_OCEAN)
            # disp和filt_Color，前者用来计算，后者用来给用户看

            # Show the result for the Depth_image
            # 应该看看关运算的结果
            # cv2.imshow('Disparity', disp)
            # cv2.imshow('Closed', closed)
            # cv2.imshow('Color Depth', disp_Color)
            cv2.imshow('Filtered Color Depth', filt_Color)
            # 蓝色右侧为可测距离的范围
            Left_nice_line = cv2.line(Left_nice, (128, 0), (128, self.frame_h), (0, 255, 0), 1)
            cv2.imshow('Left_nice_line', Left_nice_line)
            # Mouse click
            # 鼠标双击点哪里，就打印出哪里的距离
            cv2.setMouseCallback("Left_nice_line", self.coords_mouse_disp, Left_nice_line)
        capture.release()
        cv2.destroyAllWindows()

    def coords_mouse_disp(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDBLCLK:
            # 打印出来是有道理的
            print("filter_disp[y,x]: ", self.filter_disp[y, x])
            # print("差别： ", disp[y, x], fi[y, x])
            # print(x, y, disp[y, x], filteredImg[y, x])
            average = 0
            for u in range(-1, 2):
                for v in range(-1, 2):
                    average += self.filter_disp[y + u, x + v]
                    # average += fi[y + u, x + v]
            average = average / 9
            # This equation converts the values from
            # the disparity map to a distance.
            # 下面这个公式，是回归得来的，即事先准备好很多组数据（已知了视差和准确的距离）
            # 最后回归得到各个参数，成为准确的计算函数
            # 一开始，距离的单位是分米
            # 旧的公式是依照disp计算得来的，新公式是依照filter_disp计算得来的
            # Distance= -16.22*average**3 + 40.929*average**2 - 37.258*average + 14.365
            Distance = -15137 * average ** 3 + 4620.2 * average ** 2 - 494.75 * average + 21.176
            Distance = np.around(Distance * 0.01, decimals=3)
            print('Distance: ' + str(Distance) + ' m')


if __name__ == '__main__':
    stereo_vision = StereoVision(corner_num_w=9, corner_num_h=6, frame_w=1280,
                                 frame_h=480, mouse_event=cv2.EVENT_LBUTTONDBLCLK,
                                 chess_img_dir="./chessboard_1/", chess_img_num=67, camera_id=1)
    # 按s键就会对当前图片进行距离计算，并弹出图片窗口，在Left_nice_line窗口的蓝线右侧，均可双击测距，距离会打印出来，
    # Filtered Color Depth窗口为视差图
    stereo_vision.compute_distance()

