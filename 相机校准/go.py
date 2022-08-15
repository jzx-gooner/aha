# -*- coding:utf-8 -*-  
__author__ = 'Microcosm'  
  
import cv2  
import numpy as np  
import glob  
  
# 设置寻找亚像素角点的参数，采用的停止准则是最大循环次数30和最大误差容限0.001  
criteria = (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 30, 0.001)  
  
# 获取标定板角点的位置
# 定义标定板 b_w=14   b_h= 14 内部  
b_w=14   
b_h=14

# 世界坐标系中的棋盘格点,例如(0,0,0), (1,0,0), (2,0,0) ....,(8,5,0)，去掉Z坐标，记为二维矩阵
objp = np.zeros((b_w*b_h,3), np.float32)  
objp[:,:2] = np.mgrid[0:b_h,0:b_w].T.reshape(-1,2) 
  
obj_points = []    # 存储3D点  
img_points = []    # 存储2D点  
  
images = glob.glob("./chess/*")  
for fname in images:
    img = cv2.imread(fname)  
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    size = gray.shape[::-1]  
    ret, corners = cv2.findChessboardCorners(gray, (b_w,b_h), None)  
    if ret:
        obj_points.append(objp)  
  
        corners2 = cv2.cornerSubPix(gray, corners, (5,5), (-1,-1), criteria)  # 在原角点的基础上寻找亚像素角点  
        if corners2.any():  
            img_points.append(corners2)  
        else:  
            img_points.append(corners)  
  
        cv2.drawChessboardCorners(img, (b_w,b_h), corners, ret)   
        cv2.imshow('img', img)  
        cv2.waitKey(1000)
        new_path = fname.replace("chess","results")
        cv2.imwrite(new_path,img)  
  
cv2.destroyAllWindows()  
  
# 标定
# 输入：世界坐标系里的位置 像素坐标 图像的像素尺寸大小 3*3矩阵，相机内参数矩阵 畸变矩阵
# 输出：标定结果 相机的内参数矩阵 畸变系数 旋转矩阵 平移向量  
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points,size, None, None)  
  
print("ret:",ret)  
print ("内参数矩阵:\n",mtx )       # 内参数矩阵  
print ("畸变系数:\n",dist)      # 畸变系数   distortion cofficients = (k_1,k_2,p_1,p_2,k_3)  
print ("旋转向量:\n",rvecs)    # 旋转向量  # 外参数  
print ("平移向量:\n",tvecs)    # 平移向量  # 外参数  
  
print("-----------------------------------------------------")  
# 畸变校正 
# 我们已经得到了相机内参和畸变系数，在将图像去畸变之前，
# 我们还可以使用cv.getOptimalNewCameraMatrix()优化内参数和畸变系数，
# 通过设定自由自由比例因子alpha。当alpha设为0的时候，
# 将会返回一个剪裁过的将去畸变后不想要的像素去掉的内参数和畸变系数；
# 当alpha设为1的时候，将会返回一个包含额外黑色像素点的内参数和畸变系数，并返回一个ROI用于将其剪裁掉

img = cv2.imread(images[1])
h, w = img.shape[:2]  
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))  
print (newcameramtx)  
print("------------------使用undistort函数-------------------")  
dst = cv2.undistort(img,mtx,dist,None,newcameramtx)  
x,y,w,h = roi  
dst1 = dst[y:y+h,x:x+w]  
cv2.imwrite('calibresult11.jpg', dst1)  
print ("方法一:dst的大小为:", dst1.shape)  
  
# undistort方法二  
# print("-------------------使用重映射的方式-----------------------")  
# mapx,mapy = cv2.initUndistortRectifyMap(mtx,dist,None,newcameramtx,(w,h),5)  # 获取映射方程  
# dst = cv2.remap(img,mapx,mapy,cv2.INTER_LINEAR)      # 重映射  
# dst = cv2.remap(img,mapx,mapy,cv2.INTER_CUBIC)        # 重映射后，图像变小了  
# x,y,w,h = roi  
# dst2 = dst[y:y+h,x:x+w]  
# cv2.imwrite('calibresult11_2.jpg', dst)  
# print ("方法二:dst的大小为:", dst.shape)        # 图像比方法一的小  
  
print("-------------------计算反向投影误差-----------------------")  
# 反投影误差
# 通过反投影误差，我们可以来评估结果的好坏。越接近0，说明结果越理想。
# 通过之前计算的内参数矩阵、畸变系数、旋转矩阵和平移向量，使用cv2.projectPoints()计算三维点到二维图像的投影，
# 然后计算反投影得到的点与图像上检测到的点的误差，最后计算一个对于所有标定图像的平均误差，这个值就是反投影误差。

tot_error = 0  
for i in range(len(obj_points)):  
    img_points2, _ = cv2.projectPoints(obj_points[i],rvecs[i],tvecs[i],mtx,dist)
    error = cv2.norm(img_points[i],img_points2, cv2.NORM_L2)/len(img_points2)  
    tot_error += error  
  
mean_error = tot_error/len(obj_points)  
print ("total error: ", tot_error)  
print ("mean error: ", mean_error)  
