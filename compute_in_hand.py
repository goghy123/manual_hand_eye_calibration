# coding=utf-8
"""
手眼标定计算脚本（匹配手动采集脚本）
使用采集的图片和机械臂末端位姿计算相机相对于末端的旋转矩阵、平移向量和齐次矩阵
"""
import os
import logging
import yaml
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

np.set_printoptions(precision=8, suppress=True)

# 日志
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s: %(message)s")
logger = logging.getLogger("handeye_calc")

# ================== 配置 ==================
current_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "handeye_data_20p_12X9")

# 找最近一次的数据文件夹
def find_latest_data_folder(path):
    folders = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path,f))]
    folders.sort()
    return folders[-1] if folders else ""

latest_folder = find_latest_data_folder(current_path)
images_path = os.path.join(current_path, latest_folder)
file_path = os.path.join(images_path, "poses.txt")  # 机械臂末端位姿文件

# 读取棋盘格参数
with open("config.yaml", 'r', encoding='utf-8') as f:
    data = yaml.safe_load(f)

XX = data["checkerboard_args"]["XX"]
YY = data["checkerboard_args"]["YY"]
L = data["checkerboard_args"]["L"]  # 单位：米
# ========================================


def calibrate_hand_eye():
    # 亚像素角点优化参数
    criteria = (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 30, 0.001)

    # 棋盘格世界坐标
    objp = np.zeros((XX * YY, 3), np.float32)
    objp[:, :2] = np.mgrid[0:XX, 0:YY].T.reshape(-1, 2)
    objp *= L

    obj_points = []  # 3D点
    img_points = []  # 2D点

    # 获取图片文件，按数字排序
    images_num = sorted([f for f in os.listdir(images_path) if f.endswith('.jpg')],
                        key=lambda x: int(os.path.splitext(x)[0]))

    size = None

    for image_file in images_num:
        image_path = os.path.join(images_path, image_file)
        img = cv2.imread(image_path)
        if img is None:
            logger.warning(f"无法读取图片: {image_file}")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        size = gray.shape[::-1]

        ret, corners = cv2.findChessboardCorners(gray, (XX, YY), None)
        if ret:
            obj_points.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
            img_points.append(corners2 if corners2 is not None else corners)
        else:
            logger.warning(f"棋盘格角点未找到: {image_file}")

    N = len(img_points)
    if N == 0:
        logger.error("没有找到任何棋盘格角点，无法标定")
        return

    # 相机内参标定，得到相机坐标系下标定板位姿
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, size, None, None)
    logger.info(f"标定完成: {N} 张图片")

    # 读取末端位姿
    poses = np.loadtxt(file_path, delimiter=',')  # 每行: X,Y,Z,Rx,Ry,Rz
    if poses.shape[0] != N:
        logger.warning(f"图片数量({N})与位姿行数({poses.shape[0]})不一致，请检查数据顺序")

    R_tool = []
    t_tool = []

    for pose in poses:
        x, y, z, rx, ry, rz = pose
        rot_mat = R.from_euler('xyz', [rx, ry, rz]).as_matrix()
        R_tool.append(rot_mat)
        t_tool.append(np.array([x, y, z]).reshape(3, 1))

    # 手眼标定
    R_cam, t_cam = cv2.calibrateHandEye(R_tool, t_tool, rvecs, tvecs, cv2.CALIB_HAND_EYE_TSAI)

    # 四元数
    quat = R.from_matrix(R_cam).as_quat()
    x, y, z = t_cam.flatten()

    logger.info(f"\n旋转矩阵:\n{R_cam}")
    logger.info(f"\n平移向量:\n{t_cam}")
    logger.info(f"\n四元数:\n{quat}")

    # ===== 输出齐次矩阵 =====
    H_cam = np.eye(4)
    H_cam[:3, :3] = R_cam
    H_cam[:3, 3] = t_cam.flatten()
    logger.info(f"\n齐次矩阵 (4x4):\n{H_cam}")

    # ===== 新增：保存齐次矩阵到文本文件 =====
    output_file = os.path.join(images_path, "result.txt")

    np.savetxt(output_file, H_cam, fmt="%.8f")  # 不加 header
    logger.info(f"齐次矩阵已保存到: {output_file}")

    return R_cam, t_cam, quat, H_cam


if __name__ == "__main__":
    calibrate_hand_eye()
