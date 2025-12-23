# coding=utf-8
"""
handeye_check_error_live.py

用途：
  - 在相机视野中选择（或自动检测）一个标定点，计算该点在机器人基坐标系下的位置（使用手眼标定矩阵）。
  - 移动机器人让末端触碰该点，记录末端位姿，比较两者并输出误差向量与距离。
  - 使用 result.txt 中的手眼标定矩阵（T_cam^ee），再结合用户输入的末端位姿计算 T_cam^base。

主要键位：
  - 'r' : 输入当前机械臂末端位姿（用于计算 cam→base）
  - 'c' : 捕获/选择目标点（若检测到棋盘角点，会列出并选第一个角；否则切换到鼠标点击模式）
  - 鼠标左键单击：在鼠标模式下选择像素点
  - 't' : 记录“触碰”时的机械臂位姿
  - 's' : 保存当前记录（图片、点、误差）
  - 'q' : 退出程序
"""

import os
import math
import time
import logging
import numpy as np
import cv2
import pyrealsense2 as rs

# --------------- 配置 ---------------
OUTPUT_DIR = "handeye_data_20p_12X9"
RESULT_FILE = os.path.join(OUTPUT_DIR, "result.txt")  # 手眼标定矩阵 T_cam^ee
WINDOW_WIDTH = 640
WINDOW_HEIGHT = 480
ERROR_FILE = os.path.join(OUTPUT_DIR, "errors.txt")

os.makedirs(OUTPUT_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s: %(message)s")
logger = logging.getLogger("handeye_check_live")

# ----------------- 工具函数 -----------------
def load_matrix_from_file(path):
    """从文本文件读取 4x4 矩阵"""
    if not os.path.exists(path):
        return None
    try:
        arr = np.loadtxt(path, dtype=np.float64)
        if arr.shape != (4,4):
            logger.warning(f"{path} 矩阵尺寸非 4x4")
            return None
        return arr
    except Exception as e:
        logger.error(f"读取矩阵失败: {e}")
        return None

def euler_xyz_to_matrix(rx_deg, ry_deg, rz_deg):
    rx = math.radians(rx_deg)
    ry = math.radians(ry_deg)
    rz = math.radians(rz_deg)
    Rx = np.array([[1,0,0],[0,math.cos(rx),-math.sin(rx)],[0,math.sin(rx),math.cos(rx)]])
    Ry = np.array([[math.cos(ry),0,math.sin(ry)],[0,1,0],[-math.sin(ry),0,math.cos(ry)]])
    Rz = np.array([[math.cos(rz),-math.sin(rz),0],[math.sin(rz),math.cos(rz),0],[0,0,1]])
    return Rz @ Ry @ Rx

def pose_to_transform(pose):
    x,y,z,rx,ry,rz = pose
    R = euler_xyz_to_matrix(math.degrees(rx), math.degrees(ry), math.degrees(rz))
    T = np.eye(4)
    T[:3,:3] = R
    T[:3,3] = [x,y,z]
    return T

def parse_pose_input_line(line):
    s = line.strip().replace("[","").replace("]","").replace(" ","")
    if "," in s:
        parts = [p for p in s.split(",") if p!=""]
    else:
        parts = [p for p in s.split() if p!=""]
    if len(parts)!=6:
        raise ValueError("位姿输入应包含 6 个数")
    vals = [float(p) for p in parts]
    x, y, z = vals[0]/1000.0, vals[1]/1000.0, vals[2]/1000.0
    rx, ry, rz = math.radians(vals[3]), math.radians(vals[4]), math.radians(vals[5])
    return [x, y, z, rx, ry, rz]

def transform_point(T, p):
    p4 = np.ones(4)
    p4[:3] = p
    p_base = T @ p4
    return p_base[:3]

def get_depth_scale(profile):
    depth_sensor = profile.get_device().first_depth_sensor()
    return depth_sensor.get_depth_scale()

def deproject_pixel_to_point(intr, pixel, depth_m):
    x = (pixel[0]-intr.ppx)*depth_m/intr.fx
    y = (pixel[1]-intr.ppy)*depth_m/intr.fy
    z = depth_m
    return np.array([x,y,z])

# ----------------- 主功能 -----------------
def run():
    # 读取手眼标定矩阵 T_cam^ee
    cam2ee = load_matrix_from_file(RESULT_FILE)
    if cam2ee is None:
        logger.error("未找到 result.txt 或格式错误")
        return
    logger.info(f"Loaded T_cam^ee from {RESULT_FILE}")

    # Realsense pipeline
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, WINDOW_WIDTH, WINDOW_HEIGHT, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, WINDOW_WIDTH, WINDOW_HEIGHT, rs.format.z16, 30)
    try:
        profile = pipeline.start(config)
    except Exception as e:
        logger.error(f"相机连接异常: {e}")
        return

    depth_scale = get_depth_scale(profile)
    color_stream = profile.get_stream(rs.stream.color)
    color_intr = color_stream.as_video_stream_profile().get_intrinsics()

    target_pixel = None
    target_point_cam = None
    target_point_base = None
    mouse_mode = False
    clicked_pixel = None
    T_base_ee = None  # 用户输入末端位姿

    def on_mouse(event, x, y, flags, param):
        nonlocal clicked_pixel
        if event == cv2.EVENT_LBUTTONDOWN:
            clicked_pixel = (int(x), int(y))
            logger.info(f"Mouse clicked at {clicked_pixel}")

    cv2.namedWindow("handeye_check")
    cv2.setMouseCallback("handeye_check", on_mouse)
    logger.info("按 'r' 输入末端位姿，'c' 捕获目标点，'t' 触碰点，'s' 保存，'q' 退出")

    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if not color_frame or not depth_frame:
                continue
            color_image = np.asanyarray(color_frame.get_data())
            display_img = cv2.resize(color_image, (WINDOW_WIDTH, WINDOW_HEIGHT), interpolation=cv2.INTER_AREA)
            vis = display_img.copy()

            # 显示鼠标点击或已选点
            if clicked_pixel: cv2.circle(vis, clicked_pixel,1,(0,0,255),-1)
            if target_pixel: cv2.circle(vis, target_pixel,2,(255,0,0),1)

            cv2.imshow("handeye_check", vis)
            k = cv2.waitKey(1) & 0xFF

            if k==ord('c'):
                if clicked_pixel:
                    target_pixel = clicked_pixel
                u,v = target_pixel
                depth = depth_frame.get_distance(u,v)
                if depth==0 or math.isnan(depth):
                    logger.warning("无效深度")
                    continue
                target_point_cam = deproject_pixel_to_point(color_intr, (u,v), depth)
                if T_base_ee is not None:
                    cam2base = T_base_ee @ cam2ee
                    target_point_base = transform_point(cam2base, target_point_cam)
                    logger.info(f"Target point in base frame: {target_point_base} m")
                else:
                    target_point_base = None
                # timestamp = int(time.time())
                # cv2.imwrite(os.path.join(OUTPUT_DIR,f"capture_{timestamp}.jpg"), display_img)
                # logger.info("截图已保存")
                logger.info(f"标记已捕获")

            elif k==ord('r'):
                try:
                    user = input("请输入当前机械臂末端位姿 [x,y,z mm, Rx,Ry,Rz deg]:\n")
                    parsed = parse_pose_input_line(user)
                    T_base_ee = pose_to_transform(parsed)
                    logger.info("已更新 T_base^ee，用于计算 T_cam^base")
                except Exception as e:
                    logger.exception(f"解析位姿失败: {e}")

            elif k==ord('t'):
                if target_point_base is None:
                    logger.warning("目标点基坐标未知，请先按 'c' 并输入末端位姿")
                    continue
                try:
                    user = input("请输入触碰时末端位姿 [x,y,z mm, Rx,Ry,Rz deg]:\n")
                    parsed_touch = parse_pose_input_line(user)
                    T_base_ee_touch = pose_to_transform(parsed_touch)
                    touch_point_base = T_base_ee_touch[:3,3]
                    err_vec = touch_point_base - target_point_base
                    err_dist = np.linalg.norm(err_vec)

                    # 将误差向量单位转换为厘米
                    err_vec = err_vec * 100
                    # 将误差距离单位转换为厘米
                    err_dist = err_dist * 100

                    logger.info(f"误差向量 (cm): {err_vec}, 距离 (cm): {err_dist:.6f}")
                    with open(ERROR_FILE,"a+") as ef:
                        ef.write(f"time {int(time.time())} target {target_point_base.tolist()} touch {touch_point_base.tolist()} err_vec {err_vec.tolist()} err_dist {err_dist:.6f}\n")
                except Exception as e:
                    logger.exception(f"触碰位姿解析失败: {e}")

            elif k==ord('s'):
                if target_point_cam is not None:
                    ts = int(time.time())
                    fn = os.path.join(OUTPUT_DIR,f"target_{ts}.txt")
                    with open(fn,"w") as f:
                        f.write("target_cam_m " + " ".join(f"{v:.6f}" for v in target_point_cam.tolist()) + "\n")
                        if target_point_base is not None:
                            f.write("target_base_m " + " ".join(f"{v:.6f}" for v in target_point_base.tolist()) + "\n")
                    logger.info(f"已保存 target points: {fn}")

            elif k==ord('q'):
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__=="__main__":
    run()
