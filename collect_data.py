# coding=utf-8
import os
import sys
import cv2
import numpy as np
import logging
import math
import pyrealsense2 as rs

# ========== 配置 ==========
OUTPUT_DIR = "handeye_data_21p_12X9"  # 存放图片和位姿文件
WINDOW_WIDTH = 640           # 固定显示窗口宽
WINDOW_HEIGHT = 480          # 固定显示窗口高
POSE_FILE = os.path.join(OUTPUT_DIR, "poses.txt")
# ==========================

# 日志
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s: %(message)s")
logger = logging.getLogger("handeye")

# 初始化数据目录
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ========== 快捷输入位姿 ==========
def input_robot_pose(current_index):
    """
    快速输入机器人末端位姿
    直接粘贴一行: X,Y,Z,Rx,Ry,Rz
    X,Y,Z 单位:mm，Rx,Ry,Rz 单位:deg
    自动去掉空格/括号/多余字符，并转换为 m 和 rad
    显示当前编号方便对应
    """
    while True:
        user_input = input(
            f"请输入第 {current_index} 个机械臂末端位姿 "
            "(X,Y,Z mm; Rx,Ry,Rz deg，用逗号分隔，支持带中括号)：\n"
        )
        try:
            # 去掉空格、换行符和括号
            cleaned_input = (
                user_input.replace(" ", "")
                .replace("\n", "")
                .replace("[", "")
                .replace("]", "")
            )
            values = [float(v) for v in cleaned_input.split(",")]
            if len(values) != 6:
                raise ValueError
            # 单位转换
            x, y, z = values[0]/1000.0, values[1]/1000.0, values[2]/1000.0
            rx, ry, rz = math.radians(values[3]), math.radians(values[4]), math.radians(values[5])
            return [x, y, z, rx, ry, rz]
        except ValueError:
            print("输入格式错误，请确保一行包含 6 个数字，用逗号分隔（可带括号）")


# ========== 相机显示与采集 ==========
def displayD435():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, WINDOW_WIDTH, WINDOW_HEIGHT, rs.format.bgr8, 30)

    try:
        pipeline.start(config)
    except Exception as e:
        logger.error(f"相机连接异常：{e}")
        sys.exit(1)

    # 自动从已有 poses.txt 判断起始编号
    if os.path.exists(POSE_FILE):
        with open(POSE_FILE, "r") as f:
            lines = f.readlines()
            count = len(lines) + 1
    else:
        count = 1

    logger.info("开始手眼标定程序（快捷输入机械臂位姿），版本 V2.0.0")

    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())
            # 固定窗口显示
            cv_img = cv2.resize(color_image, (WINDOW_WIDTH, WINDOW_HEIGHT), interpolation=cv2.INTER_AREA)
            cv2.imshow("Capture_Video", cv_img)

            k = cv2.waitKey(30) & 0xFF
            if k == ord('s'):  # 按下 s 键保存
                pose = input_robot_pose(count)
                # 保存位姿
                with open(POSE_FILE, "a+") as f:
                    pose_str = ",".join(f"{v:.6f}" for v in pose)
                    f.write(f"{pose_str}\n")

                # 保存图像
                image_path = os.path.join(OUTPUT_DIR, f"{count}.jpg")
                cv2.imwrite(image_path, cv_img)

                logger.info(f"=== 采集第 {count} 次数据完成 ===")
                count += 1

            elif k == ord('q'):  # 按下 q 退出
                logger.info("退出采集程序")
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


# ========== 主程序 ==========
if __name__ == "__main__":
    displayD435()
