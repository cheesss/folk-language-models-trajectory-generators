#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import pyrealsense2 as rs
from PIL import Image

# ----------------------------
# box_dimensioner_multicam_demo.py 관련 모듈
# (동일 디렉터리에 존재한다고 가정)
# ----------------------------
from realsense_device_manager import DeviceManager
from calibration_kabsch import PoseEstimation
from helper_functions import get_boundary_corners_2D, get_clipped_pointcloud
# measurement_task.py에 minAreaRect 기반 bbox 계산 함수가 있다면 import
# 여기서는 예시로 measure_box_minAreaRect, project_3d_to_color, draw_3d_box를 직접 정의

# ----------------------------
# LangSAM (동일 디렉터리 내 lang_sam 폴더)
# ----------------------------
from lang_sam import LangSAM

# ----------------------------
# 전역 파라미터
# ----------------------------
RES_WIDTH = 1280
RES_HEIGHT = 720
FPS = 15

CHESSBOARD_WIDTH = 6
CHESSBOARD_HEIGHT = 9
SQUARE_SIZE = 0.0253  # 체스보드 한 칸 크기 (미터)

DISPOSE_FRAMES = 30   # 초기 안정화용
DEPTH_SCALE = 0.001   # 실제 카메라의 depth 스케일

# ---------------------------------------------------------
# 3D BBox 계산 함수 (XY 평면 minAreaRect + z-range)
# 이 함수는 XY 평면에서 “가장 작은 회전 사각형”(cv2.minAreaRect)을 구하고, Z축 범위로 높이를 정의하여 3D 직육면체를 구성하는 단순·효율적인 방식
# ---------------------------------------------------------
def measure_box_minAreaRect(points_world):
    """
    points_world: (3, N), 월드 좌표계 3D 점
    Returns:
      corners_3d: (8,3) float array (아래 4점 + 위 4점)
      (width, length, height)
    """
    if points_world.shape[1] < 10:
        return None, (0,0,0)

    pts_xy = points_world[:2, :].T.astype(np.float32)  # (N,2)
    rect = cv2.minAreaRect(pts_xy)  # ((cx, cy), (w, h), angle)
    w, l = rect[1]
    z_min = np.min(points_world[2,:])
    z_max = np.max(points_world[2,:])
    height = z_max - z_min

    # 아래면 4점
    box_xy = cv2.boxPoints(rect)  # (4,2)
    bottom_3d = []
    top_3d = []
    for corner in box_xy:
        bottom_3d.append([corner[0], corner[1], z_min])
        top_3d.append([corner[0], corner[1], z_max])
    corners_3d = np.array(bottom_3d + top_3d)  # (8,3)
    return corners_3d, (w, l, height)
# 이 때 corners_3d는 월드 좌표계에 있는 3D 점들

def project_3d_to_color(corners_3d, transform_world_to_cam, color_intrinsics, extrinsics):
    """
    corners_3d: (8,3) 월드 좌표
    transform_world_to_cam: world->camera
    color_intrinsics: rs.intrinsics
    extrinsics: rs.extrinsics (depth->color)
    Returns:
      corners_2d: 길이 8의 list of (u,v) or None
    """
    if corners_3d.shape[0] != 8:
        return [None]*8

    # (8,3) -> (3,8)
    corners_3d_T = corners_3d.T
    # world->camera
    corners_cam_3d = transform_world_to_cam.apply_transformation(corners_3d_T)  # (3,8)
    corners_cam_3d = corners_cam_3d.T  # (8,3)

    # depth->color extrinsics
    # 두 센서(depth, color) 간의 위치와 방향 차이를 보정하기 위한 회전 및 이동 정보를 나타냅니다.
    corners_color_3d = []
    for i in range(8):
        Xc, Yc, Zc = corners_cam_3d[i]
        p_color = rs.rs2_transform_point_to_point(extrinsics, [Xc, Yc, Zc])
        corners_color_3d.append(p_color)

    # color_intrinsics로 2D 투영
    corners_2d = []
    for p3d in corners_color_3d:
        Xc, Yc, Zc = p3d
        if Zc <= 0: # 카메라 뒤쪽은 z값이 0이하
            corners_2d.append(None)
            continue
        uv = rs.rs2_project_point_to_pixel(color_intrinsics, [Xc, Yc, Zc])
        corners_2d.append(tuple(uv))
    return corners_2d

def draw_3d_box(image, corners_2d, color=(0,255,0), thickness=2):
    """
    corners_2d: 길이 8, 아래면(0~3), 윗면(4~7)
    """
    if any(pt is None for pt in corners_2d):
        return image

    out_img = image.copy()
    c2d = [(int(u), int(v)) for (u,v) in corners_2d]
    # 아래면(0->1->2->3->0)
    for i in range(4):
        cv2.line(out_img, c2d[i], c2d[(i+1)%4], color, thickness)
    # 윗면(4->5->6->7->4)
    for i in range(4):
        cv2.line(out_img, c2d[4+i], c2d[4+((i+1)%4)], color, thickness)
    # 수직선 4개
    for i in range(4):
        cv2.line(out_img, c2d[i], c2d[4+i], color, thickness)

    return out_img

# ----------------------------
def main():
    # 1) RealSense 초기화
    rs_config = rs.config()
    rs_config.enable_stream(rs.stream.depth, RES_WIDTH, RES_HEIGHT, rs.format.z16, FPS)
    rs_config.enable_stream(rs.stream.infrared, 1, RES_WIDTH, RES_HEIGHT, rs.format.y8, FPS)
    rs_config.enable_stream(rs.stream.color, RES_WIDTH, RES_HEIGHT, rs.format.bgr8, FPS)

    device_manager = DeviceManager(rs.context(), rs_config)
    device_manager.enable_all_devices()

    for _ in range(DISPOSE_FRAMES): # 안정될때까지 프레임 버림
        _ = device_manager.poll_frames()

    # --------------------------
    # 2) 체스보드 캘리브레이션
    # --------------------------
    intrinsics_devices = None
    pose_estimator = None
    transformation_result_kabsch = None
    calibrated = False

    print("[Calibration] 체스보드를 보이게 한 뒤, 'c' 키를 눌러주세요.")
    while not calibrated:
        frames = device_manager.poll_frames()
        for dev_info, frame_dict in frames.items():
            if (rs.stream.infrared, 1) in frame_dict:
                ir_frame = frame_dict[(rs.stream.infrared, 1)]
                ir_image = np.asanyarray(ir_frame.get_data())
                cv2.imshow("Calibration IR", ir_image)
                break
        key = cv2.waitKey(1)
        if key == ord('c'):
            cv2.destroyWindow("Calibration IR")
            intrinsics_devices = device_manager.get_device_intrinsics(frames)
            pose_estimator = PoseEstimation(frames, intrinsics_devices,
                                            [CHESSBOARD_HEIGHT, CHESSBOARD_WIDTH, SQUARE_SIZE])
            transformation_result_kabsch = pose_estimator.perform_pose_estimation() # Kabsch 알고리즘 실행 부분, 카메라 -> 체스보드(월드) 변환을 계산
            success_count = 0
            for device_info in device_manager._available_devices:
                device = device_info[0]
                if transformation_result_kabsch[device][0]:
                    success_count += 1
            if success_count == len(device_manager._available_devices):
                calibrated = True
                print("모든 디바이스 캘리브레이션 성공!")
            else:
                print("일부 디바이스가 체스보드를 인식 못했습니다. 다시 'c' 키를 누르세요.")
        elif key == 27:
            device_manager.disable_streams()
            cv2.destroyAllWindows()
            return

    # ROI(체스보드 바닥 범위) 계산
    from helper_functions import get_chessboard_points_3D
    chessboard_points_cumulative_3d = np.array([-1,-1,-1]).reshape(3,1)
    for device_info in device_manager._available_devices:
        device = device_info[0]
        success, transform_obj, _, rmsd_val = transformation_result_kabsch[device]
        if success:
            corners3D_dict = pose_estimator.get_chessboard_corners_in3d()
            found_corners, _, points3D, validPoints = corners3D_dict[device]
            if found_corners:
                points3D_valid = points3D[:, validPoints]
                points_world = transform_obj.inverse().apply_transformation(points3D_valid)
                chessboard_points_cumulative_3d = np.column_stack((chessboard_points_cumulative_3d, points_world))
    chessboard_points_cumulative_3d = np.delete(chessboard_points_cumulative_3d, 0, 1)
    roi_2D = get_boundary_corners_2D(chessboard_points_cumulative_3d)
    print("Calibration 완료. ROI:", roi_2D)

    device_manager.enable_emitter(True)
    device_manager.load_settings_json("./HighResHighAccuracyPreset.json")

    # 3) LangSAM 초기화
    langsam_model = LangSAM()
    print("[Measurement] ESC 누르면 종료합니다.")

    # 4) dilate 파라미터
    kernel = np.ones((5,5), np.uint8)
    dilation_iterations = 1

    while True:
        frames_devices = device_manager.poll_frames()
        device_extrinsics = device_manager.get_depth_to_color_extrinsics(frames_devices)

        for dev_info, frame_dict in frames_devices.items():
            device = dev_info[0]
            color_frame = frame_dict[rs.stream.color]
            depth_frame = frame_dict[rs.stream.depth]

            color_image_bgr = np.asanyarray(color_frame.get_data()).copy()
            color_image_rgb = cv2.cvtColor(color_image_bgr, cv2.COLOR_BGR2RGB)

            # 5) LangSAM 세그멘테이션
            seg_texts = ["red box"]  # 예: "red box"
            seg_results = langsam_model.predict(images_pil=[Image.fromarray(color_image_rgb)],
                                                texts_prompt=seg_texts)
            if len(seg_results) == 0:
                continue

            seg = seg_results[0]
            mask = seg['masks'][0]  # (H,W)
            binary_mask = (mask > 0.5).astype(np.uint8)

            # --- (A) dilate 마스크 ---
            binary_mask = cv2.dilate(binary_mask, kernel, iterations=dilation_iterations)

            # 6) 마스크 픽셀 -> 3D 점
            ys, xs = np.nonzero(binary_mask)
            if len(xs) < 10:
                continue

            depth_intrinsics = color_frame.get_profile().as_video_stream_profile().get_intrinsics()
            depth_image = np.asanyarray(depth_frame.get_data())
            points_cam = []
            for (px, py) in zip(xs, ys):
                d = depth_image[py, px] * DEPTH_SCALE
                if d <= 0:
                    continue
                p3d = rs.rs2_deproject_pixel_to_point(depth_intrinsics, [float(px), float(py)], float(d))
                points_cam.append(p3d)

            if len(points_cam) < 10:
                continue
            points_cam = np.array(points_cam).T  # (3,N)

            # 카메라->월드
            success, transform_obj, _, _ = transformation_result_kabsch[device]
            transform_cam_to_world = transform_obj.inverse()
            points_world = transform_cam_to_world.apply_transformation(points_cam)

            # ROI 필터링
            points_world = get_clipped_pointcloud(points_world, roi_2D)
            if points_world.shape[1] < 10:
                continue

            # 7) 3D BBox 계산
            corners_3d, (w, l, h) = measure_box_minAreaRect(points_world)
            if corners_3d is None:
                continue

            # 8) 투영 & 시각화
            transform_world_to_cam = transform_cam_to_world.inverse()
            corners_2d = project_3d_to_color(corners_3d, transform_world_to_cam,
                                             depth_intrinsics, device_extrinsics[device])
            out_image = draw_3d_box(color_image_bgr, corners_2d, (0,255,0), 2)

            text_info = f"3D: {w:.3f} x {l:.3f} x {h:.3f} m"
            cv2.putText(out_image, text_info, (50,50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

            cv2.imshow(f"Measurement - {device}", out_image)

        key = cv2.waitKey(1)
        if key == 27:  # ESC
            break

    device_manager.disable_streams()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
