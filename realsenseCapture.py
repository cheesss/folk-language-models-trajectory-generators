import threading
import cv2
import pyrealsense2 as rs
import numpy as np
import open3d as o3d
from PIL import Image
import multiprocessing
import logging

class IntelCamera:
    def __init__(self, image):
        self.image = image
        
    def save_RGB_img(image):
        # RGB 이미지를 JPEG 파일로 저장
        cv2.imwrite("captured_image.png", image)
        # print("RGB image saved as 'captured_image.png'")

    def save_Depth_img(depth_image):
        
        cv2.imwrite("captured_depth_image.png", depth_image)
        # print("Depth image saved as 'captured_depth_image.png'")
    
    def capture_save_image():

        # Logging
        # logger = multiprocessing.log_to_stderr()
        # logger.setLevel(logging.INFO)

        pipeline = rs.pipeline()
        config = rs.config()
        
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

        profile = pipeline.start(config)
        depth_sensor = profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()
        depth_stream = profile.get_stream(rs.stream.depth)
        depth_intrinsics = depth_stream.as_video_stream_profile().get_intrinsics()

        # print(f"Depth Scale: {depth_scale} meters per unit")
        align = rs.align(rs.stream.color)

        try:
            # pipeline.start(config)

            # 프레임 읽기
            for _ in range(37):
                frames = pipeline.wait_for_frames()
                frames = align.process(frames)
                color_frame = frames.get_color_frame()
                depth_frame = frames.get_depth_frame()

            if not color_frame:
                print("No color frame captured!")

            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())
            alpha_channel = np.full((color_image.shape[0], color_image.shape[1], 1), 255, dtype=np.uint8)
            rgba_image = np.concatenate((color_image, alpha_channel), axis=-1)
            # cv2.imshow('Captured Image', color_image)
            threading.Thread(target=IntelCamera.save_RGB_img, args=(color_image,)).start()
            threading.Thread(target=IntelCamera.save_Depth_img, args=(depth_image,)).start()
            return rgba_image, depth_image, depth_intrinsics
    # 리턴값 없이 촬영한 사진을 저장한 후 불러와 사용한다.
        finally:
            pipeline.stop()
            
            
# a = IntelCamera.capture_save_image()