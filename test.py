from realsenseCapture import IntelCamera
import utils
from utils import get_bounding_cube_from_point_cloud
from utils import get_segmentation_mask
import numpy as np
from PIL import Image
import config
import math
import models
from lang_sam import LangSAM 
import pybullet as p
import pybullet_data
import time
import pyrealsense2 as rs
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



rgba_image, depth_image, depth_intrinsics = IntelCamera.capture_save_image()
langsam_model = LangSAM()

rgb_image_head = Image.open(config.rgb_image_head_path).convert("RGB")

model_predictions, boxes, segmentation_texts = models.get_langsam_output(rgb_image_head,langsam_model, segmentation_texts=["box"], segmentation_count=0)
print(f"model_predictions: {model_predictions}")


masks = get_segmentation_mask(model_predictions, config.segmentation_threshold)
print(f"masks: {masks}")
# print(1)
depth_image_head = Image.open(config.depth_image_head_path).convert("L")
# print(2)
depth_scale = 0.0010000000474974513
depth_array = np.array(depth_image_head) * depth_scale



# print("depth_array: ",depth_array)
# print(3)
head_camera_position = [0.0, 1.2, 0.6]
# print(4)
# head_camera_orientation_e = [0.0, 5 / 6 * math.pi, -math.pi / 2]
head_camera_orientation_e = [0.0, 5 / 6 * math.pi, -math.pi / 2]

# print(5)
camera_orientation_q = p.getQuaternionFromEuler(config.head_camera_orientation_e)
# print(6)
# bounding_cubes_world_coordinates, bounding_cubes_orientations = utils.get_bounding_cube_from_point_cloud(rgb_image_head, masks, depth_array, head_camera_position, camera_orientation_q, segmentation_count=0)
# print(7)

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

try:
    # 필요한 작업 수행
    bounding_cubes_world_coordinates, bounding_cubes_orientations, contour_pixel_points = get_bounding_cube_from_point_cloud(
        rgb_image_head, masks, depth_array, head_camera_position, camera_orientation_q, depth_image=depth_image ,depth_intrinsics = depth_intrinsics , segmentation_count=0
    )
    # 이게 실행되면서 o3d 시각화도 진행된다


finally:
    # 파이프라인 정리
    pipeline.stop()

# a, b =bounding_cubes_world_coordinates, bounding_cubes_orientations 
# print("bounding_cubes_world_coordinates, bounding_cubes_orientations: ",a, b)





def update():
    p.stepSimulation()
    time.sleep(config.control_dt)




import config
def check_real_size():
    for i, bounding_cube_world_coordinates in enumerate(bounding_cubes_world_coordinates):

        bounding_cube_world_coordinates[4][2] -= config.depth_offset

        object_width = np.around(np.linalg.norm(bounding_cube_world_coordinates[1] - bounding_cube_world_coordinates[0]), 3)
        object_length = np.around(np.linalg.norm(bounding_cube_world_coordinates[2] - bounding_cube_world_coordinates[1]), 3)
        object_height = np.around(np.linalg.norm(bounding_cube_world_coordinates[5] - bounding_cube_world_coordinates[0]), 3)

        print("Position of", segmentation_texts[i], ":", list(np.around(bounding_cube_world_coordinates[4], 3)))


        print("Dimensions:")
        print("Width:", object_width)
        print("Length:", object_length)
        print("Height:", object_height)

        if object_width < object_length:
            print("Orientation along shorter side (width):", np.around(bounding_cubes_orientations[i][0], 3))
            print("Orientation along longer side (length):", np.around(bounding_cubes_orientations[i][1], 3), "\n")
        else:
            print("Orientation along shorter side (length):", np.around(bounding_cubes_orientations[i][1], 3))
            print("Orientation along longer side (width):", np.around(bounding_cubes_orientations[i][0], 3), "\n")



check_real_size()

import config


# visualize_depth_array(depth_array)


# print("contour_pixel_points: ",contour_pixel_points)
# draw_box_pybullet()

