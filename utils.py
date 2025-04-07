import pybullet as p
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import torch
import math
import config
from PIL import Image
from torchvision.utils import save_image
from shapely.geometry import MultiPoint, Polygon, polygon
import multiprocessing
import logging
import pyrealsense2 as rs
import open3d as o3d

depth_scale = 0.0010000000474974513
base2cam = np.array([[ 9.97956043e-01,  1.24564979e-03,  6.38919201e-02,  5.63566259e-02],
            [-3.78161693e-02, -7.94446176e-01,  6.06156097e-01, -1.47778554e+00],
            [ 5.15137493e-02, -6.07333292e-01, -7.92775253e-01,  4.75763504e-01],
            [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00],])



logger = multiprocessing.log_to_stderr()
logger.setLevel(logging.INFO)

def get_segmentation_mask(model_predictions, segmentation_threshold):

    masks = []

    for model_prediction in model_predictions:
        model_prediction_np = model_prediction.detach().cpu().numpy()
        segmentation_threshold = np.max(model_prediction_np) - segmentation_threshold * (np.max(model_prediction_np) - np.min(model_prediction_np))
        model_prediction[model_prediction < segmentation_threshold] = False
        model_prediction[model_prediction >= segmentation_threshold] = True
        masks.append(model_prediction)

    return masks



def get_max_contour(image, image_width, image_height):

    ret, thresh = cv.threshold(image, 127, 255, 0)
    contours, hierarchy = cv.findContours(thresh, 1, 2)

    contour_index = None
    max_length = 0
    for c, contour in enumerate(contours):
        contour_points = [(c, r) for r in range(image_height) for c in range(image_width) if cv.pointPolygonTest(contour, (c, r), measureDist=False) == 1]
        if len(contour_points) > max_length:
            contour_index = c
            max_length = len(contour_points)

    if contour_index is None:
        return None

    return contours[contour_index]



# def get_intrinsics_extrinsics(image_height, camera_position, camera_orientation_q):

#     fov = (config.fov / 360) * 2 * math.pi
#     f_x = f_y = image_height / (2 * math.tan(fov / 2))
#     K = np.array([[f_x, 0, 0], [0, f_y, 0], [0, 0, 1]])

#     R = np.array(p.getMatrixFromQuaternion(camera_orientation_q)).reshape(3, 3)
#     Rt = np.hstack((R, np.array(camera_position).reshape(3, 1)))
#     Rt = np.vstack((Rt, np.array([0, 0, 0, 1])))

#     return K, Rt

def get_intrinsics_extrinsics(pipeline, image_height, camera_position, camera_orientation_q):
    profile = pipeline.get_active_profile()
    stream = profile.get_stream(rs.stream.color)
    intrinsics = stream.as_video_stream_profile().get_intrinsics()

    K = np.array([
        [intrinsics.fx, 0, intrinsics.ppx],
        [0, intrinsics.fy, intrinsics.ppy],
        [0, 0, 1]
    ])

    R = np.array(p.getMatrixFromQuaternion(camera_orientation_q)).reshape(3, 3)
    Rt = np.hstack((R, np.array(camera_position).reshape(3, 1)))
    Rt = np.vstack((Rt, np.array([0, 0, 0, 1])))

    return K, Rt

def save_xmem_image(masks):

    xmem_array = np.array(Image.open(config.xmem_input_path).convert("L"))
    xmem_array = np.unique(xmem_array, return_inverse=True)[1].reshape(xmem_array.shape)

    # for mask in masks:
    #     # mask 차원 확인 및 변환
    #     mask_np = mask.detach().cpu().numpy()
    #     if mask_np.ndim == 3:
    #         mask_np = mask_np.squeeze(0)  # 3D -> 2D (H, W)

    #     # mask 값 적용
    #     mask_index = np.max(xmem_array) + 1
    #     xmem_array[mask_np.astype(bool)] = mask_index

    xmem_array = xmem_array / np.max(xmem_array)

    save_image(torch.Tensor(xmem_array), config.xmem_input_path)



# def get_bounding_cube_from_point_cloud(pipeline, image, masks, depth_array, camera_position, camera_orientation_q, segmentation_count):
def get_bounding_cube_from_point_cloud( image, masks, depth_array, camera_position, camera_orientation_q, depth_image, depth_intrinsics, segmentation_count):
    # depth 이미지 사용!
    image_width, image_height = image.size
    # plt.imshow(image)
    # plt.show(image)

    bounding_cubes = []
    bounding_cubes_orientations = []
    depth_in_meters = depth_image.astype(np.float32)
    for i, mask in enumerate(masks):

        save_image(mask, config.bounding_cube_mask_image_path.format(object=segmentation_count, mask=i))
        mask_np = cv.imread(config.bounding_cube_mask_image_path.format(object=segmentation_count, mask=i), cv.IMREAD_GRAYSCALE)
        plt.imshow(mask_np, cmap='gray')
        plt.title("Mask Image")
        plt.show()
        contour = get_max_contour(mask_np, image_width, image_height)
        # 컨투어 값은 잘 가져오고있다.
        if contour is not None:

            contour_pixel_points = [(c, r, depth_array[c][r]) for r in range(image_height) for c in range(image_width) if cv.pointPolygonTest(contour, (r, c), measureDist=False) == 1]

            # print('Contour_pixel_points: ', contour_pixel_points)
            # contour_pixel_points 또한 정상 출력되고있다.

            # 원본코드
            # contour_world_points = [get_world_point_world_frame(pipeline, camera_position, camera_orientation_q, "head", image, pixel_point) for pixel_point in contour_pixel_points]
            

            # -------------------------------- 수정코드
            # print('Contour_world_points shape: ', contour_pixel_points.shape)
            contour_world_points = []
            for pixel_point in contour_pixel_points:
                c, r, depth = pixel_point
                if depth > 0:
                    # print("c, r: ", c, r)
                    world_point = rs.rs2_deproject_pixel_to_point(depth_intrinsics, [c, r], (depth_image[c][r])*depth_scale)
                    
                    contour_world_points.append(world_point)

            if len(contour_world_points) == 0:
                continue
            
            contour_world_points = np.array(contour_world_points)
            # print("Contour world points: ", contour_world_points)
            # --------------------------------
            
            contour_world_points = outier_removed_point_cloud(contour_world_points)
            contour_world_points = np.asarray(contour_world_points.points)



            # print("Contour world points", contour_world_points)
            # 얘도 아마 정상 출력인듯
            max_z_coordinate = np.max((contour_world_points)[:, 2])
            # print("Max_z_coordinate", max_z_coordinate)
            min_z_coordinate = np.min((contour_world_points)[:, 2])
            # print("min_z_cord: ",min_z_coordinate)
            depth_offset = 0.03
            # depth_offset = 0.08
            top_surface_world_points = [world_point for world_point in contour_world_points if world_point[2] > max_z_coordinate - depth_offset]
            # logging.info("Top surface world points"+str(top_surface_world_points))
            rect = MultiPoint([world_point[:2] for world_point in top_surface_world_points]).minimum_rotated_rectangle
            # logging.info("Rectangle"+str(rect))
            print(isinstance(rect, Polygon))
            if isinstance(rect, Polygon):
                rect = polygon.orient(rect, sign=-1)
                box = rect.exterior.coords
                box = np.array(box[:-1])
                box_min_x = np.argmin(box[:, 0])
                box = np.roll(box, -box_min_x, axis=0)
                box_top = [list(point) + [max_z_coordinate] for point in box]
                box_btm = [list(point) + [min_z_coordinate] for point in box]
                box_top.append(list(np.mean(box_top, axis=0)))
                box_btm.append(list(np.mean(box_btm, axis=0)))
                bounding_cubes.append(box_top + box_btm)

                # Calculating rotation in world frame
                bounding_cubes_orientation_width = np.arctan2(box[1][1] - box[0][1], box[1][0] - box[0][0])
                bounding_cubes_orientation_length = np.arctan2(box[2][1] - box[1][1], box[2][0] - box[1][0])
                bounding_cubes_orientations.append([bounding_cubes_orientation_width, bounding_cubes_orientation_length])

    bounding_cubes = np.array(bounding_cubes)

    return bounding_cubes, bounding_cubes_orientations, contour_pixel_points




def get_world_point_world_frame(pipeline, camera_position, camera_orientation_q, camera, image, point):

    image_width, image_height = image.size

    K, Rt = get_intrinsics_extrinsics(pipeline, image_height=image_height, camera_position=camera_position, camera_orientation_q=camera_orientation_q)
    # logging.info("Rt:" +str(Rt))
    pixel_point = np.array([[point[0] - (image_width / 2)], [(image_height / 2) - point[1]], [1.0]])

    if camera == "wrist":
        pixel_point = [pixel_point[1], pixel_point[0], pixel_point[2]]
    elif camera == "head":
        pixel_point = [-pixel_point[1], -pixel_point[0], pixel_point[2]]

    world_point_camera_frame = (np.linalg.inv(K) @ pixel_point) * point[2]
    world_point_world_frame = Rt @ np.vstack((world_point_camera_frame, np.array([1.0])))
    world_point_world_frame = world_point_world_frame.squeeze()[:-1]

    return world_point_world_frame


def outier_removed_point_cloud(points):
    point_cloud = o3d.geometry.PointCloud()
    point_cloud_vis = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    voxel_down_pcd = point_cloud.voxel_down_sample(voxel_size=0.005)
    cl, ind = voxel_down_pcd.remove_statistical_outlier(nb_neighbors=10, std_ratio=2.0)
    inlier_cloud = voxel_down_pcd.select_by_index(ind)
    # point_np = np.asarray(inlier_cloud.points)
    # ones = np.ones(len(point_np))
    # base2cam_p1 = np.c_[point_np, ones]
    # point_T = np.dot(base2cam,base2cam_p1.T)

    inlier_cloud = inlier_cloud.transform(base2cam)

    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
    c_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
    c_frame.transform(base2cam)
    # 포인트 클라우드 시각화
    # point_cloud.points = o3d.utility.Vector3dVector(point_T.T[:, :-1])

    o3d.visualization.draw_geometries([inlier_cloud,frame, c_frame])
    # o3d.visualization.draw_geometries([voxel_down_pcd, frame])
    
    return inlier_cloud