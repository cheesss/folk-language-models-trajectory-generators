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
from config import depth_offset

depth_scale = 0.0010000000474974513
base2cam = np.array([[ 9.97956043e-01,  1.24564979e-03,  6.38919201e-02,  5.63566259e-02],
            [-3.78161693e-02, -7.94446176e-01,  6.06156097e-01, -1.47778554e+00],
            [ 5.15137493e-02, -6.07333292e-01, -7.92775253e-01,  4.75763504e-01],
            [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00],])



logger = multiprocessing.log_to_stderr()
logger.setLevel(logging.INFO)


def align_pointcloud_to_plane(pcd):
    plane_model, inliers = pcd.segment_plane(distance_threshold=0.01,
                                             ransac_n=3,
                                             num_iterations=1000)
    [a, b, c, d] = plane_model
    print(f"Plane equation: {a:.3f}x + {b:.3f}y + {c:.3f}z + {d:.3f} = 0")

    normal = np.array([a, b, c])
    z_axis = np.array([0.0, 0.0, 1.0])
    v = np.cross(normal, z_axis)
    s = np.linalg.norm(v)
    if s < 1e-6:
        return pcd
    c = np.dot(normal, z_axis)
    vx = np.array([[0, -v[2], v[1]],
                   [v[2], 0, -v[0]],
                   [-v[1], v[0], 0]])
    R = np.eye(3) + vx + vx @ vx * ((1 - c) / (s**2))

    pcd.rotate(R, center=(0, 0, 0))
    return pcd

def create_bounding_box_from_points(pcd):
    aabb = pcd.get_axis_aligned_bounding_box()
    aabb.color = (1, 0, 0)
    return pcd, aabb

def get_segmentation_mask(model_predictions, segmentation_threshold):
    """
    Converts LangSAM model output (logits) into binary masks using a threshold value.
    Thresholding is done relative to max-min normalization.

    Args:
        model_predictions (List[torch.Tensor]): A list of raw prediction logits from LangSAM.
        segmentation_threshold (float): A value between 0 and 1 to control thresholding.

    Returns:
        List[torch.Tensor]: Binary masks (True/False) for each predicted object.
    """
    masks = []
    for pred in model_predictions:
        pred = pred.detach().cpu()
        max_val = torch.max(pred)
        min_val = torch.min(pred)
        threshold_value = max_val - segmentation_threshold * (max_val - min_val)
        mask = pred >= threshold_value
        masks.append(mask.bool())
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
# def get_bounding_cube_from_point_cloud( image, masks, depth_array, camera_position, camera_orientation_q, depth_image, depth_intrinsics, segmentation_count):
#     # depth 이미지 사용!
#     image_width, image_height = image.size
#     # plt.imshow(image)
#     # plt.show(image)

#     bounding_cubes = []
#     bounding_cubes_orientations = []
#     depth_in_meters = depth_image.astype(np.float32)
#     for i, mask in enumerate(masks):

#         # save_image(mask, config.bounding_cube_mask_image_path.format(object=segmentation_count, mask=i))
#         save_image(mask.float(), config.bounding_cube_mask_image_path.format(object=segmentation_count, mask=i))

#         mask_np = cv.imread(config.bounding_cube_mask_image_path.format(object=segmentation_count, mask=i), cv.IMREAD_GRAYSCALE)
#         plt.imshow(mask_np, cmap='gray')
#         plt.title("Mask Image")
#         plt.show()
#         contour = get_max_contour(mask_np, image_width, image_height)
#         # 컨투어 값은 잘 가져오고있다.
#         if contour is not None:

#             contour_pixel_points = [(c, r, depth_array[c][r]) for r in range(image_height) for c in range(image_width) if cv.pointPolygonTest(contour, (r, c), measureDist=False) >= 1]

#             # print('Contour_pixel_points: ', contour_pixel_points)
#             # contour_pixel_points 또한 정상 출력되고있다.

#             # 원본코드
#             # contour_world_points = [get_world_point_world_frame(pipeline, camera_position, camera_orientation_q, "head", image, pixel_point) for pixel_point in contour_pixel_points]
            

#             # -------------------------------- 수정코드
#             # print('Contour_world_points shape: ', contour_pixel_points.shape)
#             contour_world_points = []
#             for pixel_point in contour_pixel_points:
#                 c, r, depth = pixel_point
#                 if depth > 0:
#                     # print("c, r: ", c, r)
#                     world_point = rs.rs2_deproject_pixel_to_point(depth_intrinsics, [c, r], (depth_image[c][r])*depth_scale)
                    
#                     contour_world_points.append(world_point)

#             if len(contour_world_points) == 0:
#                 continue
            
#             contour_world_points = np.array(contour_world_points)
#             # print("Contour world points: ", contour_world_points)
#             # --------------------------------
            
#             pcd = outier_removed_point_cloud(np.array(contour_world_points))

#             # 바운딩 박스 시각화
#             pcd_vis, aabb = create_bounding_box_from_points(pcd)
#             o3d.visualization.draw_geometries([pcd_vis, aabb], window_name="Segmented PointCloud + 3D Bounding Box")

#             # 정렬된 좌표 배열로 추출
#             points_np = np.asarray(pcd.points)

#             # top/bottom z값 계산
#             max_z_coordinate = np.max(points_np[:, 2])
#             min_z_coordinate = np.min(points_np[:, 2])
            

#             # 상단 평면만 추출 (정렬된 z 기준)
#             top_surface_world_points = [pt for pt in points_np if pt[2] > max_z_coordinate - depth_offset]

#             # 2D XY 평면에 사각형 fitting
#             rect = MultiPoint([pt[:2] for pt in top_surface_world_points]).minimum_rotated_rectangle

#             if isinstance(rect, Polygon):
#                 rect = polygon.orient(rect, sign=-1)
#                 box = np.array(rect.exterior.coords[:-1])  # 마지막 점 중복 제거
#                 box_min_x = np.argmin(box[:, 0])
#                 box = np.roll(box, -box_min_x, axis=0)

#                 box_top = [list(point) + [max_z_coordinate] for point in box]
#                 box_btm = [list(point) + [min_z_coordinate] for point in box]
#                 box_top.append(list(np.mean(box_top, axis=0)))
#                 box_btm.append(list(np.mean(box_btm, axis=0)))
#                 bounding_cubes.append(box_top + box_btm)

#                 # 회전각도 계산
#                 bounding_cubes_orientation_width = np.arctan2(box[1][1] - box[0][1], box[1][0] - box[0][0])
#                 bounding_cubes_orientation_length = np.arctan2(box[2][1] - box[1][1], box[2][0] - box[1][0])
#                 bounding_cubes_orientations.append([bounding_cubes_orientation_width, bounding_cubes_orientation_length])
#                 bounding_cubes = np.array(bounding_cubes)
#     bounding_cubes = np.array(bounding_cubes)

#     return bounding_cubes, bounding_cubes_orientations, contour_pixel_points
def get_bounding_cube_from_point_cloud(image, masks, depth_array, camera_position, camera_orientation_q, depth_image, depth_intrinsics, segmentation_count):
    image_width, image_height = image.size

    bounding_cubes = []
    bounding_cubes_orientations = []
    for i, mask in enumerate(masks):
        save_image(mask.float(), config.bounding_cube_mask_image_path.format(object=segmentation_count, mask=i))

        mask_np = cv.imread(config.bounding_cube_mask_image_path.format(object=segmentation_count, mask=i), cv.IMREAD_GRAYSCALE)
        plt.imshow(mask_np, cmap='gray')
        plt.title("Mask Image")
        plt.show()

        contour = get_max_contour(mask_np, image_width, image_height)
        if contour is not None:
            contour_pixel_points = [(c, r, depth_array[r, c]) for r in range(image_height) for c in range(image_width)
                                        if cv.pointPolygonTest(contour, (c, r), measureDist=False) >= 0]

            contour_world_points = []
            for pixel_point in contour_pixel_points:
                c, r, depth = pixel_point
                if depth > 0:
                    world_point = rs.rs2_deproject_pixel_to_point(depth_intrinsics, [c, r], (depth_image[r, c]) * depth_scale)
                    contour_world_points.append(world_point)

            if len(contour_world_points) == 0:
                continue

            contour_world_points = np.array(contour_world_points)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(contour_world_points)
            pcd = pcd.voxel_down_sample(voxel_size=0.005)
            cl, ind = pcd.remove_statistical_outlier(nb_neighbors=150, std_ratio=2.0)
            pcd = pcd.select_by_index(ind)

            plane_model, _ = pcd.segment_plane(distance_threshold=0.01, ransac_n=3, num_iterations=1000)
            normal = np.array(plane_model[:3])
            z_axis = np.array([0.0, 0.0, 1.0])
            v = np.cross(normal, z_axis)
            s = np.linalg.norm(v)
            if s >= 1e-6:
                c = np.dot(normal, z_axis)
                vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
                R = np.eye(3) + vx + vx @ vx * ((1 - c) / (s**2))
                pcd.rotate(R, center=(0, 0, 0))

            pcd_vis = pcd.paint_uniform_color([0.1, 0.8, 0.1])
            aabb = pcd.get_axis_aligned_bounding_box()
            aabb.color = (1, 0, 0)
            o3d.visualization.draw_geometries([pcd_vis, aabb], window_name="Segmented PointCloud + 3D Bounding Box")

            points_np = np.asarray(pcd.points)
            max_z_coordinate = np.max(points_np[:, 2])
            min_z_coordinate = np.min(points_np[:, 2])
            depth_offset = 0.03

            top_surface_world_points = [pt for pt in points_np if pt[2] > max_z_coordinate - depth_offset]
            rect = MultiPoint([pt[:2] for pt in top_surface_world_points]).minimum_rotated_rectangle

            if isinstance(rect, Polygon):
                rect = polygon.orient(rect, sign=-1)
                box = np.array(rect.exterior.coords[:-1])
                box_min_x = np.argmin(box[:, 0])
                box = np.roll(box, -box_min_x, axis=0)

                box_top = [list(point) + [max_z_coordinate] for point in box]
                box_btm = [list(point) + [min_z_coordinate] for point in box]
                box_top.append(list(np.mean(box_top, axis=0)))
                box_btm.append(list(np.mean(box_btm, axis=0)))
                bounding_cubes.append(box_top + box_btm)

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
    point_cloud.points = o3d.utility.Vector3dVector(points)

    # 다운샘플링 + 노이즈 제거
    voxel_down_pcd = point_cloud.voxel_down_sample(voxel_size=0.005)
    cl, ind = voxel_down_pcd.remove_statistical_outlier(nb_neighbors=100, std_ratio=2.0)
    inlier_cloud = voxel_down_pcd.select_by_index(ind)

    # World 좌표계로 변환
    inlier_cloud = inlier_cloud.transform(base2cam)

    # ✅ 정렬 추가 (z축 정렬)
    inlier_cloud = align_pointcloud_to_plane(inlier_cloud)

    # 시각화
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
    c_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
    c_frame.transform(base2cam)
    o3d.visualization.draw_geometries([inlier_cloud, frame, c_frame])

    return inlier_cloud
