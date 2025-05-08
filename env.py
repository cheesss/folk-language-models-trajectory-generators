import pybullet as p
import numpy as np
import pybullet_data
import time
import config
from robot import Robot
from config import OK, PROGRESS, FAIL, ENDC
from config import CAPTURE_IMAGES, ADD_BOUNDING_CUBES, ADD_TRAJECTORY_POINTS, EXECUTE_TRAJECTORY, OPEN_GRIPPER, CLOSE_GRIPPER, SUCTION, TASK_COMPLETED, RESET_ENVIRONMENT
import random
import math


origin=[0, 0, 0]
size = 100

class Environment:

    def __init__(self, args):

        self.mode = args.mode
        self.obj_ids = []

    def load(self):

        p.resetDebugVisualizerCamera(config.camera_distance, config.camera_yaw, config.camera_pitch, config.camera_target_position)

        object_start_position = config.object_start_position
        object_start_orientation_q = p.getQuaternionFromEuler(config.object_start_orientation_e)
        obj_id = p.loadURDF("ycb_assets/005_tomato_soup_can.urdf",object_start_position, object_start_orientation_q, useFixedBase=False, globalScaling=config.global_scaling)
        self.obj_ids.append(obj_id)
        
        # object_start_position = [random.uniform(-0.2, 0.2), random.uniform(0.4, 0.8), 0.1]
        # object_start_orientation_e = [0.0, 0.0, random.uniform(-math.pi, math.pi)]
        # object_start_orientation_q = p.getQuaternionFromEuler(object_start_orientation_e)
        # object_model = p.loadURDF("ycb_assets/003_cracker_box.urdf",object_start_position, object_start_orientation_q, useFixedBase=False, globalScaling=config.global_scaling)

        # object_start_position = [random.uniform(-0.2, 0.2), random.uniform(0.4, 0.8), 0.1]
        # object_start_orientation_e = [0.0, 0.0, random.uniform(-math.pi, math.pi)]
        # object_start_orientation_q = p.getQuaternionFromEuler(object_start_orientation_e)
        # object_model = p.loadURDF("ycb_assets/003_cracker_box.urdf",object_start_position, object_start_orientation_q, useFixedBase=False, globalScaling=config.global_scaling)

        # object_start_position = [random.uniform(-0.2, 0.2), random.uniform(0.4, 0.8), 0.1]
        # object_start_orientation_e = [0.0, 0.0, random.uniform(-math.pi, math.pi)]
        # object_start_orientation_q = p.getQuaternionFromEuler(object_start_orientation_e)
        # object_model = p.loadURDF("ycb_assets/003_cracker_box.urdf",object_start_position, object_start_orientation_q, useFixedBase=False, globalScaling=config.global_scaling)

        if self.mode == "default":

            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0) 
            p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)



    def update(self):

        p.stepSimulation()
        time.sleep(config.control_dt)



# def run_simulation_environment(args, env_connection, logger):

#     # Environment set-up
#     logger.info(PROGRESS + "Setting up environment..." + ENDC)

#     physics_client = p.connect(p.GUI)
#     p.setAdditionalSearchPath(pybullet_data.getDataPath())
#     p.setGravity(0, 0, -9.81)
#     plane = p.loadURDF("plane.urdf")
#     # X축 (빨강)
#     p.addUserDebugLine(
#         origin, [origin[0] + size, origin[1], origin[2]], [1, 0, 0], 3
#     )
#     # Y축 (초록)
#     p.addUserDebugLine(
#         origin, [origin[0], origin[1] + size, origin[2]], [0, 1, 0], 3
#     )
#     # Z축 (파랑)
#     p.addUserDebugLine(
#         origin, [origin[0], origin[1], origin[2] + size], [0, 0, 1], 3
#     )
#     env = Environment(args)
#     env.load()
#     # 물체 소환

#     camera_distance = 0.8
#     camera_yaw = 225.0
#     camera_pitch = -30.0
#     camera_target = [0.0, 0.6, 0.3]

#     view_matrix = p.computeViewMatrixFromYawPitchRoll(
#         cameraTargetPosition=camera_target,
#         distance=camera_distance,
#         yaw=camera_yaw,
#         pitch=camera_pitch,
#         roll=0,
#         upAxisIndex=2
#     )


#     projection_matrix = p.computeProjectionMatrixFOV(
#         fov=60, aspect=1.0, nearVal=0.01, farVal=100
#     )


#     width, height, rgbImg, depthImg, segImg = p.getCameraImage(
#         width=640, height=480,
#         viewMatrix=view_matrix,
#         projectionMatrix=projection_matrix
#     )




#     robot = Robot(args)
#     # args 전달을 통해 무슨 로봇인지 정할수 있다.
#     robot.move(env, robot.ee_start_position, robot.ee_start_orientation_e, gripper_open=True, is_trajectory=False)
#     # pybullet을 이용한 시뮬레이션 구동 is_trajectory = False인 이유는, 목표점을 향하는 단계가 아니라 처음 실행하는 단계이므로 trajectory_step를 갱신하지 않기 위해서이다.
#     env_connection_message = OK + "Finished setting up environment!" + ENDC
#     env_connection.send([env_connection_message])

#     while True:

#         if env_connection.poll():

#             env_connection_received = env_connection.recv()
#             # main_connection에서 보낸 요구사항 저장

#             if env_connection_received[0] == CAPTURE_IMAGES:
#                 #CAPTURE_IMAGES받았을때 pybullet 시뮬레이션 상에서 두 카메라로 찍은 이미지를 전송해준다.
                
#                 # _0 img 저장
#                 # _, _ = robot.get_camera_image("head", env, save_camera_image=True, rgb_image_path=config.rgb_image_trajectory_path.format(step=0), depth_image_path=config.depth_image_trajectory_path.format(step=0))
                
#                 head_camera_position, head_camera_orientation_q = robot.get_camera_image("head", env, save_camera_image=True, rgb_image_path=config.rgb_image_head_path, depth_image_path=config.depth_image_head_path)
#                 wrist_camera_position, wrist_camera_orientation_q = robot.get_camera_image("wrist", env, save_camera_image=True, rgb_image_path=config.rgb_image_wrist_path, depth_image_path=config.depth_image_wrist_path)
#                 # 아무래도 초기 카메라 위치 정보를 실제로 알아야 수정 가능할듯

#                 env_connection_message = OK + "Finished capturing head camera image!" + ENDC
#                 env_connection.send([head_camera_position, head_camera_orientation_q, wrist_camera_position, wrist_camera_orientation_q, env_connection_message])
#                 # main_connection.recv()로 해당 값을 반환 받을 수 있다.

#             elif env_connection_received[0] == ADD_BOUNDING_CUBES:

#                 bounding_cubes_world_coordinates = env_connection_received[1]
#                 # env_connection_received[0]는 ADD_BOUNDING_CUBES이고, env_connection_received[1]은 box cordinates이다.ounding_cube_world_coordinates[1], [0, 1, 0], lif
                
#                 for bounding_cube_world_coordinates in bounding_cubes_world_coordinates:
#                     p.addUserDebugLine(bounding_cube_world_coordinates[0], bounding_cube_world_coordinates[1], [0, 1, 0], lifeTime=0)
#                     p.addUserDebugLine(bounding_cube_world_coordinates[1], bounding_cube_world_coordinates[2], [0, 1, 0], lifeTime=0)
#                     p.addUserDebugLine(bounding_cube_world_coordinates[2], bounding_cube_world_coordinates[3], [0, 1, 0], lifeTime=0)
#                     p.addUserDebugLine(bounding_cube_world_coordinates[3], bounding_cube_world_coordinates[0], [0, 1, 0], lifeTime=0)
#                     p.addUserDebugLine(bounding_cube_world_coordinates[5], bounding_cube_world_coordinates[6], [0, 1, 0], lifeTime=0)
#                     p.addUserDebugLine(bounding_cube_world_coordinates[6], bounding_cube_world_coordinates[7], [0, 1, 0], lifeTime=0)
#                     p.addUserDebugLine(bounding_cube_world_coordinates[7], bounding_cube_world_coordinates[8], [0, 1, 0], lifeTime=0)
#                     p.addUserDebugLine(bounding_cube_world_coordinates[8], bounding_cube_world_coordinates[5], [0, 1, 0], lifeTime=0)
#                     p.addUserDebugLine(bounding_cube_world_coordinates[0], bounding_cube_world_coordinates[5], [0, 1, 0], lifeTime=0)
#                     p.addUserDebugLine(bounding_cube_world_coordinates[1], bounding_cube_world_coordinates[6], [0, 1, 0], lifeTime=0)
#                     p.addUserDebugLine(bounding_cube_world_coordinates[2], bounding_cube_world_coordinates[7], [0, 1, 0], lifeTime=0)
#                     p.addUserDebugLine(bounding_cube_world_coordinates[3], bounding_cube_world_coordinates[8], [0, 1, 0], lifeTime=0)
#                     p.addUserDebugPoints(bounding_cube_world_coordinates, [[0, 1, 0]] * len(bounding_cube_world_coordinates), pointSize=5, lifeTime=0)

#                 env_connection_message = OK + "Finished adding bounding cubes to the environment!" + ENDC
#                 env_connection.send([env_connection_message])

#             elif env_connection_received[0] == ADD_TRAJECTORY_POINTS:

#                 trajectory = env_connection_received[1]

#                 trajectory_points = [point[:3] for point in trajectory]
#                 p.addUserDebugPoints(trajectory_points, [[0, 1, 1]] * len(trajectory_points), pointSize=5, lifeTime=0)

#                 logger.info(OK + "Finished adding trajectory points to the environment!" + ENDC)

#             elif env_connection_received[0] == EXECUTE_TRAJECTORY:

#                 trajectory = env_connection_received[1]

#                 for point in trajectory:
#                     robot.move(env, point[:3], np.array(robot.ee_start_orientation_e) + np.array([0, 0, point[3]]), gripper_open=robot.gripper_open, is_trajectory=True)

#                 for _ in range(100):
#                     env.update()

#                 logger.info(OK + "Finished executing generated trajectory!" + ENDC)

#             elif env_connection_received[0] == OPEN_GRIPPER:

#                 ee_current_position = p.getLinkState(robot.id, robot.ee_index, computeForwardKinematics=True)[0]
#                 ee_current_orientation_q = p.getLinkState(robot.id, robot.ee_index, computeForwardKinematics=True)[1]
#                 ee_current_orientation_e = p.getEulerFromQuaternion(ee_current_orientation_q)

#                 robot.move(env, ee_current_position, ee_current_orientation_e, gripper_open=True, is_trajectory=False)

#                 robot.gripper_open = True

#                 logger.info(OK + "Finished opening gripper!" + ENDC)

#             elif env_connection_received[0] == CLOSE_GRIPPER:

#                 ee_current_position = p.getLinkState(robot.id, robot.ee_index, computeForwardKinematics=True)[0]
#                 ee_current_orientation_q = p.getLinkState(robot.id, robot.ee_index, computeForwardKinematics=True)[1]
#                 ee_current_orientation_e = p.getEulerFromQuaternion(ee_current_orientation_q)

#                 robot.move(env, ee_current_position, ee_current_orientation_e, gripper_open=False, is_trajectory=False)

#                 robot.gripper_open = False
                
                
#             elif env_connection_received[0] == SUCTION:
#                 ee_current_position = p.getLinkState(robot.id, robot.ee_index, computeForwardKinematics=True)[0]
#                 ee_current_orientation_q = p.getLinkState(robot.id, robot.ee_index, computeForwardKinematics=True)[1]
#                 ee_current_orientation_e = p.getEulerFromQuaternion(ee_current_orientation_q)

#                 robot.move(env, ee_current_position, ee_current_orientation_e, gripper_open=False, is_trajectory=False)

#                 robot.gripper_open = False
#                 print("[INFO] SUCTION command received and processed.")

                

#                 logger.info(OK + "Finished closing gripper!" + ENDC)

#             elif env_connection_received[0] == TASK_COMPLETED:

#                 env_connection_message = OK + "Finished executing all generated trajectories!" + ENDC
#                 env_connection.send([env_connection_message])

#             elif env_connection_received[0] == RESET_ENVIRONMENT:

#                 robot.move(env, robot.ee_start_position, robot.ee_start_orientation_e, gripper_open=True, is_trajectory=False)
#                 robot.gripper_open = True
#                 robot.trajectory_step = 1

#                 for _ in range(100):
#                     env.update()

#                 env_connection_message = OK + "Finished resetting environment!" + ENDC
#                 env_connection.send([env_connection_message])

#         env.update()


def run_simulation_environment(args, env_connection, logger):
    logger.info("[MockEnv] Starting mock simulation environment...")

    # 환경 세팅 완료 메시지
    env_connection_message = "[MockEnv] Environment setup complete (no simulation)"
    env_connection.send([env_connection_message])

    # 더미 로봇 상태
    dummy_position = [0.0, 0.0, 0.0]
    dummy_orientation = [0.0, 0.0, 0.0]

    while True:
        if env_connection.poll():
            env_connection_received = env_connection.recv()
            command = env_connection_received[0]

            logger.info(f"[MockEnv] Received command: {command}")

            if command == CAPTURE_IMAGES:
                env_connection.send([
                    dummy_position,
                    dummy_orientation,
                    dummy_position,
                    dummy_orientation,
                    "[MockEnv] Dummy camera images captured"
                ])

            elif command == ADD_BOUNDING_CUBES:
                logger.info("[MockEnv] Pretending to add bounding cubes")
                env_connection.send(["[MockEnv] Bounding cubes added (no-op)"])

            elif command == ADD_TRAJECTORY_POINTS:
                logger.info("[MockEnv] Pretending to add trajectory points")

            elif command == EXECUTE_TRAJECTORY:
                logger.info("[MockEnv] Pretending to execute trajectory")

            elif command == OPEN_GRIPPER:
                logger.info("[MockEnv] Pretending to open gripper")

            elif command == CLOSE_GRIPPER:
                logger.info("[MockEnv] Pretending to close gripper")

            elif command == SUCTION:
                logger.info("[MockEnv] Pretending to perform suction")

            elif command == TASK_COMPLETED:
                env_connection.send(["[MockEnv] Task marked as complete"])

            elif command == RESET_ENVIRONMENT:
                logger.info("[MockEnv] Resetting dummy environment")
                env_connection.send(["[MockEnv] Environment reset"])

        time.sleep(0.01)
