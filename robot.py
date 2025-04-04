import pybullet as p
import numpy as np
import math
import config
from PIL import Image
from config import fov, aspect, near_plane, far_plane

class Robot:

    def __init__(self, args):

        if args.robot == "sawyer":
            self.base_start_position = config.base_start_position_sawyer
            self.base_start_orientation_q = p.getQuaternionFromEuler(config.base_start_orientation_e_sawyer)
            self.joint_start_positions = config.joint_start_positions_sawyer
            self.id = p.loadURDF("sawyer_robot/sawyer_description/urdf/sawyer.urdf", self.base_start_position, self.base_start_orientation_q, useFixedBase=True)
            self.robot = "sawyer"
            self.ee_index = config.ee_index_sawyer
        elif args.robot == "franka":
            self.base_start_position = config.base_start_position_franka
            self.base_start_orientation_q = p.getQuaternionFromEuler(config.base_start_orientation_e_franka)
            self.joint_start_positions = config.joint_start_positions_franka
            self.id = p.loadURDF("franka_robot/panda.urdf", self.base_start_position, self.base_start_orientation_q, useFixedBase=True)
            self.robot = "franka"
            self.ee_index = config.ee_index_franka
        elif args.robot == "franka_suction":
            self.base_start_position = config.base_start_position_franka
            self.base_start_orientation_q = p.getQuaternionFromEuler(config.base_start_orientation_e_franka)
            self.joint_start_positions = config.joint_start_positions_franka
            self.id = p.loadURDF("franka_robot/panda.urdf", self.base_start_position, self.base_start_orientation_q, useFixedBase=True)
            self.robot = "franka_suction"
            self.ee_index = config.ee_index_franka
        self.ee_start_position = config.ee_start_position
        self.ee_start_orientation_e = config.ee_start_orientation_e
        self.ee_current_position = config.ee_start_position
        self.ee_current_orientation_e = config.ee_start_orientation_e

        self.gripper_open = True
        self.trajectory_step = 1

        i = 0
        for j in range(p.getNumJoints(self.id)):
            joint_type = p.getJointInfo(self.id, j)[2]
            if joint_type == p.JOINT_PRISMATIC or joint_type == p.JOINT_REVOLUTE:
                p.resetJointState(self.id, j, self.joint_start_positions[i])
                i += 1



    def move(self, env, ee_target_position, ee_target_orientation_e, gripper_open, is_trajectory):

        if self.robot == "sawyer":
            gripper1_index = None
            gripper2_index = None
            gripper_target_position = config.gripper_goal_position_open_sawyer if gripper_open else config.gripper_goal_position_closed_sawyer
        elif self.robot == "franka":
            print("franka is selected.")
            gripper1_index = 9
            gripper2_index = 10
            # gripper index 1,2는 그리퍼 양끝 엔드포인트를 나타낸다.
            gripper_target_position = config.gripper_goal_position_open_franka if gripper_open else config.gripper_goal_position_closed_franka
            # 여기서 gripper_open이 False로 오면 gripper_goal_position_closed_franka = 0.0005로 설정된다
        elif self.robot == "franka_suction":
            print("franka_suction is selected.")
            gripper1_index = 9
            gripper2_index = 10
            gripper_target_position = config.gripper_goal_position_open_franka if gripper_open else config.gripper_goal_position_closed_franka

            ee_pos = p.getLinkState(self.id, self.ee_index, computeForwardKinematics=True)[0]
            ee_orn = p.getLinkState(self.id, self.ee_index, computeForwardKinematics=True)[1]

            if not hasattr(self, "suction_constraint_id"):
                self.suction_constraint_id = None

            if not gripper_open and self.suction_constraint_id is None:
                target_object_id = env.obj_ids[0]  # 리스트의 첫 번째 객체를 사용
                print(f"gripper_open: {gripper_open}, target_object_id: {target_object_id}")
                self.suction_constraint_id = p.createConstraint(
                    parentBodyUniqueId=self.id,
                    parentLinkIndex=self.ee_index,
                    childBodyUniqueId=target_object_id,
                    childLinkIndex=-1,
                    jointType=p.JOINT_FIXED,
                    jointAxis=[0, 0, 0],
                    parentFramePosition=[0, 0, 0],
                    childFramePosition=[0, 0, 0],
                    parentFrameOrientation=ee_orn
                )
                print("[INFO] Suction grip created.")

            elif gripper_open and self.suction_constraint_id is not None:
                p.removeConstraint(self.suction_constraint_id)
                self.suction_constraint_id = None
                print("[INFO] Suction grip released.")



        min_joint_positions = [p.getJointInfo(self.id, i)[8] for i in range(p.getNumJoints(self.id)) if p.getJointInfo(self.id, i)[2] == p.JOINT_PRISMATIC or p.getJointInfo(self.id, i)[2] == p.JOINT_REVOLUTE]
        max_joint_positions = [p.getJointInfo(self.id, i)[9] for i in range(p.getNumJoints(self.id)) if p.getJointInfo(self.id, i)[2] == p.JOINT_PRISMATIC or p.getJointInfo(self.id, i)[2] == p.JOINT_REVOLUTE]
        joint_ranges = [abs(max_joint_position - min_joint_position) for min_joint_position, max_joint_position in zip(min_joint_positions, max_joint_positions)]
        rest_poses = list((np.array(max_joint_positions) + np.array(min_joint_positions)) / 2)

        ee_target_orientation_q = p.getQuaternionFromEuler(ee_target_orientation_e)

        ee_current_position = p.getLinkState(self.id, self.ee_index, computeForwardKinematics=True)[0]
        ee_current_orientation_q = p.getLinkState(self.id, self.ee_index, computeForwardKinematics=True)[1]
        ee_current_orientation_e = p.getEulerFromQuaternion(ee_current_orientation_q)
        gripper1_current_position = p.getJointState(self.id, gripper1_index)[0]
        gripper2_current_position = p.getJointState(self.id, gripper2_index)[0]

        time_step = 0

        while (not (ee_current_position[0] <= ee_target_position[0] + config.margin_error and ee_current_position[0] >= ee_target_position[0] - config.margin_error and
                    ee_current_position[1] <= ee_target_position[1] + config.margin_error and ee_current_position[1] >= ee_target_position[1] - config.margin_error and
                    ee_current_position[2] <= ee_target_position[2] + config.margin_error and ee_current_position[2] >= ee_target_position[2] - config.margin_error and
                    ee_current_orientation_e[0] <= ee_target_orientation_e[0] + config.margin_error and ee_current_orientation_e[0] >= ee_target_orientation_e[0] - config.margin_error and
                    ee_current_orientation_e[1] <= ee_target_orientation_e[1] + config.margin_error and ee_current_orientation_e[1] >= ee_target_orientation_e[1] - config.margin_error and
                    ee_current_orientation_e[2] <= ee_target_orientation_e[2] + config.margin_error and ee_current_orientation_e[2] >= ee_target_orientation_e[2] - config.margin_error and
                    gripper1_current_position <= gripper_target_position + config.gripper_margin_error and gripper1_current_position >= gripper_target_position - config.gripper_margin_error and
                    gripper2_current_position <= gripper_target_position + config.gripper_margin_error and gripper2_current_position >= gripper_target_position - config.gripper_margin_error)):

            target_joint_positions = p.calculateInverseKinematics(self.id, self.ee_index, ee_target_position, targetOrientation=ee_target_orientation_q, lowerLimits=min_joint_positions, upperLimits=max_joint_positions, jointRanges=joint_ranges, restPoses=rest_poses, maxNumIterations=500)

            if self.robot == "sawyer":
                pass
            elif self.robot == "franka":
                p.setJointMotorControlArray(self.id, range(7), p.POSITION_CONTROL, targetPositions=target_joint_positions[:-2], forces=[config.arm_movement_force_franka] * 7)
                p.setJointMotorControl2(self.id, gripper1_index, p.POSITION_CONTROL, targetPosition=gripper_target_position, force=config.gripper_movement_force_franka)
                p.setJointMotorControl2(self.id, gripper2_index, p.POSITION_CONTROL, targetPosition=gripper_target_position, force=config.gripper_movement_force_franka)
                # 여기서 gripper_target_position을 받고 gripper을 컨트롤 한다 
            elif self.robot == "franka_suction":
                p.setJointMotorControlArray(
                    self.id, range(7), p.POSITION_CONTROL,
                    targetPositions=target_joint_positions[:-2],
                    forces=[config.arm_movement_force_franka] * 7
                )
                p.setJointMotorControl2(
                    self.id, gripper1_index, p.POSITION_CONTROL,
                    targetPosition=gripper_target_position,
                    force=config.gripper_movement_force_franka
                )
                p.setJointMotorControl2(
                    self.id, gripper2_index, p.POSITION_CONTROL,
                    targetPosition=gripper_target_position,
                    force=config.gripper_movement_force_franka
                )
            env.update()
            self.get_camera_image("head", env, save_camera_image=is_trajectory, rgb_image_path=config.rgb_image_trajectory_path.format(step=self.trajectory_step), depth_image_path=config.depth_image_trajectory_path.format(step=self.trajectory_step))
            if is_trajectory:
                self.trajectory_step += 1

            ee_current_position = p.getLinkState(self.id, self.ee_index, computeForwardKinematics=True)[0]
            ee_current_orientation_q = p.getLinkState(self.id, self.ee_index, computeForwardKinematics=True)[1]
            ee_current_orientation_e = p.getEulerFromQuaternion(ee_current_orientation_q)
            gripper1_new_position = p.getJointState(self.id, gripper1_index)[0]
            gripper2_new_position = p.getJointState(self.id, gripper2_index)[0]

            self.ee_current_position = ee_current_position
            self.ee_current_orientation_e = ee_current_orientation_e
            self.gripper_open = gripper_open

            if ((ee_current_position[0] <= ee_target_position[0] + config.margin_error and ee_current_position[0] >= ee_target_position[0] - config.margin_error and
                ee_current_position[1] <= ee_target_position[1] + config.margin_error and ee_current_position[1] >= ee_target_position[1] - config.margin_error and
                ee_current_position[2] <= ee_target_position[2] + config.margin_error and ee_current_position[2] >= ee_target_position[2] - config.margin_error and
                ee_current_orientation_e[0] <= ee_target_orientation_e[0] + config.margin_error and ee_current_orientation_e[0] >= ee_target_orientation_e[0] - config.margin_error and
                ee_current_orientation_e[1] <= ee_target_orientation_e[1] + config.margin_error and ee_current_orientation_e[1] >= ee_target_orientation_e[1] - config.margin_error and
                ee_current_orientation_e[2] <= ee_target_orientation_e[2] + config.margin_error and ee_current_orientation_e[2] >= ee_target_orientation_e[2] - config.margin_error) and
                (not gripper_open) and
                math.isclose(gripper1_new_position, gripper1_current_position, rel_tol=config.rel_tol, abs_tol=config.abs_tol) and
                math.isclose(gripper2_new_position, gripper2_current_position, rel_tol=config.rel_tol, abs_tol=config.abs_tol)):
                break

            gripper1_current_position = gripper1_new_position
            gripper2_current_position = gripper2_new_position

            time_step += 1

            if is_trajectory:
                if time_step > 0:
                    break
            else:
                if time_step > 99:
                    break



    def get_camera_image(self, camera, env, save_camera_image, rgb_image_path, depth_image_path):

        if camera == "wrist":
            camera_position = p.getLinkState(self.id, self.ee_index, computeForwardKinematics=True)[0]
            '''
            id:로봇의 고유 아이디
            eeindex:로봇의 상태
            computeForwardKinematics: True로 설정하면 forward kinematics를 사용한 로봇의 위치와 방향 계산
            '''
            camera_orientation_q = p.getLinkState(self.id, self.ee_index, computeForwardKinematics=True)[1]
            # 걍 쿼터니안 형식일뿐 위랑 차이 없음
        elif camera == "head":
            camera_position = config.head_camera_position
            camera_orientation_q = p.getQuaternionFromEuler(config.head_camera_orientation_e)

        projection_matrix = p.computeProjectionMatrixFOV(fov, aspect, near_plane, far_plane)
        rotation_matrix = np.array(p.getMatrixFromQuaternion(camera_orientation_q)).reshape(3, 3)

        if camera == "wrist":
            init_camera_vector = [0, 0, 1]
            init_up_vector = [1, 0, 0]
        elif camera == "head":
            init_camera_vector = [0, 0, 1]
            init_up_vector = [-1, 0, 0]

        camera_vector = rotation_matrix.dot(init_camera_vector)
        up_vector = rotation_matrix.dot(init_up_vector)
        view_matrix = p.computeViewMatrix(camera_position, camera_position + camera_vector, up_vector)

        image = p.getCameraImage(config.image_width, config.image_height, viewMatrix=view_matrix, projectionMatrix=projection_matrix, renderer=p.ER_BULLET_HARDWARE_OPENGL)
        
        rgb_buffer = image[2]

        depth_buffer = image[3]
        # list of float [0..width*height]
        # https://stackoverflow.com/questions/6652253/getting-the-true-z-value-from-the-depth-buffer 해당 링크를 참고하면 이해가 쉽다.
        # 
        if save_camera_image:
            rgb_image = Image.fromarray(rgb_buffer)
            rgb_image.save(rgb_image_path)

            # =================================================================
            # depth버퍼 실제 거리 데이터로 변환
            n = config.near_plane
            f = config.far_plane
            depth_array = 2 * n * f / (f + n - (2 * depth_buffer - 1.0) * (f - n))

            depth_array = np.clip(depth_array, 0, 1)
            # depth_array의 범위를 클리핑하여 혹시모를 변수 제거
            depth_image = Image.fromarray(depth_array * 255)
            # depth_image를 이미지로 변환
            depth_image.convert("L").save(depth_image_path)

        return camera_position, camera_orientation_q
