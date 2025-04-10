import numpy as np
import sys
import torch
import math
import config
import models
import utils
from realsenseCapture import IntelCamera
from PIL import Image
from prompts.success_detection_prompt import SUCCESS_DETECTION_PROMPT
from config import OK, PROGRESS, FAIL, ENDC
from config import CAPTURE_IMAGES, ADD_BOUNDING_CUBES, ADD_TRAJECTORY_POINTS, EXECUTE_TRAJECTORY, OPEN_GRIPPER, CLOSE_GRIPPER, SUCTION, TASK_COMPLETED, RESET_ENVIRONMENT
# 멀티프로세싱 넘버 불러오기
from io import StringIO
from contextlib import redirect_stdout
import requests
import os
from dotenv import load_dotenv


depth_scale = 0.0010000000474974513
def save_code_block_to_file(code_block, file_name="code_blocks.txt"):
    with open(file_name, "a") as file:
        file.write(str(code_block))  # 코드 블록을 파일에 저장
        file.write("\n\n")  

class API:

    def __init__(self, args, main_connection, logger, langsam_model, xmem_model, device, client, thread, assistant):

        self.args = args
        self.main_connection = main_connection
        self.logger = logger
        self.langsam_model = langsam_model
        self.xmem_model = xmem_model
        self.device = device
        self.client = client
        self.thread = thread
        self.assistant = assistant
        self.segmentation_texts = []
        self.segmentation_count = 0
        self.trajectory_length = 0
        self.attempted_task = False
        self.completed_task = False
        self.failed_task = False
        self.head_camera_position = None
        self.head_camera_orientation_q = None
        self.wrist_camera_position = None
        self.wrist_camera_orientation_q = None
        self.command = None

# ================================================================
    def ENVUnderstand(self):
        '''
        prompt 상에서 LLM에게 "실제 환경을 파악하기 위하여 이미지를 전달하는 함수"로 알려준 함수이다.
        Pybullet camera에서 촬영한 
        '''
        load_dotenv("openaiAPI.env")
        client_id = os.getenv("client_id")
        self.logger.info(PROGRESS + "Capturing head and wrist camera images..." + ENDC)
        self.main_connection.send([CAPTURE_IMAGES])
        [head_camera_position, head_camera_orientation_q, wrist_camera_position, wrist_camera_orientation_q, env_connection_message] = self.main_connection.recv()
        self.logger.info(env_connection_message)

        self.head_camera_position = head_camera_position

        # 이미지 업로드 기능
        if client_id:
            # RGB 이미지 업로드
            headers = {'Authorization': f'Client-ID {client_id}'}
            with open(config.rgb_image_head_path, 'rb') as f:
                response = requests.post(
                    'https://api.imgur.com/3/upload',
                    headers=headers,
                    files={'image': f}
                )
            data = response.json()
            image_url = data['data']['link']
            
            # URL을 파일에 저장
            with open("image_url.txt", "w") as f:
                f.write(image_url)
            
            self.logger.info(f"Image URL is saved: {image_url}")
            print(f"Image URL is saved: {image_url}")
            return image_url
        else:
            print("Image URL is not saved.")
            return None
            
    def get_image_url(self):
        """저장된 이미지 URL을 반환합니다. URL 파일이 없으면 None을 반환합니다."""
        try:
            with open("image_url.txt", "r") as f:
                return f.read().strip()
        except FileNotFoundError:
            return None
            
    def delete_image_url(self):
        """저장된 이미지 URL 파일을 삭제합니다."""
        try:
            os.remove("image_url.txt")
            self.logger.info("이미지 URL 파일이 삭제되었습니다.")
            return True
        except FileNotFoundError:
            return False
# ================================================================


# ================================================================

    def detect_object(self, segmentation_text):

        self.logger.info(PROGRESS + "Capturing head and wrist camera images..." + ENDC)
        self.main_connection.send([CAPTURE_IMAGES])
        [head_camera_position, head_camera_orientation_q, wrist_camera_position, wrist_camera_orientation_q, env_connection_message] = self.main_connection.recv()
        # main_connection.send([CAPTURE_IMAGES])를 통해 받은 env의 아웃풋을 5개의 변수에 저장해준다.
        # 이를 위해 recv() 함수를 사용한다. = 출력값을 요구하는 함수

        self.logger.info(env_connection_message)

        self.head_camera_position = head_camera_position
        self.head_camera_orientation_q = head_camera_orientation_q
        self.wrist_camera_position = wrist_camera_position
        self.wrist_camera_orientation_q = wrist_camera_orientation_q

        # 원본 코드
        # rgb_image_head = Image.open(config.rgb_image_head_path).convert("RGB")
        # depth_image_head = Image.open(config.depth_image_head_path).convert("L")

        # realsense 적용코드
        rgba_image, depth_image, depth_intrinsics = IntelCamera.capture_save_image()
        rgb_image_head_path = config.rgb_image_head_path
        rgb_image_head = Image.open(rgb_image_head_path).convert("RGB")

        depth_image_head_path = config.depth_image_head_path
        depth_image_head = Image.open(depth_image_head_path).convert("L")
        depth_array = np.array(depth_image_head) * depth_scale
        # depth_array = depth_image_head
        # self.logger.info("depth_array min is "+str(np.average(depth_array)))


        if self.segmentation_count == 0:
            xmem_image = Image.fromarray(np.zeros_like(depth_array)).convert("L")
            xmem_image.save(config.xmem_input_path)

        segmentation_texts = [segmentation_text]
        # point = ["."]
        # segmentation_text = segmentation_texts + point
        self.logger.info(PROGRESS + "Segmenting head camera image..." + ENDC)
        # print("this is test", rgb_image_head, self.langsam_model, segmentation_texts, self.segmentation_count)
        # self.logger.info("segmantation_texts: " + str(segmentation_texts)+ str(type(segmentation_texts)))
        self.logger.info("segmentation_texts: "+ str(segmentation_text))
        print("segmentation_texts: ", segmentation_texts)
        model_predictions, boxes, segmentation_texts = models.get_langsam_output(rgb_image_head, self.langsam_model, segmentation_texts, self.segmentation_count)
        self.logger.info(OK + "Finished segmenting head camera image!" + ENDC)

        masks = utils.get_segmentation_mask(model_predictions, config.segmentation_threshold)
        # self.logger.info("mask reasult is "+ str(masks))

        # 예측결과를 이진화하여(True, False) 마스크 내부 예측 결과를 확정한다.

        bounding_cubes_world_coordinates, bounding_cubes_orientations,contour_pixel_points = utils.get_bounding_cube_from_point_cloud(rgb_image_head, masks, depth_array, self.head_camera_position, self.head_camera_orientation_q, depth_image, depth_intrinsics, self.segmentation_count)
        # 여기서 주는 depth array가 거리 관련 데이터인듯
        # self.logger.info("bounding_cubes_world_coordinates: "+str(bounding_cubes_world_coordinates))

        utils.save_xmem_image(masks)

        self.segmentation_texts.extend(segmentation_texts)

        self.logger.info(PROGRESS + "Adding bounding cubes to the environment..." + ENDC)
        self.main_connection.send([ADD_BOUNDING_CUBES, bounding_cubes_world_coordinates])
        # main_connection.send([ADD_BOUNDING_CUBES, bounding_cubes_world_coordinates])를 통해 env_connection에 bounding_cubes_world_coordinates 전송
        # get_bounding_cube_from_point_cloud 함수에서 리턴 받은 box 정보를 env_connection에 전달하여 pybullet상에 반영
        [env_connection_message] = self.main_connection.recv()
        self.logger.info(env_connection_message)

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

        self.segmentation_count += 1


    def execute_trajectory(self, trajectory):

        self.logger.info(PROGRESS + "Adding trajectory points to the environment..." + ENDC)
        self.main_connection.send([ADD_TRAJECTORY_POINTS, trajectory])

        self.logger.info(PROGRESS + "Executing generated trajectory..." + ENDC)
        self.main_connection.send([EXECUTE_TRAJECTORY, trajectory])

        self.trajectory_length += len(trajectory)
        # self.logger.info(f"Trajectory length: {trajectory}")


    def open_gripper(self):

        self.logger.info(PROGRESS + "Opening gripper..." + ENDC)
        self.main_connection.send([OPEN_GRIPPER])
        # Pipe()의 소켓에 오픈 그리퍼를 전달한다.


    def close_gripper(self):

        self.logger.info(PROGRESS + "Closing gripper..." + ENDC)
        self.main_connection.send([CLOSE_GRIPPER])

    def suction(self):

        self.logger.info(PROGRESS + "suction..." + ENDC)
        self.main_connection.send([SUCTION])

    def scissor_fingertip_grasp(self):
        self.logger.info(PROGRESS + "Executing scissor fingertip grasp..." + ENDC)
        self.main_connection.send([CLOSE_GRIPPER])

    def basic_fingertip_grasp(self):
        self.logger.info(PROGRESS + "Executing basic fingertip grasp..." + ENDC)
        self.main_connection.send([CLOSE_GRIPPER])

    def basic_encompassing_grasp(self):
        self.logger.info(PROGRESS + "Executing basic encompassing grasp..." + ENDC)
        self.main_connection.send([CLOSE_GRIPPER])

    def wide_encompassing_grasp(self):
        self.logger.info(PROGRESS + "Executing wide encompassing grasp..." + ENDC)
        self.main_connection.send([CLOSE_GRIPPER])

    def wide_fingertip_grasp(self):
        self.logger.info(PROGRESS + "Executing wide fingertip grasp..." + ENDC)
        self.main_connection.send([CLOSE_GRIPPER])

    def pinch_fingertip_grasp(self):
        self.logger.info(PROGRESS + "Executing pinch fingertip grasp..." + ENDC)
        self.main_connection.send([CLOSE_GRIPPER])

    def suctionOnly(self):
        self.logger.info(PROGRESS + "Activating suction only..." + ENDC)
        self.main_connection.send([SUCTION])

    def graspOnly(self):
        self.logger.info(PROGRESS + "Activating grasp only..." + ENDC)
        self.main_connection.send([CLOSE_GRIPPER])

    def suctionANDgrasp(self):
        self.logger.info(PROGRESS + "Activating suction and grasp..." + ENDC)
        self.main_connection.send([CLOSE_GRIPPER])

    def suctionRelease(self):
        self.logger.info(PROGRESS + "Releasing suction..." + ENDC)
        self.main_connection.send([OPEN_GRIPPER])




    def task_completed(self):

        if self.attempted_task:

            self.completed_task = True

        else:

            self.logger.info(PROGRESS + "Waiting to execute all generated trajectories..." + ENDC)
            self.main_connection.send([TASK_COMPLETED])
            [env_connection_message] = self.main_connection.recv()
            self.logger.info(env_connection_message)

            self.logger.info(PROGRESS + "Generating XMem output..." + ENDC)
            # self.logger.info(f"Trajectory length: {self.trajectory_length}")
            # tragectory_length란 몇개의 경로지점을 생성하여 수행했는지 보여준다.
            masks = models.get_xmem_output(self.xmem_model, self.device, self.trajectory_length)
            # 여기서 xmem에게 이미지를 전달해준 후 성공여부를 확인한다. 
            self.logger.info(OK + "Finished generating XMem output!" + ENDC)

            num_objects = len(np.unique(masks[0])) - 1
            # num_objects = len(np.unique(masks[0]))
            # 여기 또 있네

            new_prompt = SUCCESS_DETECTION_PROMPT.replace("[INSERT TASK]", self.command)
            new_prompt += "\n"

            self.logger.info(PROGRESS + "Calculating object bounding cubes..." + ENDC)

            for object in range(1, num_objects + 1):

                object_positions = []
                object_orientations = []

                idx_offset = 0

                for i, mask in enumerate(masks):
                    # enumeratesms mask 내부의 인덱스와 데이터를 동시에 불러온다.

                    # rgb_image = Image.open(config.rgb_image_trajectory_path.format(step=i * config.xmem_output_every)).convert("RGB")
                    # depth_image = Image.open(config.depth_image_trajectory_path.format(step=i * config.xmem_output_every)).convert("L")
                    # 원본코드

                    rgb_image = Image.open(config.rgb_image_path.format(step=i * config.xmem_output_every)).convert("RGB")
                    depth_image = Image.open(config.depth_image_path.format(step=i * config.xmem_output_every)).convert("L")
                    depth_array = np.array(depth_image) / 255.

                    object_mask = mask.copy()
                    object_mask[object_mask != object] = False
                    object_mask[object_mask == object] = True
                    object_mask = torch.Tensor(object_mask)

                    bounding_cubes, orientations = utils.get_bounding_cube_from_point_cloud(
                        rgb_image, 
                        [object_mask], 
                        depth_array, 
                        self.head_camera_position, 
                        self.head_camera_orientation_q, 
                        object - 1
                    )
                    if len(bounding_cubes) == 0:
                        
                        self.logger.info("No bounding cube found: removed.")
                        idx_offset += 1

                    else:

                        [bounding_cube] = bounding_cubes
                        [orientation] = orientations
                        position = bounding_cube[4]
                        orientation = orientation[0]
                        orientation = np.mod(orientation + math.pi, 2 * math.pi) - math.pi

                        object_positions.append(position)

                        if i == 0:

                            object_orientations.append(orientation)

                        else:

                            previous_orientation = object_orientations[i - 1 - idx_offset]
                            possible_orientations = np.array([np.mod(orientation + i * math.pi / 2 + math.pi, 2 * math.pi) - math.pi for i in range(4)])
                            circular_difference = np.minimum(np.abs(possible_orientations - previous_orientation), 2 * math.pi - np.abs(possible_orientations - previous_orientation))
                            min_index = np.argmin(circular_difference)
                            orientation = possible_orientations[min_index]
                            object_orientations.append(orientation)

                # new_prompt += self.segmentation_texts[object - 1] + " trajectory positions and orientations:\n"
                # 여기서 물건의 위치를 Xmem을 이용하여 추적 후 LangSAM으로 찾고, 바운딩 박스를 다시 그려 gpt에게 다시 전달해준다. 
                new_prompt += "".join(self.segmentation_texts[object - 1]) + " trajectory positions and orientations:\n"
                new_prompt += "Positions:\n"
                new_prompt += str(np.around([position for p, position in enumerate(object_positions) if p % config.xmem_lm_input_every == 0], 3)) + "\n"
                new_prompt += "Orientations:\n"
                new_prompt += str(np.around([orientation for o, orientation in enumerate(object_orientations) if o % config.xmem_lm_input_every == 0], 3)) + "\n"
                new_prompt += "\n"
                # 여기선 SUCCESS_DETECTION_PROMPT를 쓰는데, 그냥 그대로 써도 될듯함
                
            self.logger.info(OK + "Finished calculating object bounding cubes!" + ENDC)

            self.attempted_task = True

            messages = []

            self.logger.info(PROGRESS + "Generating ChatGPT output..." + ENDC)
            # messages = models.get_chatgpt_output(self.args.language_model, new_prompt, messages, "system", file=sys.stderr)
            text_string = models.memory_chatgpt_output(
                client=self.client,
                thread_id=self.thread.id,
                assistant_id=self.assistant.id,
                prompt=new_prompt,
                logger=self.logger  # 기존 로깅도 그대로 사용 가능
            )

            self.logger.info(OK + "Finished generating ChatGPT output!" + ENDC)

            # code_block = messages[-1]["content"].split("```python")

            # task_completed = self.task_completed
            # task_failed = self.task_failed

            # for block in code_block:
            #     if len(block.split("```")) > 1:
            #         code = block.split("```")[0]
            #         exec(code)
            if "```python" in text_string:
                code_block = text_string.split("```python")
                block_number = 0
                for block in code_block:
                    if "```" in block:
                        code = block.split("```")[0].strip()  # 코드만 깔끔하게 추출
                        save_code_block_to_file(code)        # 저장 함수 호출
                        block_number += 1
                        try:
                            f = StringIO()
                            with redirect_stdout(f):
                                exec(code)
                        except Exception:
                            error = True


    def task_failed(self):

        self.failed_task = True

        self.logger.info(PROGRESS + "Resetting environment..." + ENDC)
        self.main_connection.send([RESET_ENVIRONMENT])
        [env_connection_message] = self.main_connection.recv()
        self.logger.info(env_connection_message)

        self.segmentation_count = 0
        self.trajectory_length = 0
        self.segmentation_texts = []
        self.attempted_task = False
