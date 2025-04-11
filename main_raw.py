# 원본 메인코드

import numpy as np
import math
import openai
import torch
import os
import sys
import argparse
import traceback
import multiprocessing
import logging
import functools
import models
import config
from lang_sam import LangSAM
from multiprocessing import Process, Pipe
from io import StringIO
from contextlib import redirect_stdout
from api import API
from env import run_simulation_environment
from prompts.main_prompt import MAIN_PROMPT
from prompts.error_correction_prompt import ERROR_CORRECTION_PROMPT
from prompts.print_output_prompt import PRINT_OUTPUT_PROMPT
from prompts.task_failure_prompt import TASK_FAILURE_PROMPT
from prompts.task_summary_prompt import TASK_SUMMARY_PROMPT
from config import OK, PROGRESS, FAIL, ENDC

sys.path.append("./XMem/")
print = functools.partial(print, flush=True)

from XMem.model.network import XMem
import os
from dotenv import load_dotenv


def save_code_block_to_file(code_block, file_name="code_blocks.txt"):
    with open(file_name, "a") as file:
        file.write(str(code_block))  # 코드 블록을 파일에 저장
        file.write("\n\n")  




load_dotenv("openaiAPI.env")
api_key = os.getenv("api_key")
# api_key가져오기

if __name__ == "__main__":

    openai.api_key = api_key

    # Parse args
    parser = argparse.ArgumentParser(description="Main Program.")
    parser.add_argument("-lm", "--language_model", choices=["gpt-4o-mini", "gpt-4-32k", "gpt-3.5-turbo", "gpt-3.5-turbo-16k"], default="gpt-4o-mini", help="select language model")
    parser.add_argument("-r", "--robot", choices=["sawyer", "franka"], default="sawyer", help="select robot")
    parser.add_argument("-m", "--mode", choices=["default", "debug"], default="default", help="select mode to run")
    args = parser.parse_args()

    # Logging
    logger = multiprocessing.log_to_stderr()
    logger.setLevel(logging.INFO)

    # Device
    if torch.cuda.is_available():
        logger.info("Using GPU.")
        device = torch.device("cuda")
    else:
        logger.info("CUDA not available. Please connect to a GPU instance if possible.")
        device = torch.device("cpu")

    torch.set_grad_enabled(False)

    # Load models
    langsam_model = LangSAM()
    xmem_model = XMem(config.xmem_config, "./XMem/saves/XMem.pth", device).eval().to(device)
    # 모델 로드
    
    # API set-up
    main_connection, env_connection = Pipe()
    # 얘가 핵심인듯
    # main_connection은 우리가 pybullet상에서 실행된 결과를 받기 위한 파이프 끝점이다.
    # env_connection은 pybullet상에서 env_process가 pybullet상에서 실행된 결과를 보내기 위한 파이프 끝점이다.
    api = API(args, main_connection, logger, langsam_model, xmem_model, device)

    detect_object = api.detect_object
    execute_trajectory = api.execute_trajectory
    open_gripper = api.open_gripper
    close_gripper = api.close_gripper
    task_completed = api.task_completed
    
    
    # Start process
    env_process = Process(target=run_simulation_environment, name="EnvProcess", args=[args, env_connection, logger])
    env_process.start()

    [env_connection_message] = main_connection.recv() # recv means receive data from main_connection
    logger.info(env_connection_message)

    # User input
    command = input("Enter a command: ")
    api.command = command

    # ChatGPT
    logger.info(PROGRESS + "STARTING TASK..." + ENDC)

    messages = []

    error = False

    new_prompt = MAIN_PROMPT.replace("[INSERT EE POSITION]", str(config.ee_start_position)).replace("[INSERT TASK]", command)
    # 메인 프롬프트에서 비어있는 곳을 수정한다.
    # EE POSITION: config.ee_start_position, TASK: command


    logger.info(PROGRESS + "Generating ChatGPT output..." + ENDC) #ENDC means that Enter

    messages = models.get_chatgpt_output(args.language_model, new_prompt, messages, "system")
    # 언어 모델, 프롬프트를 정하여 정해준다.
    # 이 함수에서 메모리 기능을 적용해야한다.
    
    logger.info(OK + "Finished generating ChatGPT output!" + ENDC)

    while True:

        while not api.completed_task:

            new_prompt = ""

            if len(messages[-1]["content"].split("```python")) > 1:
                # llm이 전달해준 메세지를 자른다. 
                code_block = messages[-1]["content"].split("```python")
                #   {"role": "assistant", "content": "```python\nprint('Hello, World!')\n```"} 꼴의 데이터에서 'Hello, World!'를 가져온다,
                #   코드가 리턴되므로 코드 블럭이라는 변수에 저장해준다.
                block_number = 0
                save_code_block_to_file(code_block)
                for block in code_block:
                    if len(block.split("```")) > 1:
                        # 생성된 코드문을 받은 후, ''' -------''' 기준으로 쪼개 실행한다.
                        code = block.split("```")[0]
                        save_code_block_to_file(code)
                        block_number += 1
                        try:
                            f = StringIO()
                            with redirect_stdout(f):
                                exec(code)
                    # 여기서 받은 코드를 실행하는듯 하다.
                    # 만약 llm이 detect_object("box")를 실행하기로 결정한다면, 위에서 정의한 detect_object = api.detect_object가 실행된다.
                        except Exception:
                            error_message = traceback.format_exc()
                            new_prompt += ERROR_CORRECTION_PROMPT.replace("[INSERT BLOCK NUMBER]", str(block_number)).replace("[INSERT ERROR MESSAGE]", error_message)
                            # 에러메세지를 다시 전달한다. 
                            new_prompt += "\n"
                            error = True
                        else:
                            s = f.getvalue()
                            error = False
                            if s != "" and len(s) < 2000:
                                logger.info("s is :" + s)
                                new_prompt += PRINT_OUTPUT_PROMPT.replace("[INSERT PRINT STATEMENT OUTPUT]", s)
                                new_prompt += "\n"
                                error = True
                            
            if error:

                api.completed_task = False
                api.failed_task = False

            if not api.completed_task:
                logger.info(f"completed_task is False")
                if api.failed_task:
                    logger.info(f("failed_task is True"))
                    logger.info(FAIL + "FAILED TASK! Generating summary of the task execution attempt..." + ENDC)

                    new_prompt += TASK_SUMMARY_PROMPT
                    # 이전 프롬프트에 실패했다는 내용을 더하여 전달
                    new_prompt += "\n"
                    # 작동 실패시 이전 내용을 요약하여 다시 리턴
                    logger.info(PROGRESS + "Generating ChatGPT output..." + ENDC)
                    messages = models.get_chatgpt_output(args.language_model, new_prompt, messages, "user")
                    # 실패한 내용과, 이전 메세지를 같이 전달
                    logger.info(OK + "Finished generating ChatGPT output!" + ENDC)

                    logger.info(PROGRESS + "RETRYING TASK..." + ENDC)

                    new_prompt = MAIN_PROMPT.replace("[INSERT EE POSITION]", str(config.ee_start_position)).replace("[INSERT TASK]", command)
                    new_prompt += "\n"
                    new_prompt += TASK_FAILURE_PROMPT.replace("[INSERT TASK SUMMARY]", messages[-1]["content"])

                    messages = []

                    error = False

                    logger.info(PROGRESS + "Generating ChatGPT output..." + ENDC)
                    messages = models.get_chatgpt_output(args.language_model, new_prompt, messages, "system")
                    logger.info(OK + "Finished generating ChatGPT output!" + ENDC)

                    api.failed_task = False

                else:

                    logger.info(PROGRESS + "Generating ChatGPT output..." + ENDC)
                    # print(f"new_prompt: {new_prompt}")
                    messages = models.get_chatgpt_output(args.language_model, new_prompt, messages, "user")
                    print(f"messages: {messages}")
                    logger.info(OK + "Finished generating ChatGPT output!" + ENDC)

        logger.info(OK + "FINISHED TASK!" + ENDC)

        new_prompt = input("Enter a command: ")
        # 실행 마무리되면 원래 다음 명령 넣는듯
        logger.info(PROGRESS + "Generating ChatGPT output..." + ENDC)
        messages = models.get_chatgpt_output(args.language_model, new_prompt, messages, "user")
        logger.info(OK + "Finished generating ChatGPT output!" + ENDC)

        api.completed_task = False
        # 완료 아님으로 다시 리턴