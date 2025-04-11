# 메모리 기능 추가

import numpy as np
import matplotlib.pyplot as plt
import sys
import torch
import config
from openai import OpenAI
from PIL import Image
from torchvision import transforms
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks
from dotenv import load_dotenv
import os
import json
import multiprocessing
from PIL import Image
import time
import base64
import main
import requests

sys.path.append("./XMem/")
load_dotenv("openaiAPI.env")
api_key = os.getenv("api_key")

import logging

# 로깅 기본 설정
logging.basicConfig(
    level=logging.INFO,  # 출력할 최소 레벨 설정
    format='%(asctime)s - %(levelname)s - %(message)s',  # 출력 형식 설정
    handlers=[logging.StreamHandler()]  # 콘솔 출력 핸들러
)
logger = logging.getLogger(__name__)



logger = multiprocessing.log_to_stderr()
logger.setLevel(logging.INFO)



from XMem.inference.inference_core import InferenceCore
from XMem.inference.interact.interactive_utils import image_to_torch, index_numpy_to_one_hot_torch, torch_prob_to_numpy_mask, overlay_davis


# ================================================================
# gripper 작동 이미지 저장 및 imgur상에 업로드 
def upload_multiple_images(image_paths, client_id, title=None, description=None):
    '''
    image_paths는 /home/ws/Desktop/VLM_memory_LMTG/language-models-trajectory-generators/girpper_image/robotiqGripper
    이런식을 폴더 내부의 gripper 설명 이미지 경로를 받는다.
    for문을 이용하여 path 내부에 있는 여러 이미지를 모두 업로드 한 후, 업로드 한 url을 return 해준다.
    '''
    image_urls = []
    for path in image_paths:
        try:
            headers = {'Authorization': f'Client-ID {client_id}'}

            # 기본값: 파일 이름을 title로
            if title is None:
                title = os.path.basename(path)
            # 파일명에 사진의 설명을 적어줘야한다.
            if description is None:
                description = f"Uploaded from local path: {path}"

            with open(path, 'rb') as f:
                response = requests.post(
                    'https://api.imgur.com/3/upload',
                    headers=headers,
                    data={
                        'title': title,
                        'description': description
                    },
                    files={'image': f}
                )

            data = response.json()
            print(data)
            url = data['data']['link']
            # print(f"Uploaded: {path} -> {url}")
            image_urls.append(url)
        except Exception as e:
            print(f"Failed to upload {path}: {str(e)}")
    return image_urls
# ================================================================



def memory_chatgpt_output(client, thread_id, assistant_id, prompt, logger):
    """
    Sends a user prompt to the specified thread, runs the assistant,
    waits for the result, and returns the assistant's latest message text.

    Parameters:
    - client: OpenAI client
    - thread_id: ID of the conversation thread
    - assistant_id: ID of the assistant
    - prompt: user message content (string)
    - logger: optional logger object

    Returns:
    - text_string: assistant's latest response (string)
    """

    client.beta.threads.messages.create(
        thread_id=thread_id,
        role="user",
        content=prompt,
    )

    if logger:
        logger.info("Prompt sent to thread")

    run = client.beta.threads.runs.create(
        thread_id=thread_id,
        assistant_id=assistant_id
    )

    while run.status != "completed":
        run = client.beta.threads.runs.retrieve(thread_id=thread_id, run_id=run.id)

    if logger:
        logger.info("Assistant run completed")

    messages = list(client.beta.threads.messages.list(thread_id=thread_id, limit=20))
    last_message = next((msg for msg in messages if msg.role == "assistant"), None)

    if not last_message:
        raise ValueError("No assistant message found in thread.")

    text_string = last_message.content[0].text.value

    if logger:
        logger.info("Assistant response retrieved")
        # logger.debug(f"GPT Output:\n{text_string}")

    return text_string




def encode_image(image_path):
    if image_path.startswith("http://") or image_path.startswith("https://"):
        raise ValueError("URL은 base64 인코딩할 수 없습니다.")
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


# ===============================================================================================================


def memory_chatgpt_output_with_image(client, thread_id, assistant_id, prompt,
                                     image_paths=None, logger=None):
    """
    - prompt: 텍스트 프롬프트
    - image_paths: 하나 또는 여러 개의 이미지 URL 리스트 (string 또는 list of strings)
    """

    # 텍스트 메시지 먼저 추가
    client.beta.threads.messages.create(
        thread_id=thread_id,
        role="user",
        content=[{"type": "text", "text": prompt}]
    )

    if logger:
        # logger.info("프롬프트 메시지를 스레드에 추가함")
        None

    # 이미지 하나만 오는 경우도 리스트로 처리
    if image_paths:
        if isinstance(image_paths, str):
            image_paths = [image_paths]

        for image_url in image_paths:
            client.beta.threads.messages.create(
                thread_id=thread_id,
                role="user",
                content=[
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_url,
                            "detail": "high"
                        }
                    }
                ]
            )
            if logger:
                # logger.info(f"이미지 메시지를 추가함: {image_url}")
                None

    # Assistant 실행
    run = client.beta.threads.runs.create(
        thread_id=thread_id,
        assistant_id=assistant_id
    )

    if logger:
        # logger.info("Assistant 실행 요청 완료")
        None

    # 실행 완료 대기
    while True:
        run = client.beta.threads.runs.retrieve(thread_id=thread_id, run_id=run.id)
        if run.status == "completed":
            break
        elif run.status == "failed":
            logger.error(f"Assistant 실행 실패: {run.last_error}")
            raise RuntimeError("Assistant 실행 실패")
        time.sleep(1)

    if logger:
        # logger.info("Assistant 실행 완료")
        None

    # 응답 가져오기
    messages = list(client.beta.threads.messages.list(thread_id=thread_id, limit=20))
    last_message = next((msg for msg in messages if msg.role == "assistant"), None)

    if not last_message:
        raise ValueError("Assistant의 응답을 찾을 수 없습니다.")

    return last_message.content[0].text.value


# =================================================================================================

def get_langsam_output(image, model, segmentation_texts, segmentation_count):

    segmentation_texts = " . ".join(segmentation_texts)

    # masks, boxes, phrases, logits = model.predict(image, segmentation_texts)
    data= model.predict(image, segmentation_texts)
    output_file_txt = "model_output.txt"
    with open(output_file_txt, "w") as f:
        f.write(str(data))



    result_dict = data 
    # print("result_dict=",result_dict)

    logits = [item['scores'] for item in result_dict]
    phrases = [item['labels'] for item in result_dict]
    boxes = [item['boxes'] for item in result_dict]
    masks = [item['masks'] for item in result_dict]

    logger.info("boxes length = "+ str(len(boxes)))

    output_file = "output_data.txt"
    with open(output_file, "w") as f:
        f.write(f"logits: {logits}\n")
        f.write(f"phrases: {phrases}\n")
        f.write(f"boxes: {boxes}\n")
        f.write(f"masks: {masks}\n")
    print(np.shape(masks))
    _, ax = plt.subplots(1, 1 + len(masks), figsize=(5 + (5 * len(masks)), 5))
    [a.axis("off") for a in ax.flatten()]
    ax[0].imshow(image)



    count = sum(len(sublist) for sublist in boxes)
    colors1 = []
    colors2 = []
    for i in range(count):
        colors1.append("red")
        colors2.append("cyan")
    logger.info("boxes length = "+str(count))
    logger.info("boxes colors = "+str(colors1)+str(colors2))


    for i, (mask, box, phrase) in enumerate(zip(masks, boxes, phrases)):
        to_tensor = transforms.PILToTensor()
        image_tensor = to_tensor(image)
        box = torch.tensor(box)
        # logger.info(box.shape)
        # box = box.unsqueeze(dim=0)
        # logger.info(box.shape)
        # 물체 개수가 변하면 아래 색깔 개수를 바꿔줘야한다.
        image_tensor = draw_bounding_boxes(image_tensor, box, colors=colors1, width=3)
        mask = torch.tensor(mask)
        mask = mask.bool()
        print(f"image_tensor: {image_tensor}")
        image_tensor = draw_segmentation_masks(image_tensor, mask, alpha=0.5, colors=colors2)
        to_pil_image = transforms.ToPILImage()
        image_pil = to_pil_image(image_tensor)

        ax[1 + i].imshow(image_pil)
        ax[1 + i].text(box[0][0], box[0][1] - 15, phrase, color="red", bbox={"facecolor":"white", "edgecolor":"red", "boxstyle":"square"})


    plt.savefig(config.langsam_image_path.format(object=segmentation_count))
    plt.show()

    masks = torch.tensor(masks)
    masks = masks.float()

    return masks, boxes, phrases



def get_chatgpt_output(model, new_prompt, messages, role, file=sys.stdout):
    # model명, 프롬프트, 메세지, 역할, 
    print(role + ":", file=file)
    print(new_prompt, file=file)
    messages.append({"role":role, "content":new_prompt})
    # llm에게 전달해줄 메세지를 편집해준다.

    client = OpenAI(api_key=api_key)

    completion = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=messages,
        stream=True
    )

    print("assistant:", file=file)

    new_output = ""
    # 변수 초기화

    for chunk in completion:
        chunk_content = chunk.choices[0].delta.content
        finish_reason = chunk.choices[0].finish_reason
        if chunk_content is not None:
            print(chunk_content, end="", file=file)
            new_output += chunk_content
        else:
            print("finish_reason:", finish_reason, file=file)

    messages.append({"role":"assistant", "content":new_output})
    main.save_code_block_to_file(messages, file_name="message_blocks.txt")

    return messages



def get_xmem_output(model, device, trajectory_length):

    mask = np.array(Image.open(config.xmem_input_path).convert("L"))
    mask = np.unique(mask, return_inverse=True)[1].reshape(mask.shape)
    # logger.info(f"mask : {mask}")
    logger.info(f"-----------------------------------------------------------")
    mask_image = Image.fromarray(mask.astype(np.uint8))  # 흑백 이미지로 변환
    mask_image.save("mask.png")

    # num_objects = len(np.unique(mask)) - 1
    # 아마 물건이 한개로 설정되는데 위 코드에서 -1해서 물건 개수가 0으로 지정된듯
    num_objects = len(np.unique(mask))

    torch.cuda.empty_cache()
    # 메모리를 비운다.
    processor = InferenceCore(model, config.xmem_config)
    processor.set_all_labels(range(1, num_objects + 1))
    # 추적할 객체 라벨링
    # logger.info(f"trajectory_length: {trajectory_length}, num_objects: {num_objects}")
    masks = []

    with torch.cuda.amp.autocast(enabled=True):

        for i in range(0, trajectory_length + 1, config.xmem_output_every):
            # 설정목표로 가는 각각의 이미지를 하나씩 불러온다. config.xmem_output_every는 1이다.
            frame = np.array(Image.open(config.rgb_image_head_path).convert("RGB"))

            # 경로상의 이미지를 각각 불러와 열어준다.

            frame_torch, _ = image_to_torch(frame, device)
            if i == 0:
                mask_torch = index_numpy_to_one_hot_torch(mask, num_objects + 1).to(device)
                prediction = processor.step(frame_torch, mask_torch[1:])
            else:
                prediction = processor.step(frame_torch)
                # Xmem에 전달한다.

            prediction = torch_prob_to_numpy_mask(prediction)
            masks.append(prediction)

            if i % config.xmem_visualise_every == 0:
                visualisation = overlay_davis(frame, prediction)
                output = Image.fromarray(visualisation)
                output.save(config.xmem_output_path.format(step=i))

    return masks