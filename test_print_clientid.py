import os
from dotenv import load_dotenv

# .env 파일 불러오기 (현재 경로 기준 또는 절대경로)
load_dotenv(dotenv_path="openaiAPI.env")

# 환경변수에서 불러오기
client_id = os.getenv("client_id")  # .env에 client_id=... 라고 되어 있어야 함

# 출력
print(f"Loaded Client ID: {client_id}")
