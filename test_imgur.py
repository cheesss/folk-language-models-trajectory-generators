import os
import requests
from dotenv import load_dotenv

# 1. .env 파일에서 환경변수 불러오기
load_dotenv(dotenv_path="openaiAPI.env")

# 2. 환경변수에서 Client-ID 가져오기
client_id = os.getenv("client_id")  # 환경변수 이름은 여기에 맞게 조정하세요
image_path = "gripper_image/robotiq3Finger/1000031344.png"  # 테스트할 이미지 파일 경로

# 3. 유효성 확인
if not client_id:
    print("❌ Client-ID가 환경변수에서 불러와지지 않았습니다.")
    exit(1)

# 4. Imgur 업로드 요청
headers = {'Authorization': f'Client-ID {client_id}'}
with open(image_path, 'rb') as f:
    files = {'image': ('test.png', f, 'image/png')}
    response = requests.post('https://api.imgur.com/3/image', headers=headers, files=files)

# 5. 결과 출력
print(f"✅ Status Code: {response.status_code}")
if response.ok:
    try:
        image_url = response.json()['data']['link']
        print(f"✅ Upload 성공! Image URL: {image_url}")
    except Exception as e:
        print(f"⚠️ JSON 파싱 실패: {e}")
        print("🔍 응답 내용:\n", response.text[:300])
else:
    print(f"❌ 업로드 실패! 상태코드: {response}")
    print("🔍 응답 내용:\n", response.text[:300])
