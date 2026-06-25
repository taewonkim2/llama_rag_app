from llama_index.llms.google_genai import GoogleGenAI
import google.genai.types as types

config = types.GenerateContentConfig(
    temperature=0.1, response_modalities=["Text", "Image"]
)
llm = GoogleGenAI(
    model="gemini-2.5-flash-image-preview", generation_config=config
)

from llama_index.core.llms import ChatMessage, TextBlock, ImageBlock
messages = [
    ChatMessage(role="user", content="Please generate an image of a cute dog")
]

resp = llm.chat(messages)

from PIL import Image
from IPython.display import display

# 미나이가 *"귀여운 강아지 사진입니다!"*라는 글자와 함께 진짜 강아지 사진 파일 1장을 보냈다면, 
# blocks 리스트 안에는 [텍스트 블록, 이미지 블록] 이렇게 두 개가 순서대로 들어있습니다. 이 조각들을 하나씩 꺼내서 block이라는 변수에 담고 루프를 돕니다
for block in resp.message.blocks:
    if isinstance(block, ImageBlock):
        image = Image.open(block.resolve_image())
        display(image)
    elif isinstance(block, TextBlock):
        print(block.text)
# AI가 멀티모달(텍스트 + 이미지)로 대답을 주기 때문에, 파이썬 코드단에서도 
# 그 모달리티를 종류별로 분리 수거해서 맞춤형(글자는 print, 사진은 display)으로 보여주려고 저 복잡한 for문을 돌리는 것