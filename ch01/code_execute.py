from llama_index.llms.google_genai import GoogleGenAI
from llama_index.core.llms import ChatMessage
from google.genai import types

# 구글 제미나이가 지원하는 내장 기능인 '코드 실행기(ToolCodeExecution)'를 생성
code_execution_tool = types.Tool(code_execution=types.ToolCodeExecution())

llm = GoogleGenAI(
    model="gemini-2.5-flash",
    built_in_tool=code_execution_tool,
)

# resp = llm.complete("Calculate 20th fibonacci number.")
# Request a calculation that will likely use code execution

messages = [
    ChatMessage(
        role="user", content="What is the sum of the first 50 prime numbers?"
    )
]

resp = llm.chat(messages)

# 제미나이가 돌려준 응답에 LlamaIndex 가공 전의 날것 데이터(raw)가 존재하는지, 그리고 그 안에 알맹이 내용(content)이 잘 들어있는지 체크
if hasattr(resp, "raw") and "content" in resp.raw:
    #[조각 1: "제가 코드를 짜볼게요"], [조각 2: 실제 실행한 파이썬 코드], \
    #[조각 3: 컴퓨터가 뱉은 실행 결과], [조각 4: "따라서 정답은 XXX입니다"] 이런 식입니다. 그 조각 리스트를 가져오는 code임
    parts = resp.raw["content"].get("parts", [])

    for i, part in enumerate(parts):
        print(f"Part {i+1}:")

        if "text" in part and part["text"]:
            print(f"  Text: {part['text'][:100]}", end="")
            print(" ..." if len(part["text"]) > 100 else "")
        # 만약 현재 조각에 AI가 문제를 풀기 위해 직접 작성한 파이썬 소스코드가 들어있다면, 그 코드를 화면에 그대로 출력
        if "executable_code" in part and part["executable_code"]:
            print(f"  Executable Code: {part['executable_code']}")
        # AI가 짠 코드가 가상 컴퓨터 환경에서 실행되어 나온 실제 터미널 출력값(콘솔 결과)이 들어있다면, 그 최종 계산된 숫자나 결과값을 출력
        if "code_execution_result" in part and part["code_execution_result"]:
            print(f"  Code Result: {part['code_execution_result']}")
else:
    print("No detailed parts found in raw response")

