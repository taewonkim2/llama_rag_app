# 실제 질문이 들어왔을 때 적합한 function을 찾는 과정의 code예시

from llama_index.core.objects import ObjectIndex, SimpleToolNodeMapping
from llama_index.core.agent import FunctionCallingAgentWorker
from llama_index.core.tools import FunctionTool 

# 1. 수천만 개의 툴이 배열에 있다고 가정 (실제로는 DB나 레지스트리에서 관리)
tool1 = FunctionTool.from_defaults(fn=get_current_time)
tool2 = FunctionTool.from_defaults(fn=get_weather)

all_tools = [tool1, tool2, ...]

# 2. 툴들의 '설명(Docstring)'을 기반으로 인덱스(검색 엔진)를 생성
tool_mapping = SimpleToolNodeMapping.from_objects(all_tools)
tool_index = ObjectIndex.from_objects(all_tools, tool_mapping, index_cls=VectorStoreIndex)

# 3. 질문이 들어오면 자동으로 관련 툴만 '검색'해주는 녀석을 만듦 (핵심!)
tool_retriever = tool_index.as_retriever(similarity_top_k=3)

# 4. 에이전트는 이제 수천만 개가 아니라, 그때그때 검색된 3개의 툴만 보고 판단함
agent_worker = FunctionCallingAgentWorker.from_tools(
    tool_retriever=tool_retriever, # 모든 툴 대신 검색기를 쥐여줌!
    llm=llm
)
agent = agent_worker.as_agent()

# 실행하면 내부적으로 [툴 검색] -> [최적의 툴 배치] -> [호출]이 한 번에 일어남
resp = agent.chat("What is the current time in New York?")