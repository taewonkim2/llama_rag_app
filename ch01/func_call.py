from llama_index.core.tools import FunctionTool
from llama_index.core.llms import ChatMessage
from llama_index.llms.google_genai import GoogleGenAI
from datetime import datetime

llm = GoogleGenAI(model="gemini-2.5-flash")

def get_current_time(timezone: str) -> dict:
    """Get the current time"""
    return {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "timezone": timezone,
    }

# uses the tool name, any type annotations, and docstring to describe the tool
tool = FunctionTool.from_defaults(fn=get_current_time)

chat_history = [
    ChatMessage(role="user", content="What is the current time in New York?")
]
tools_by_name = {t.metadata.name: t for t in [tool]}

resp = llm.chat_with_tools([tool], chat_history=chat_history)
tool_calls = llm.get_tool_calls_from_response(
    resp, error_on_no_tool_call=False
)

if not tool_calls:
    print(resp)
else:
    while tool_calls:
        # add the LLM's response to the chat history
        chat_history.append(resp.message)

        for tool_call in tool_calls:
            tool_name = tool_call.tool_name
            tool_kwargs = tool_call.tool_kwargs

            print(f"Calling {tool_name} with {tool_kwargs}")
            tool_output = tool.call(**tool_kwargs)
            print("Tool output: ", tool_output)
            chat_history.append(
                ChatMessage(
                    role="tool",
                    content=str(tool_output),
                    # most LLMs like Gemini, Anthropic, OpenAI, etc. need to know the tool call id
                    additional_kwargs={"tool_call_id": tool_call.tool_id},
                )
            )

            resp = llm.chat_with_tools([tool], chat_history=chat_history)
            tool_calls = llm.get_tool_calls_from_response(
                resp, error_on_no_tool_call=False
            )
    print("Final response: ", resp.message.content)