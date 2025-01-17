from model_configurations import get_model_configuration

from langchain_openai import AzureChatOpenAI
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage, AIMessage
from langchain import hub
from langchain.agents import AgentExecutor, create_openai_functions_agent, create_tool_calling_agent
from langchain_core.tools import tool

from typing import List

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic import BaseModel, Field
from langchain_core.runnables.history import RunnableWithMessageHistory

import base64
from mimetypes import guess_type
import requests

gpt_chat_version = 'gpt-4o'
gpt_config = get_model_configuration(gpt_chat_version)

llm = AzureChatOpenAI(
        model=gpt_config['model_name'],
        deployment_name=gpt_config['deployment_name'],
        openai_api_key=gpt_config['api_key'],
        openai_api_version=gpt_config['api_version'],
        azure_endpoint=gpt_config['api_base'],
        temperature=gpt_config['temperature'],
        model_kwargs={ "response_format": { "type": "json_object" } }
)

def generate_hw01(question):
    final_prompt = ChatPromptTemplate([
            ("system", "請用JSON格式回答問題，不需要markdown標記：結果爲Result，日期爲date，紀念日名稱爲name"),
            ("user", "{input}"),
    ])
    chain = final_prompt | llm
    response = chain.invoke({"input": question})
    return response.content

def generate_hw02(question):
    instructions = "請用JSON格式作答"
    base_prompt = hub.pull("langchain-ai/openai-functions-template")
    prompt = base_prompt.partial(instructions=instructions)

    @tool
    def get_holidays(country: str, year: int, month: int):
        """得到指定國家、年份、月份的紀念日列表(JSON格式呈現)"""
        url = f"https://calendarific.com/api/v2/holidays?&api_key=NplCnEzX4afR3qpnwFF858d1S05XvzP7&country={country}&year={year}&month={month}"
        response = requests.get(url)
        question = response.text
        return ChatPromptTemplate([
            ("system", """請找到holidays下的所有日期和中文名稱，并保存到Result中，如：{"Result": [
             {
                 "date": "2024-10-10",
                 "name": "國慶日"
             }]}"""),
            ("user", question)
        ])   

    tools = [get_holidays]
    agent = create_openai_functions_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False)
    response = agent_executor.invoke({"input": question})
    return response["output"]

class InMemoryHistory(BaseChatMessageHistory, BaseModel):
    """In memory implementation of chat message history."""

    messages: List[BaseMessage] = Field(default_factory=list)

    def add_messages(self, messages: List[BaseMessage]) -> None:
        """Add a list of messages to the store"""
        self.messages.extend(messages)

    def clear(self) -> None:
        self.messages = []

# Here we use a global variable to store the chat message history.
# This will make it easier to inspect it to see the underlying results.
store = {}

def get_by_session_id(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryHistory()
    return store[session_id]

def generate_hw03(question2, question3):
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Respond in json format. {requirement}"),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
        ("ai", "{ai_response}"),
    ])

    chain = prompt | llm
    chain_with_history = RunnableWithMessageHistory(
        chain,
        get_by_session_id,
        input_messages_key="question",
        history_messages_key="history",
    )

    response = chain_with_history.invoke(  # noqa: T201
        {"requirement": "", "question": question2, "ai_response": generate_hw02(question2)},
        config={"configurable": {"session_id": "foo"}}
    )

    response = chain_with_history.invoke(  # noqa: T201
        {"requirement": """輸出結果保存在Result中，包括add和reason。其中add是一個布林值，表示是否需要將節日新增到節日清單中。根據問題判斷該節日是否存在於清單中，如果不存在，則為 true；否則為 false。reason : 描述為什麼需要或不需要新增節日，具體說明是否該節日已經存在於清單中，以及當前清單的內容。""", "question": question3, "ai_response": "{output}"},
        config={"configurable": {"session_id": "foo"}}
    )
    return response.content

# Function to encode a local image into data URL 
def local_image_to_data_url(image_path):
    # Guess the MIME type of the image based on the file extension
    mime_type, _ = guess_type(image_path)
    if mime_type is None:
        mime_type = 'application/octet-stream'  # Default MIME type if none is found

    # Read and encode the image file
    with open(image_path, "rb") as image_file:
        base64_encoded_data = base64.b64encode(image_file.read()).decode('utf-8')

    # Construct the data URL
    return f"data:{mime_type};base64,{base64_encoded_data}"

def generate_hw04(question):
    messages = [
        SystemMessage(
            content="請用JSON格式作答，不需要markdown標記，結果爲Result，積分爲Result下的score。"
        ),
        HumanMessage(
            content=[  
                { 
                    "type": "text", 
                    "text": question
                },
                { 
                    "type": "image_url",
                    "image_url": {
                        "url": local_image_to_data_url("baseball.png")
                    }
                }
            ]
        )
    ]
    response = llm.invoke(messages, max_tokens=2000)
    return response.content
    
def demo(question):
    message = HumanMessage(
            content=[
                {"type": "text", "text": question},
            ]
    )
    response = llm.invoke([message])
    return response

#import json
#print(json.loads(generate_hw01('2024年台灣10月紀念日有哪些?')))
#print(json.loads(generate_hw02('2024年台灣10月紀念日有哪些?')))
#print(json.loads(generate_hw03('2024年台灣10月紀念日有哪些?', '根據先前的節日清單，這個節日是否有在該月份清單？{"date": "10-31", "name": "蔣公誕辰紀念日"}')))
#print(json.loads(generate_hw04('請問中華台北的積分是多少?')))