import json
import traceback

from model_configurations import get_model_configuration

from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain import hub
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_openai import ChatOpenAI

import base64
from mimetypes import guess_type

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
    #https://calendarific.com/api/v2/holidays?&api_key=NplCnEzX4afR3qpnwFF858d1S05XvzP7&country=TW&year=2024
    instructions = """You are an expert researcher."""
    base_prompt = hub.pull("langchain-ai/openai-functions-template")
    prompt = base_prompt.partial(instructions=instructions)
    from langchain_community.tools.semanticscholar.tool import SemanticScholarQueryRun
    tools = [SemanticScholarQueryRun()]
    agent = create_openai_functions_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
    )
    agent_executor.invoke(
        {
            "input": question
        }
    )
    return agent_executor.response

    
def generate_hw03(question2, question3):
    pass
    
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

#print(json.loads(generate_hw01('2024年台灣10月紀念日有哪些?')))
#print(json.loads(generate_hw02('2024年台灣10月紀念日有哪些?')))
#print(json.loads(generate_hw03('2024年台灣10月紀念日有哪些?', '根據先前的節日清單，這個節日是否有在該月份清單？{"date": "10-31", "name": "蔣公誕辰紀念日"}')))
#print(json.loads(generate_hw04('請問中華台北的積分是多少?')))