import json
import traceback

from model_configurations import get_model_configuration

from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate

gpt_chat_version = 'gpt-4o'
gpt_config = get_model_configuration(gpt_chat_version)

llm = AzureChatOpenAI(
        model=gpt_config['model_name'],
        deployment_name=gpt_config['deployment_name'],
        openai_api_key=gpt_config['api_key'],
        openai_api_version=gpt_config['api_version'],
        azure_endpoint=gpt_config['api_base'],
        temperature=gpt_config['temperature']
)

def generate_hw01(question):
    examples = [
        {"input": "請僅用純JSON格式呈現之後的結果，結果爲Result，日期爲date，紀念日名稱爲name", 
         "output": """{
                "Result": [
                    {
                        "date": "2024-02-28",
                        "name": "和平紀念日"
                    }
                ]
            }"""},
    ]
    example_prompt = ChatPromptTemplate.from_messages(
        [
            ("human", "{input}"),
            ("ai", "{output}"),
        ]
    )
    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=examples,
    )
    final_prompt = ChatPromptTemplate.from_messages(
        [
            few_shot_prompt,
            ("human", "{input}"),
        ]
    )
    chain = final_prompt | llm
    response = chain.invoke({"input": question})
    return response.content

def generate_hw02(question):
    #https://calendarific.com/api/v2/holidays?&api_key=NplCnEzX4afR3qpnwFF858d1S05XvzP7&country=TW&year=2024
    pass
    
def generate_hw03(question2, question3):
    pass
    
def generate_hw04(question):
    pass
    
def demo(question):
    message = HumanMessage(
            content=[
                {"type": "text", "text": question},
            ]
    )
    response = llm.invoke([message])
    
    return response
