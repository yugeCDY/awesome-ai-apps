"""
Lesson 1: Basic Agent Creation

This script demonstrates the fundamentals of creating a simple AI agent using the AWS Strands SDK.

We will build a weather assistant that can:
1.  Understand a natural language query about weather.
2.  Use the `http_request` tool to fetch data from the National Weather Service API.
3.  Synthesize the data into a human-readable response.
"""

import os
from dotenv import load_dotenv
from strands import Agent
from strands.models.litellm import LiteLLMModel
from strands_tools import http_request

# Load environment variables from a .env file
load_dotenv()

# Define a detailed system prompt to guide the agent's behavior
WEATHER_SYSTEM_PROMPT = """你是一位友好且专业的天气助手，具备 HTTP 请求能力。

你的主要职责是通过美国国家气象局（National Weather Service）API，为美国地区提供准确的天气预报。

请按以下步骤完成用户请求：
1. 如果还没有网格坐标，先调用 points API 获取坐标信息。
   - 经纬度查询：https://api.weather.gov/points/{latitude},{longitude}
   - 美国邮编查询：https://api.weather.gov/points/{zipcode}
2. points API 会返回 `forecast` URL。使用该 URL 再发起一次 HTTP 请求，获取实际天气预报数据。
3. 处理预报数据，并用清晰、易懂的方式呈现给用户。

在输出结果时请注意：
- 突出关键信息，例如温度、降水和天气预警。
- 用简单语言解释技术术语。
- 如果出现错误，请先致歉，并说明无法成功获取天气信息。
"""


def create_weather_agent() -> Agent:
    """
    Creates and configures a weather-focused agent.

    Returns:
        An Agent instance configured with a model, system prompt, and tools.
    """
    # Configure the language model (LLM) that will power the agent.
    # We use LiteLLMModel to connect to DeepSeek via the OpenAI-compatible API.
    model = LiteLLMModel(
        client_args={
            "api_key": os.getenv("DEEPSEEK_API_KEY"),
            "base_url": "https://api.deepseek.com",
        },
        model_id="openai/deepseek-reasoner",
        params={
            "max_tokens": 1500,
            "temperature": 0.7,
        },
    )

    # Create the agent instance.
    # The agent is the core component that orchestrates the LLM, tools, and system prompt.
    weather_agent = Agent(
        system_prompt=WEATHER_SYSTEM_PROMPT,
        tools=[http_request],  # Grant the agent the ability to make HTTP requests.
        model=model,
    )
    return weather_agent


def main():
    """
    Main function to run the weather agent.
    """
    # Create the weather agent
    weather_agent = create_weather_agent()

    # Define a user query
    user_query = "请比较一下纽约和芝加哥这个周末的气温。"

    # Invoke the agent with the query and get the response
    print(f"User Query: {user_query}\n")
    response = weather_agent(user_query)

    # Print the agent's final response
    print("Weather Agent Response:")
    print(response)


if __name__ == "__main__":
    main()
