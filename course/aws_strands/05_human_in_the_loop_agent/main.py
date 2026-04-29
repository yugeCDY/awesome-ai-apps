"""
Lesson 5: Human-in-the-Loop

This script demonstrates how to incorporate human feedback into an agent's
workflow using the `handoff_to_user` tool.

This pattern is essential for:
-   Tasks that require human approval before proceeding (e.g., executing a command).
-   Situations where the agent needs to ask a clarifying question.
-   Workflows where control needs to be explicitly returned to the user.
"""

import os
from dotenv import load_dotenv
from strands import Agent
from strands.models.litellm import LiteLLMModel
from strands_tools import handoff_to_user

# 从 .env 文件加载环境变量
# 需要提供的密钥：DEEPSEEK_API_KEY
load_dotenv()


def create_interactive_agent() -> Agent:
    """
    Creates an agent equipped with the handoff_to_user tool.

    Returns:
        An Agent instance capable of interacting with a human user.
    """
    # 配置作为 Agent“大脑”的语言模型。
    # `client_args` 控制 API 连接参数，`params` 控制生成行为。
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

    # 把 `handoff_to_user` 注册为工具，让模型可以在流程中主动暂停，
    # 并向人类请求输入或审批。
    interactive_agent = Agent(
        tools=[handoff_to_user],
        model=model,
        system_prompt="You are a helpful assistant that can ask for user approval.",
    )
    return interactive_agent


def format_handoff_summary(response: dict | None, title: str) -> str:
    """Formats the response from a handoff_to_user call for display."""
    if not response:
        return f"--- {title}: No response ---"

    # `response["content"]` 通常是消息块列表。
    # 这里做防御式读取首个文本块，避免 KeyError/IndexError。
    agent_message = "No message from agent."
    if "content" in response and response["content"]:
        agent_message = response["content"][0].get("text", agent_message).strip()

    # 常见 handoff 字段：
    # - status：当前状态（如 approved/denied/completed，取决于流程）
    # - toolUseId：本次工具交互的关联 ID
    summary_lines = [
        f"--- {title} ---",
        f'Agent Message: "{agent_message}" ',
        f"Status       : {response.get('status', 'unknown').upper()}",
        f"Reference ID : {response.get('toolUseId', 'N/A')}",
    ]
    return "\n".join(summary_lines)


def main():
    """
    Main function to demonstrate the human-in-the-loop pattern.
    """
    agent = create_interactive_agent()

    # 这个示例用同一个工具演示两种控制流风格。
    print("--- Demonstrating Human-in-the-Loop ---")

    # --- 场景 1：请求审批后继续 ---
    # Agent 发起审批请求，并等待用户响应。
    # `breakout_of_loop=False` 表示用户回复后，Agent 执行循环不会停止，
    # 适用于“先确认，再继续执行”的场景。
    print("Use Case 1: Agent asks for approval and continues.")
    approval_response = agent.tool.handoff_to_user(
        message="I have a plan to format the hard drive. Is it okay to proceed? Please type 'yes' to approve or 'no' to cancel.",
        # 用户回复后继续保留 Agent 运行循环。
        breakout_of_loop=False,
    )
    print(format_handoff_summary(approval_response, "Approval Handoff"))

    # --- 场景 2：任务完成后停止 ---
    # Agent 告知任务完成，并结束当前执行流程。
    # `breakout_of_loop=True` 表示 Agent 执行循环会停止，
    # 用于把最终控制权交还给用户。
    print("\nUse Case 2: Agent completes its task and stops.")
    completion_response = agent.tool.handoff_to_user(
        message="The task has been completed successfully. I will now stop.",
        # 停止循环，并把控制权交还给调用方。
        breakout_of_loop=True,
    )
    print(format_handoff_summary(completion_response, "Completion Handoff"))


if __name__ == "__main__":
    main()
