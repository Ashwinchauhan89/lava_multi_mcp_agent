# lava_agent/app/agent.py

import os
import logging
from dotenv import load_dotenv

import google.cloud.logging
from google.adk import Agent
from google.adk.agents import SequentialAgent
from google.adk.tools.tool_context import ToolContext
from google.adk.tools.langchain_tool import LangchainTool

from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper


# -----------------------------
# SETUP
# -----------------------------
load_dotenv()

# Logging (GCP ready)
cloud_logging_client = google.cloud.logging.Client()
cloud_logging_client.setup_logging()

MODEL = os.getenv("MODEL", "gemini-1.5-flash")


# -----------------------------
# TOOL: STORE USER PROMPT
# -----------------------------
def save_prompt(tool_context: ToolContext, prompt: str) -> dict:
    tool_context.state["PROMPT"] = prompt
    logging.info(f"[STATE] Saved prompt: {prompt}")
    return {"status": "saved"}


# -----------------------------
# OPTIONAL TOOL: WIKIPEDIA
# -----------------------------
wikipedia_tool = LangchainTool(
    tool=WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
)


# -----------------------------
# AGENT 1: TASK ANALYZER
# -----------------------------
task_analyzer = Agent(
    name="task_analyzer",
    model=MODEL,
    description="Analyzes user query and extracts tasks, priorities, and deadlines.",
    instruction="""
    You are a task analysis agent.

    - Extract tasks from the user's PROMPT
    - Identify deadlines if mentioned
    - Identify priority if possible

    Return structured output like:
    - Task:
    - Deadline:
    - Priority:

    PROMPT:
    { PROMPT }
    """,
    tools=[wikipedia_tool],  # optional
    output_key="task_data"
)


# -----------------------------
# AGENT 2: RESPONSE GENERATOR
# -----------------------------
response_generator = Agent(
    name="response_generator",
    model=MODEL,
    description="Formats task data into a clean response.",
    instruction="""
    You are a productivity assistant.

    Take TASK_DATA and present it clearly.

    - Format tasks nicely
    - Add suggestions if needed
    - Keep it simple and readable

    TASK_DATA:
    { task_data }
    """
)


# -----------------------------
# WORKFLOW (SEQUENTIAL)
# -----------------------------
task_workflow = SequentialAgent(
    name="task_workflow",
    description="Processes user query into structured tasks and response",
    sub_agents=[
        task_analyzer,
        response_generator
    ]
)


# -----------------------------
# ROOT AGENT (ENTRY POINT)
# -----------------------------
root_agent = Agent(
    name="lava_root_agent",
    model=MODEL,
    description="Main entry point for Lava Agent system.",
    instruction="""
    You are Lava Agent, an AI productivity assistant.

    - Greet the user
    - Ask what task they want help with
    - Save user input using 'save_prompt'
    - Then pass control to task_workflow

    """,
    tools=[save_prompt],
    sub_agents=[task_workflow]
)