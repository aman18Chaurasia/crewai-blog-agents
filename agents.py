from crewai import Agent
from crewai_tools import YoutubeChannelSearchTool
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Setup LLM
llm = ChatOpenAI(
    model=os.getenv("OPENAI_MODEL_NAME", "gpt-4o"),
    temperature=0.7,
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

# Tool
youtube_channel_tool = YoutubeChannelSearchTool()

# Blog researcher agent
blog_researcher = Agent(
    role="Blog Researcher from YouTube videos",
    goal="Get relevant video content for the topic {topic} from the YouTube channel",
    verbose=True,
    memory=True,
    backstory=(
        "Expert in understanding videos in AI, Data Science, Machine Learning, and Gen AI, "
        "and providing structured suggestions for content creation."
    ),
    tools=[youtube_channel_tool],
    llm=llm,
    allow_delegation=True
)

# Blog writer agent
blog_writer = Agent(
    role="Writer",
    goal="Narrate compelling tech stories about the video {topic}",
    verbose=True,
    memory=True,
    backstory=(
        "With a flair for simplifying complex topics, you craft engaging narratives "
        "that captivate and educate, bringing new discoveries to light in an accessible manner."
    ),
    llm=llm,
    allow_delegation=False
)
