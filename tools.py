from crewai import Agent, Task, Crew
from crewai_tools import YoutubeChannelSearchTool

# Initialize the tool for general YouTube channel searches
youtube_channel_tool = YoutubeChannelSearchTool()

# Define an agent that uses the tool
channel_researcher = Agent(
    role="Channel Researcher",
    goal="Extract relevant information from YouTube channels",
    backstory="An expert researcher who specializes in analyzing YouTube channel content.",
    tools=[youtube_channel_tool],
    verbose=True,
)

# Example task to search for information in a specific channel
research_task = Task(
    description="Search for information about machine learning tutorials in the YouTube channel {youtube_channel_handle}",
    expected_output="A summary of the key machine learning tutorials available on the channel.",
    agent=channel_researcher,
)

# Create and run the crew
crew = Crew(agents=[channel_researcher], tasks=[research_task])
result = crew.kickoff(inputs={"youtube_channel_handle": "@krishnaik06"})