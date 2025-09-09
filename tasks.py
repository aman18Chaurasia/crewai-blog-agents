from crewai import Task
from agents import blog_researcher, blog_writer
from agents import youtube_channel_tool

# Research task
research_task = Task(
    description=(
        "Identify videos related to {topic} from the YouTube channel. "
        "Get detailed information about the most relevant video content."
    ),
    expected_output="A comprehensive 3-paragraph long report based on the {topic} video content.",
    tools=[youtube_channel_tool],
    agent=blog_researcher
)

# Writing task
writing_task = Task(
    description=(
        "Using the research report from the researcher agent, "
        "write a well-structured blog article on {topic}. "
        "The article should be engaging, clear, and optimized for readability."
    ),
    expected_output="A blog post of at least 600 words with an introduction, body, and conclusion.",
    agent=blog_writer,
    async_execution=False,
    output_file="new-blog-post.md"
)
