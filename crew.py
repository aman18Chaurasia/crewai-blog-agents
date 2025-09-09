import sys
from crewai import Crew, Process
from agents import blog_researcher, blog_writer
from tasks import research_task, writing_task

# Build crew
crew = Crew(
    agents=[blog_researcher, blog_writer],
    tasks=[research_task, writing_task],
    process=Process.sequential,
    memory=True,
    cache=True,
    max_rpm=100,
    share_crew=True
)

if __name__ == "__main__":
    # Get topic from command line, or use default
    topic = sys.argv[1] if len(sys.argv) > 1 else "AI vs ML vs DL vs Data Science"
    
    print(f"\n🚀 Running CrewAI pipeline for topic: {topic}\n")
    result = crew.kickoff(inputs={"topic": topic})

    print("\n=== Final Blog Output ===\n")
    print(result)
