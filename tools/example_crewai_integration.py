"""
Example: Using DQN Tool in CrewAI Agents

This script demonstrates how to integrate the DQN stock research tool
into a CrewAI crew with multiple agents.
"""

from crewai import Agent, Task, Crew, Process
from tools.dqn_crewai_tool import dqn_stock_research


def create_stock_analysis_crew():
    """
    Create a CrewAI crew that uses the DQN tool for stock analysis.
    """
    
    # Agent 1: Stock Research Analyst (uses DQN tool)
    stock_research_agent = Agent(
        role='Stock Research Analyst',
        goal='Analyze stocks using advanced RL-based multi-agent system to provide investment recommendations',
        backstory="""You are an expert stock analyst with access to a sophisticated 
        Reinforcement Learning system that orchestrates multiple specialized agents 
        (research, technical analysis, sentiment analysis) to provide comprehensive 
        stock analysis and investment recommendations.
        
        Use the DQN tool to get detailed analysis, then synthesize the results
        into actionable insights for investors.""",
        tools=[dqn_stock_research],
        verbose=True,
        allow_delegation=False
    )
    
    # Agent 2: Investment Advisor (synthesizes DQN results)
    investment_advisor = Agent(
        role='Investment Advisor',
        goal='Provide strategic investment advice based on RL model analysis',
        backstory="""You are a senior investment advisor who receives analysis
        from the RL-powered stock research system. Your job is to interpret the
        technical analysis and provide clear, actionable investment advice
        considering risk tolerance and market conditions.""",
        verbose=True,
        allow_delegation=False
    )
    
    # Task 1: Analyze stock using DQN
    research_task = Task(
        description="""Analyze the stock NVDA using the DQN stock research tool. 
        Provide a comprehensive investment recommendation based on the RL model's analysis.
        Include details about the recommendation, confidence level, key insights, and technical indicators.""",
        agent=stock_research_agent,
        expected_output="A detailed stock analysis report with recommendation, confidence level, key insights, and supporting data from the DQN model."
    )
    
    # Task 2: Provide investment advice
    advice_task = Task(
        description="""Based on the DQN analysis provided, create a clear investment
        recommendation for a moderate-risk investor. Explain the reasoning, highlight
        key risks and opportunities, and suggest an appropriate investment strategy.""",
        agent=investment_advisor,
        expected_output="Clear investment advice with risk assessment and strategic recommendations."
    )
    
    # Create the crew
    crew = Crew(
        agents=[stock_research_agent, investment_advisor],
        tasks=[research_task, advice_task],
        process=Process.sequential,
        verbose=True
    )
    
    return crew


def run_example():
    """Run the example crew."""
    print("=" * 70)
    print("DQN Stock Research - CrewAI Integration Example")
    print("=" * 70)
    print()
    
    # Create crew
    crew = create_stock_analysis_crew()
    
    # Execute
    print("\n🚀 Starting crew execution...\n")
    result = crew.kickoff()
    
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(result)
    
    return result


if __name__ == '__main__':
    # Make sure dqn_model.pth exists before running
    import os
    if not os.path.exists('dqn_model.pth'):
        print("⚠️  Warning: dqn_model.pth not found!")
        print("   Please download it from Colab first.")
        print("   See tools/COLAB_MODEL_DOWNLOAD_GUIDE.md for instructions.")
        print()
    
    run_example()

