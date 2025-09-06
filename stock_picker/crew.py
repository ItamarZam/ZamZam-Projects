from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List
from pydantic import BaseModel,Field
from crewai_tools import SerperDevTool
from .tools.push_notification_tool import PushNotificationTool#this is how you import a custom tool


class TrendingCompany(BaseModel):
    """A company that is going through a big event"""
    company_name: str = Field(description="Company name")
    ticker: str = Field(description="Stock ticker symbol")
    event: str = Field(description="The event that the company is going through")

class TrendingCompanyList(BaseModel):
    """List of multiple companies that are going through a big event"""
    companies: List[TrendingCompany] =Field(description="List of companies that are going through a big event")


class TrendingCompanyResearch(BaseModel):
    """Detailed research on a company"""
    company_name: str =Field(description="Company name")
    market_position: str = Field(description="Current market position and competitive analysis")
    future_outlook: str =Field(description="Future outlook and growth prospects")
    investment_potential: str =Field(description="Investment potential and suitability for investment")
    #more variables

class TrendingCompanyResearchList(BaseModel):
    """A list of researches on all of the companies that are going through a big event"""
    research_list:List[TrendingCompanyResearch]=Field(description="Comprehensive research on all trending companies")
@CrewBase
class StockPicker2():
    """StockPicker2 crew"""

    agents: List[BaseAgent]
    tasks: List[Task]

    @agent
    def trending_company_finder(self) -> Agent:
        return Agent(
            config=self.agents_config['trending_company_finder'], 
            tools=[SerperDevTool()]
        )
    
    @agent
    def financial_researcher(self) -> Agent:
        return Agent(
            config=self.agents_config['financial_researcher'], 
            tools=[SerperDevTool()]
        )
    
    @agent
    def stock_picker(self) -> Agent:
        return Agent(
            config=self.agents_config['stock_picker']
        )
    
    @agent
    def final_reporter(self) -> Agent:
        return Agent(
            config=self.agents_config['final_reporter'],
            tools=[SerperDevTool()]
        )
    
    @task
    def find_trending_companies(self) -> Task:
        return Task(
            config=self.tasks_config['find_trending_companies'],
            output_pydantic=TrendingCompanyList        
        )
    
    @task
    def research_trending_companies(self) ->Task:
        return Task(
            config=self.tasks_config['research_trending_companies'],
            output_pydantic= TrendingCompanyResearchList
        )
    
    @task
    def pick_best_company(self) -> Task:
        return Task(
            config=self.tasks_config['pick_best_company']
        )
    
    @task
    def create_final_report(self) -> Task:
        return Task(
            config=self.tasks_config['create_final_report']
        )

    @crew
    def crew(self) -> Crew:
        """creates the StockPicker crew"""
        
        #manager defining
        manager= Agent(
            config=self.agents_config['manager'],
            allow_delegation=True
        )

        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.hierarchical,#hierarchical because of the manager
            verbose=True,
            manager_agent=manager,#this is the manager assigning
            max_iter=3#this is the max number of iterations for the crew to run before abborting
            #manager_llm="openai/gpt-4o-mini"  this also does the whole manager thing without the need of creating the whole manager before
        )
    