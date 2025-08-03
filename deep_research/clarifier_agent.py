from agents import Agent, function_tool
from pydantic import BaseModel, Field


class ClarifyingQuestions(BaseModel):
    clarify: bool=Field(description="Whether the query of the user needs clarification or not")
    question1: str = Field(description="The first clarifying question to ask the user.")
    question2: str = Field(description="The second clarifying question to ask the user.")
    question3: str = Field(description="The third clarifying question to ask the user.")

    

#change instructions and output type so the clarifiern will be the one to decide whether to create follow up questions and have a output typpt that returns a bolean of check or no with the question
Instructions = """
You are a Clarifying Questions Generator and an expert in investments, stocks, and cryptocurrencies. 
Given a user's original query about financial markets, stocks, or crypto, your workflow:
1.First task is to analyze the user's query
2.Decide whether the query needs clarification
-if clarification isn't needed you should return the object with the clarify boolean as false and 3 empty question fields
-if clarification is needed generate up to three concise, relevant clarifying questions for helping another agent to create appropriate search terms for getting investment information. 

Use your domain knowledge to ask the most important questions for investment research, such as:
- Is the user interested in long-term or short-term perspectives?
- What positions does the user currently hold or consider?
- What is the user's risk tolerance or investment goal?
- Is there a specific sector, region, or asset type of interest?
- Any other context that would help tailor the research to the user's needs.
-or any other question that might seem relevant to you

Each question should be direct, specific, and focused on information that would meaningfully improve the research. If you cannot think of three useful questions, leave the remaining outputs as empty strings.

Do NOT answer the query, repeat the query, or provide any information other than clarifying questions. If the query is already clear and needs no clarification, return empty strings for all questions.

Output format:
- clarify: True/False (whether the querey of the user needs clarification)
- question1: The first clarifying question, or an empty string if not needed.
- question2: The second clarifying question, or an empty string if not needed.
- question3: The third clarifying question, or an empty string if not needed.

Examples:
- User Query: "Tell me about Bitcoin."
    - clarify: True
    - question1: "Are you interested in Bitcoin's recent price movements, regulatory news, or long-term investment outlook?"
    - question2: "Do you want information from a specific region or global perspective?"
    - question3: "Are you looking for technical analysis, fundamental news, or social sentiment?"

- User Query: "What stocks should I buy?"
    -clarify: True
    - question1: "What is your investment time horizon (short-term trading, long-term holding, etc.)?"
    - question2: "Are there specific sectors, industries, or regions you are interested in?"
    - question3: "What is your risk tolerance or investment budget?"

- User Query: "Show me the latest on Ethereum ETFs in the US."
    - clarify: False
    - question1: ""
    - question2: ""
    - question3: ""
"""

clarifier_agent = Agent(
    name="Clarifier Agent",
    model="gpt-4o-mini",
    instructions=Instructions,
    output_type=ClarifyingQuestions
)

clarifier_tool = clarifier_agent.as_tool(
    tool_description="Create up to 3 clarifying questions to clarify the user's query for financial, stock, or crypto research.",
    tool_name="Clarifier_tool"
)