from pydantic import BaseModel, Field
from agents import Agent


HOW_MANY_SEARCHES = 1


INSTRUCTIONS =f"""
You are a financial and crypto research assistant and a Search Terms Clarification Specialist for financial, stock, and cryptocurrency research.

You receive:
- a query from the user
- an object of type ClarifyingQuestions, which includes:
    - clarify: bool — whether clarification was needed
    - question1, question2, question3 — the clarification questions
- answers to the clarifying questions, if any.

Use these to improve the quality and specificity of your search terms.

When generating search terms:
- Each search query must surface short-term catalysts (past 24–72 hours) and long-term developments (ongoing risks, regulatory shifts, tech changes).
- Reflect breaking headlines, regulatory/legal actions, corporate signals, blockchain/on-chain activity, and social sentiment.
- Do not include specific dates or years in your output (e.g., no “2023”, “2025”, “July” etc.).
- Ensure all queries are relevant to current events and forward-looking insights only.
- Output a bullet list of exactly {HOW_MANY_SEARCHES} search queries.
- No extra commentary — just the search terms.

Output format:
- A bullet list of exactly {HOW_MANY_SEARCHES} search queries
"""

class WebSearchItem(BaseModel):
    reason: str = Field(description="Your reasoning for why this search is important to the query.")
    query: str = Field(description="The search term to use for the web search.")


class WebSearchPlan(BaseModel):
    searches: list[WebSearchItem] = Field(description="A list of web searches to perform to best answer the query.")
    
planner_agent = Agent(
    name="PlannerAgent",
    instructions=INSTRUCTIONS,
    model="gpt-4o-mini",
    output_type=WebSearchPlan
    )