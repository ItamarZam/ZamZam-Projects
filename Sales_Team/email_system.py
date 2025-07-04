from dotenv import load_dotenv
from agents import Agent, Runner, trace, function_tool
from typing import Dict, List
import sendgrid
import os
from sendgrid.helpers.mail import Mail, Email, To, Content
import asyncio
import json
import requests
from bs4 import BeautifulSoup

load_dotenv(override=True)

RECIPIENTS = [
    {"email": "itamarzam1@gmail.com", "name": "Microsoft", "website": "https://www.microsoft.com"},
    {"email": "itamarzam1@gmail.com", "name": "Google", "website": "https://www.google.com"},
    {"email": "itamarzam1@gmail.com", "name": "OpenAI", "website": "https://www.openai.com"}
]

SENDER_NAME = "Itamar Zam"
SENDER_POSITION = "Sales Manager"
COMPANY_NAME = "ComplAI"

@function_tool
def mail_merge_send(subject: str, html_body_templates: str, recipients: str) -> str:
    """
    recipients: JSON string of a list of dicts, each with 'email' and 'name'
    """
    print(f"[DEBUG] HTML body to send:\n{html_body_templates}")  # Debug print
    recipients_list = json.loads(recipients)
    sg = sendgrid.SendGridAPIClient(api_key=os.environ.get("SENDGRID_API_KEY"))
    from_email = Email("itamarzam1@gmail.com")
    results = []
    for recipient in recipients_list:
        try:
            html_body = html_body_templates.format(name=recipient["name"])
            to_email = To(recipient["email"])
            content = Content("text/html", html_body)
            mail = Mail(from_email, to_email, subject, content).get()
            response = sg.client.mail.send.post(request_body=mail)
            results.append(f"Email to {recipient['email']}: status {response.status_code}")
        except Exception as e:
            results.append(f"Email to {recipient['email']}: error {str(e)}")
    return "\n".join(results)

@function_tool
def scrape_company_website(website_url: str) -> str:
    """
    Scrape the main text content from a company's website homepage and return it as a string.
    """
    try:
        response = requests.get(website_url, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        # Remove script and style elements
        for script in soup(["script", "style", "noscript"]):
            script.decompose()
        # Get visible text
        text = ' '.join(soup.stripped_strings)
        # Limit to first 2000 characters for brevity
        return text[:2000]
    except Exception as e:
        return f"ERROR: Failed to scrape website - {str(e)}"

# דמו של נתוני לינקדאין (מומצא)
demo_json = {
  "full_name": "Dana Levi",
  "headline": "CEO & Co-Founder at BrightWave Digital | Helping brands scale through performance-driven marketing",
  "location": "Tel Aviv, Israel",
  "industry": "Digital Marketing",
  "current_position": {
    "title": "Chief Executive Officer",
    "company": "BrightWave Digital",
    "start_date": "2019-06",
    "description": "Leading a team of 25 to deliver data-driven digital marketing strategies for eCommerce and B2B SaaS companies. Focused on growth, automation, and creative performance."
  },
  "past_positions": [
    {
      "title": "VP of Strategy",
      "company": "NextVision Media",
      "start_date": "2016-01",
      "end_date": "2019-05",
      "description": "Directed marketing strategy and client growth initiatives for over 30 tech clients."
    },
    {
      "title": "Digital Marketing Consultant",
      "company": "Freelance",
      "start_date": "2013-07",
      "end_date": "2015-12",
      "description": "Provided growth consulting and campaign execution for early-stage startups."
    }
  ],
  "education": [
    {
      "institution": "IDC Herzliya",
      "degree": "B.A. in Business & Marketing",
      "start_year": "2009",
      "end_year": "2012"
    }
  ],
  "skills": [
    "Digital Strategy",
    "Performance Marketing",
    "Team Leadership",
    "Marketing Automation",
    "Client Acquisition"
  ],
  "summary": "Experienced digital strategist and entrepreneur. Passionate about helping brands grow through data, creativity, and automation. Leading BrightWave to become one of Israel's top boutique digital agencies.",
  "interests": [
    "Growth marketing",
    "Tech startups",
    "Marketing automation tools",
    "E-commerce innovation",
    "SaaS platforms"
  ]
}

strategy_instructions = (
    "You are a professional B2B salesperson.\n"
    "You will receive a JSON object containing the company name, email, and website URL of a potential client.\n\n"
    "Your task is to:\n"
    "- If a website URL is provided, use the scrape_company_website tool to extract the main content from the company's homepage.\n"
    "- Analyze the website content to understand the company's business, industry, products, and possible pain points.\n"
    "- If scraping fails or no website is provided, use the company name and your general knowledge to infer likely pain points and needs.\n"
    "- Craft a personalized and compelling B2B sales outreach strategy tailored specifically to this company.\n\n"
    "The message should:\n"
    "- Address the company's likely pain points or business needs.\n"
    "- Demonstrate clear understanding of their business and context.\n"
    "- Position your solution as uniquely relevant to their challenges.\n"
    "- Be written in a natural, professional tone that is likely to spark curiosity and increase the chance of scheduling a call.\n\n"
    "Output format:\n"
    "Pain Points (bulleted list)\n"
    "Sales Strategy (a short, personalized outreach message of ~100–150 words)\n\n"
    "Focus on relevance, personalization, and intrigue. The goal is to get the company interested enough to book a call."
)

instructions1 = (
    "You are a professional B2B sales agent working for ComplAI, a company offering a SaaS solution for automating daily work tasks.\n\n"
    "You write serious, professional cold emails designed to intrigue companies and get them to schedule a call.\n\n"
    "For each company, use the Sales Strategy Tool (a separate tool assigned to you) to analyze the company website and data.\n"
    "Extract the company's pain points and a personalized sales strategy from the tool.\n"
    "Use the strategy to write a highly personalized cold email that:\n"
    "- Speaks directly to the company's likely needs or challenges\n"
    "- Clearly shows how ComplAI is relevant to their business\n"
    "- Uses a professional, credible tone\n"
    "- Ends with a soft CTA to schedule a call or learn more\n\n"
    "Use [Your Name] as a placeholder for the sender's name - this will be replaced with the actual sender name later.\n\n"
    "❗️Your output should only include the email body – no headers, no lists, no metadata.\n\n"
    "🎯 Focus on personalization, professionalism, and sparking curiosity — not on hard selling."
)

instructions2 = (
    "You are a professional B2B sales agent working for ComplAI, a company offering a SaaS solution for automating daily work tasks.\n\n"
    "You write witty, engaging cold emails designed to intrigue companies and get them to schedule a call.\n\n"
    "For each company, use the Sales Strategy Tool (a separate tool assigned to you) to analyze the company website and data.\n"
    "Extract the company's pain points and a personalized sales strategy from the tool.\n"
    "Use the strategy to write a highly personalized cold email that:\n"
    "- Speaks directly to the company's likely needs or challenges\n"
    "- Clearly shows how ComplAI is relevant to their business\n"
    "- Uses a professional, credible tone\n"
    "- Ends with a soft CTA to schedule a call or learn more\n\n"
    "Use [Your Name] as a placeholder for the sender's name - this will be replaced with the actual sender name later.\n\n"
    "❗️Your output should only include the email body – no headers, no lists, no metadata.\n\n"
    "🎯 Focus on personalization, professionalism, and sparking curiosity — not on hard selling."
)

instructions3 = (
    "You are a professional B2B sales agent working for ComplAI, a company offering a SaaS solution for automating daily work tasks.\n\n"
    "You write concise, to the point cold emails designed to intrigue companies and get them to schedule a call.\n\n"
    "For each company, use the Sales Strategy Tool (a separate tool assigned to you) to analyze the company website and data.\n"
    "Extract the company's pain points and a personalized sales strategy from the tool.\n"
    "Use the strategy to write a highly personalized cold email that:\n"
    "- Speaks directly to the company's likely needs or challenges\n"
    "- Clearly shows how ComplAI is relevant to their business\n"
    "- Uses a professional, credible tone\n"
    "- Ends with a soft CTA to schedule a call or learn more\n\n"
    "Use [Your Name] as a placeholder for the sender's name - this will be replaced with the actual sender name later.\n\n"
    "❗️Your output should only include the email body – no headers, no lists, no metadata.\n\n"
    "🎯 Focus on personalization, professionalism, and sparking curiosity — not on hard selling."
)

strategy_agent = Agent(
    name="Sales Strategy Agent",
    instructions=strategy_instructions,
    tools=[scrape_company_website],
    model="gpt-4o-mini"
)
strategy_tool = strategy_agent.as_tool(tool_name="strategy_agent", tool_description="create a sales strategy for a prospect using scraped LinkedIn data (demo only)")

# כל סוכן מקבל את כלי האסטרטגיה
sales_agent1 = Agent(
    name="Professional Sales Agent",
    instructions=instructions1,
    tools=[strategy_tool],
    model="gpt-4o-mini"
)
sales_agent2 = Agent(
    name="Engaging Sales Agent",
    instructions=instructions2,
    tools=[strategy_tool],
    model="gpt-4o-mini"
)
sales_agent3 = Agent(
    name="Busy Sales Agent",
    instructions=instructions3,
    tools=[strategy_tool],
    model="gpt-4o-mini"
)

description = "Write a cold sales email"
tool1 = sales_agent1.as_tool(tool_name="sales_agent1", tool_description=description)
tool2 = sales_agent2.as_tool(tool_name="sales_agent2", tool_description=description)
tool3 = sales_agent3.as_tool(tool_name="sales_agent3", tool_description=description)

tools = [tool1, tool2, tool3]

subject_instructions = "You can write a subject for a cold sales email. \nYou are given a message and you need to write a subject for an email that is likely to get a response. \nYou only write one subject for only one email."

html_instructions = (
    "You are a professional web designer specializing in email marketing.\n"
    "You receive a plain text sales email body and must convert it into a clean, professional HTML email.\n"
    "Use simple, clean HTML with basic styling suitable for most email clients.\n"
    "Replace [Your Name] with the actual sender name provided in the message.\n"
    "Always return a valid HTML email body, even if it is very simple.\n"
    "Never return an error message—always return a valid HTML email body.\n"
    "If you are unsure, wrap the text in <html><body>...</body></html> tags and return it.\n"
    "Example:\n"
    "<html><body><p>Hello NAME,</p><p>This is a sample email.</p><p>Best regards,<br>Itamar Zam</p></body></html>"
)
subject_writer= Agent(name="Email subject writer", instructions=subject_instructions,model="gpt-4o-mini")
subject_tool=subject_writer.as_tool(tool_name="subject_writer", tool_description="write a subject for a cold sales email")

html_converter=Agent(name="HTML email body converter", instructions= html_instructions,model="gpt-4o-mini")
html_tool=html_converter.as_tool(tool_name="subject_writer", tool_description="convert a text email body to an HTML email body")

emailer_tools = [subject_tool, html_tool, mail_merge_send]

emailer_instructions = (
    "You receive the plain text body of a sales email.\n"
    "First, use the html_tool to transform this email body to a HTML formatted email.\n"
    "Then, use the subject tool to generate the title for the email.\n"
    "Finally, use the mail_merge_send tool to send the email to all recipients.\n"
    "Do not repeat any step more than once. Once the email is sent, stop.\n"
    "If the html_tool returns something that is not valid HTML, use it as-is and send anyway.\n"
    "Never ask for clarification. Never repeat steps."
)
emailer_agent = Agent(
    name="Email Manager",
    instructions=emailer_instructions,
    tools=emailer_tools,
    model="gpt-4o-mini",
    handoff_description="Convert an email to HTML and send it"
)

handoffs = [emailer_agent]

sales_manager_instructions = (
    "You are a sales manager for ComplAI, a company providing SaaS for automating for daily work tasks. "
    "Your job is to generate the best possible cold sales email for a list of prospects. "
    "First, use all three sales agent tools to generate different versions of the email body. "
    "Carefully compare the results and select the single best email content, based on which is most likely to get a positive response. "
    "Do not write or edit the email yourself—always use the tools. "
    "Once you have chosen the best email body, hand off the content and the full recipient list to the Email Manager agent, "
    "who will handle subject generation, HTML formatting, and sending."
)
sales_manager = Agent(
    name="Sales Manager",
    instructions=sales_manager_instructions,
    tools=tools,
    handoffs=handoffs,
    model="gpt-4o-mini"
)

message = "Send a cold sales email to all prospects"

async def send_b2b_sales_emails(message: str, recipients: list, sender_name: str):
    print("=== Starting automated email sending process ===")
    with trace("Automated SDR"):
        full_message = (
            f"{message}\n"
            "RECIPIENTS_JSON:\n"
            f"{json.dumps(recipients)}\n"
            "SENDER_NAME:\n"
            f"{sender_name}\n"
        )
        print("\n[DEBUG] Message sent to sales_manager:")
        print(full_message)
        result = await Runner.run(sales_manager, full_message)
        print("\n[DEBUG] Final result from the system:")
        print(result)
    print("=== Process finished ===")
    return result

if __name__ == "__main__":
    # Example usage
    recipients = [
        {"email": "itamarzam1@gmail.com", "name": "Microsoft", "website": "https://www.microsoft.com"},
        {"email": "itamarzam1@gmail.com", "name": "Google", "website": "https://www.google.com"},
        {"email": "itamarzam1@gmail.com", "name": "OpenAI", "website": "https://www.openai.com"}
    ]
    message = "Send a cold sales email to the following recipients."
    sender_name = "Itamar Zam"
    import asyncio
    asyncio.run(send_b2b_sales_emails(message, recipients, sender_name))