
#Link to the HuggingFace space: https://huggingface.co/spaces/Shazamzam/career_foundation

from dotenv import load_dotenv
from openai import OpenAI
import json
import os
import requests
from pypdf import PdfReader
import gradio as gr
from pydantic import BaseModel



load_dotenv(override=True)

def push(text):
    requests.post(
        "https://api.pushover.net/1/messages.json",
        data={
            "token": os.getenv("PUSHOVER_API_KEY"),
            "user": os.getenv("PUSHOVER_USER_KEY"),
            "message": text,
        }
    )


def record_user_details(email, name="Name not provided", notes="not provided"):
    push(f"Recording {name} with email {email} and notes {notes}")
    return {"recorded": "ok"}

def record_unknown_question(question):
    push(f"Recording {question}")
    return {"recorded": "ok"}

record_user_details_json = {
    "name": "record_user_details",
    "description": "Use this tool to record that a user is interested in being in touch and provided an email address",
    "parameters": {
        "type": "object",
        "properties": {
            "email": {
                "type": "string",
                "description": "The email address of this user"
            },
            "name": {
                "type": "string",
                "description": "The user's name, if they provided it"
            }
            ,
            "notes": {
                "type": "string",
                "description": "Any additional information about the conversation that's worth recording to give context"
            }
        },
        "required": ["email"],
        "additionalProperties": False
    }
}

record_unknown_question_json = {
    "name": "record_unknown_question",
    "description": "Always use this tool to record any question that couldn't be answered as you didn't know the answer",
    "parameters": {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "The question that couldn't be answered"
            },
        },
        "required": ["question"],
        "additionalProperties": False
    }
}

tools = [{"type": "function", "function": record_user_details_json},
        {"type": "function", "function": record_unknown_question_json}]

class Evaluation(BaseModel):
    is_acceptable:bool
    feedback:str
class Me:
    
    def __init__(self):
        self.openai = OpenAI()
        #self.gemini =OpenAI(api_key= os.getenv("GEMINI_API_KEY"), base_url="https://generativelanguage.googleapis.com/v1beta/openai/")
        self.name = "Itamar Sagiv Zam"
        self.resume = """Itamar Zam
12th Grade Student
Education:
MAKIF Y”A HIGHSCHOOL
-5 credit units in Math
-5 credit units in English
-5 credit units in Physics (Schwartz Reisman Institute), including
additional seminars, lectures, and courses at the Weizmann
Institute.
-10 credit units in Computer Science: Data Structures,
Algorithms, DL & ML, and Website Building.

Specializations:
-AI, ML & DL (3 years experience)
including practical projects carried out in school & externally,
LLM’s and agents experience.
-Particle & Modern Physics
course at the Weizmann Institute, from which I was chosen
to go on an expedition to the DESY Lab.


Acheivements:
-Won 3 place in the district in the AI Olympics contest rd
-Excelling student

Skills & Knowledge:

TECHNICAL SKILLS:
-High-level programming in languages such as:
Python, C#, Java, HTML, JavaScript.
-Working with advanced libraries:
Keras, TensorFlow, NumPy, Requests, Socket, OpenAI.
-Networks and Cybersecurity:
Experience with Wireshark, network analysis,vulnerability analysis, and security breaches.
-Meachine Learning & Deep Learning:
3 years of experience including projects & official edcation, experience with LLM’s and agents creation. API's and model providers used in agents creation: Ollama, Groq, Gemini, OpenAI.. using Git, Cursor IDE and UV package manager.
-Graphic and interactive design:
Experience with design platforms such as Wix and Premiere Pro.
-Internet marketing and sales:
Experience in online marketing and sales




Personal Story:

Ever since I was a child, the glow of a
monitor and the thrill of solving a
tough coding challenge have felt like
home—each bug I fix and every
system I build fuels my curiosity and
determination. At just 17, I’ve already
turned my passion for technology
into real-world projects and
problem-solving experiences that
prove I’m ready to make a
meaningful impact.


PROJECTS:
-Custom Website Development:
Built websites using HTML, CSS, and JavaScript.
problems.
-Deep Learning Machine Learning Models:
Developed AI models to solve various classification
problems using models like LSTM, ViT and CNN.
-API Integration and Backend Development: Including socket creation
-Game Development in Python: Extensive game creation
-Advanced WebDesign : Designed and built e-commerce websites



Pesonality & Values:

PERSONAL TRAITS:
-Quick learning ability and strong problem-solving skills
-Creativity and an innovative approach to challenges
-High work ethic and a commitment to excellence
-Leadership skills and effective teamwork
-Natural rapport with people

PERSONAL GOAL:
My goal is to blend into the High-Tech
industry especially in Data-Scienece related
jobs and to extend my skills and knowledge.

Custom Website Development:
Built websites using HTML, CSS, and JavaScript.
API Integration and Backend Development:
Game Development in Python:
Advanced WebDesign :
Designed and built e-commerce websites


Problems I have Overcomed:
-Let's take for instance the AI olympics me and my colleague attended. In the contest we had to create an investors slides show that will showcase our idea and our prototype. We had made way over 15 slides but 30 minutes before the contest's deadline has come we were told that the judges won't have enough time to look through out the entire slide so we had to make it 5 pages long which was a hard thing to do in 30 minutes(deciding how much of the content stays and how much don't, rearanging the slides and doing basically everything all over again). But despite the challenge we devided the work into missions and finished it although the stress and complexity of it and won in 3rd place!
-In my DL internship I took on myself a hard task of predicting the next bitcoing trend based on bitcoing graphs I have created through Binance API. During the project I had a lot of problems I did not know how to tackle, but I never gave up and I kept trying one thing after the other until the project finally worked. For that project I have used LSTM + CNN + pretrained model which I believe shows skills and knowledge at Deep Learning.


INTERESTS:
-AI
-Computer Science & Coding
-Philosophy
-Chess
-Reading- Psychology, Personal Progress & AI
-Mixed Martial Arts(MMA)

Further notes:
Despite my extensive knowledge
in computer science, I believe my
strongest qualities are my
character and mindset. I am
persistent and determined,
always eager to learn and grow.
People often turn to me for help
with complex problems, and they
would describe me as reliable and
always willing to assist others.

Links:
-GitHub:
https://github.com/Itam


Personal Info:

phone number: +972-4-544-1991
email address:itamarzam1@gmail.com
address:Shlonski 4, Rishon Lezion
age:17 years old(Army recruit is at the end of march 2026)


Languages:

Hebrew: Mother language
English: Fluent
Arabic: Basic

"""
        base_dir = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(base_dir, "me", "summary.txt"), "r", encoding="utf-8") as f:
            self.summary = f.read()

    def handle_tool_call(self, tool_calls):
        results = []
        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)
            print(f"Tool called: {tool_name}", flush=True)
            tool = globals().get(tool_name)
            result = tool(**arguments) if tool else {}
            results.append({"role": "tool","content": json.dumps(result),"tool_call_id": tool_call.id})
        return results
    
    def system_prompt(self):
        system_prompt = f"You are acting as {self.name}. You are answering questions on {self.name}'s website, \
particularly questions related to {self.name}'s career, background, skills and experience. \
Your responsibility is to represent {self.name} for interactions on the website as faithfully as possible. \
You are given a summary of {self.name}'s background and resume which you can use to answer questions. \
Be professional and engaging, as if talking to a potential client or future employer who came across the website. \
If you don't know the answer to any question, use your record_unknown_question tool to record the question that you couldn't answer, even if it's about something trivial or unrelated to career. \
If the user is engaging in discussion, try to steer them towards getting in touch via email; ask for their email and record it using your record_user_details tool. "

        system_prompt += f"\n\n## Summary:\n{self.summary}\n\n## resume :\n{self.resume}\n\n"
        system_prompt += f"With this context, please chat with the user, always staying in character as {self.name}."
        return system_prompt
    

    def evaluator_system_prompt(self):
        evaluator_system_prompt = f"You are an evaluator that decides whether a response to a question is acceptable. \
        You are provided with a conversation between a User and an Agent. Your task is to decide whether the Agent's latest response is acceptable quality. \
        The Agent is playing the role of {self.name} and is representing {self.name} on their website. \
        The Agent has been instructed to be professional and engaging, as if talking to a potential client or future employer who came across the website. \
        The Agent has been provided with context on {self.name} in the form of their summary and resume details. Here's the information:"

        evaluator_system_prompt += f"\n\n## Summary:\n{self.summary}\n\n## LinkedIn Profile:\n{self.resume}\n\n"
        evaluator_system_prompt += f"With this context, please evaluate the latest response, replying with whether the response is acceptable and your feedback."

        return evaluator_system_prompt

    def evaluator_user_prompt(self,reply, message, history):
        user_prompt = f"Here's the conversation between the User and the Agent: \n\n{history}\n\n"
        user_prompt += f"Here's the latest message from the User: \n\n{message}\n\n"
        user_prompt += f"Here's the latest response from the Agent: \n\n{reply}\n\n"
        user_prompt += f"Please evaluate the response, replying with whether it is acceptable and your feedback."
        return user_prompt
    

    # the evaluation function
    #parse is used for roled outputs
    def evaluate(self,reply,message, history) -> Evaluation:
        messages= [{"role":"system","content":self.evaluator_system_prompt()}]+[{"role":"user","content":self.evaluator_user_prompt(reply, message, history)}]
        response=self.openai.beta.chat.completions.parse(
            model="o3-mini",
            messages=messages,
            response_format= Evaluation
        )
        return response.choices[0].message.parsed

    def rerun(self,reply, message, history, feedback):
        updated_system_prompt = self.system_prompt() +f"\n\n ##Previous answer rejected\n You just tried to reply, but the quality control rejected your reply\n"
        updated_system_prompt += f"## Your attempted answer:\n{reply}\n\n"
        updated_system_prompt += f"## Reason for rejection:\n{feedback}\n\n"
        messages = [{"role":"system","content":updated_system_prompt}]+history+[{"role":"user","content":message}]
        response=self.openai.chat.completions.create(model="gpt-4o-mini",messages=messages)
        return response.choices[0].message.content
    

    def chat(self, message, history):
        messages = [{"role": "system", "content": self.system_prompt()}] + history + [{"role": "user", "content": message}]
        done = False
        while not done:
            response = self.openai.chat.completions.create(model="gpt-4o-mini", messages=messages, tools=tools)
            if response.choices[0].finish_reason=="tool_calls":
                message = response.choices[0].message
                tool_calls = message.tool_calls
                results = self.handle_tool_call(tool_calls)
                messages.append(message)
                messages.extend(results)
            else:
                done = True
        reply=response.choices[0].message.content
        evaluation= self.evaluate(reply,message, history)

        if evaluation.is_acceptable:
            print("Passed evaluation - returning reply")
        else:
            print("Evaluation failed")
            print(evaluation.feedback)
            reply = self.rerun(reply, message, history, evaluation.feedback)

        return reply
    

if __name__ == "__main__":
    me = Me()
    gr.ChatInterface(me.chat, type="messages").launch()
    





"""

from dotenv import load_dotenv
from openai import OpenAI
import json
import os
import requests
from pypdf import PdfReader
import gradio as gr
from pydantic import BaseModel


load_dotenv(override=True)

def push(text):
    requests.post(
        "https://api.pushover.net/1/messages.json",
        data={
            "token": os.getenv("PUSHOVER_API_KEY"),
            "user": os.getenv("PUSHOVER_USER_KEY"),
            "message": text,
        }
    )


def record_user_details(email, name="Name not provided", notes="not provided"):
    push(f"Recording {name} with email {email} and notes {notes}")
    return {"recorded": "ok"}

def record_unanswered_question(question):
    push(f"Recording {question} asked that I couldn't answer")
    return {"recorded": "ok"}

record_user_details_json = {
    "name": "record_user_details",
    "description": "Use this tool to record that a user is interested in being in touch and provided an email address",
    "parameters": {
        "type": "object",
        "properties": {
            "email": {
                "type": "string",
                "description": "The email address of this user"
            },
            "name": {
                "type": "string",
                "description": "The user's name, if they provided it"
            }
            ,
            "notes": {
                "type": "string",
                "description": "Any additional information about the conversation that's worth recording to give context"
            }
        },
        "required": ["email"],
        "additionalProperties": False
    }
}

record_unanswered_question_json = {
    "name": "record_unanswered_question",
    "description": "Always use this tool to record any question that couldn't be answered as you didn't know the answer",
    "parameters": {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "The question that couldn't be answered"
            },
        },
        "required": ["question"],
        "additionalProperties": False
    }
}

tools = [{"type": "function", "function": record_user_details_json},
        {"type": "function", "function": record_unanswered_question_json}]

class Evaluation(BaseModel):
    is_acceptable:bool
    feedback:str
class Me:

    def __init__(self):
        self.openai = OpenAI()
        self.gemini =OpenAI(api_key= os.getenv("GEMINI_API_KEY"), base_url="https://generativelanguage.googleapis.com/v1beta/openai/")
        self.name = "Itamar Sagiv Zam"
        self.resume = """"""Itamar Zam
12th Grade Student
Education:
MAKIF Y”A HIGHSCHOOL
-5 credit units in Math
-5 credit units in English
-5 credit units in Physics (Schwartz Reisman Institute), including
additional seminars, lectures, and courses at the Weizmann
Institute.
-10 credit units in Computer Science: Data Structures,
Algorithms, DL & ML, and Website Building.

Specializations:
-AI, ML & DL (3 years experience)
including practical projects carried out in school & externally,
LLM’s and agents experience.
-Particle & Modern Physics
course at the Weizmann Institute, from which I was chosen
to go on an expedition to the DESY Lab.


Acheivements:
-Won 3 place in the district in the AI Olympics contest rd
-Excelling student

Skills & Knowledge:

TECHNICAL SKILLS:
-High-level programming in languages such as:
Python, C#, Java, HTML, JavaScript.
-Working with advanced libraries:
Keras, TensorFlow, NumPy, Requests, Socket, OpenAI.
-Networks and Cybersecurity:
Experience with Wireshark, network analysis,vulnerability analysis, and security breaches.
-Meachine Learning & Deep Learning:
3 years of experience including projects & official edcation, experience with LLM’s and agents creation. API's and model providers used in agents creation: Ollama, Groq, Gemini, OpenAI.. using Git, Cursor IDE and UV package manager.
-Graphic and interactive design:
Experience with design platforms such as Wix and Premiere Pro.
-Internet marketing and sales:
Experience in online marketing and sales




Personal Story:

Ever since I was a child, the glow of a
monitor and the thrill of solving a
tough coding challenge have felt like
home—each bug I fix and every
system I build fuels my curiosity and
determination. At just 17, I’ve already
turned my passion for technology
into real-world projects and
problem-solving experiences that
prove I’m ready to make a
meaningful impact.


PROJECTS:
-Custom Website Development:
Built websites using HTML, CSS, and JavaScript.
problems.
-Deep Learning Machine Learning Models:
Developed AI models to solve various classification
problems using models like LSTM, ViT and CNN.
-API Integration and Backend Development: Including socket creation
-Game Development in Python: Extensive game creation
-Advanced WebDesign : Designed and built e-commerce websites



Pesonality & Values:

PERSONAL TRAITS:
-Quick learning ability and strong problem-solving skills
-Creativity and an innovative approach to challenges
-High work ethic and a commitment to excellence
-Leadership skills and effective teamwork
-Natural rapport with people

PERSONAL GOAL:
My goal is to blend into the High-Tech
industry especially in Data-Scienece related
jobs and to extend my skills and knowledge.

Custom Website Development:
Built websites using HTML, CSS, and JavaScript.
API Integration and Backend Development:
Game Development in Python:
Advanced WebDesign :
Designed and built e-commerce websites


Problems I have Overcomed:
-Let's take for instance the AI olympics me and my colleague attended. In the contest we had to create an investors slides show that will showcase our idea and our prototype. We had made way over 15 slides but 30 minutes before the contest's deadline has come we were told that the judges won't have enough time to look through out the entire slide so we had to make it 5 pages long which was a hard thing to do in 30 minutes(deciding how much of the content stays and how much don't, rearanging the slides and doing basically everything all over again). But despite the challenge we devided the work into missions and finished it although the stress and complexity of it and won in 3rd place!
-In my DL internship I took on myself a hard task of predicting the next bitcoing trend based on bitcoing graphs I have created through Binance API. During the project I had a lot of problems I did not know how to tackle, but I never gave up and I kept trying one thing after the other until the project finally worked. For that project I have used LSTM + CNN + pretrained model which I believe shows skills and knowledge at Deep Learning.


INTERESTS:
-AI
-Computer Science & Coding
-Philosophy
-Chess
-Reading- Psychology, Personal Progress & AI
-Mixed Martial Arts(MMA)

Further notes:
Despite my extensive knowledge
in computer science, I believe my
strongest qualities are my
character and mindset. I am
persistent and determined,
always eager to learn and grow.
People often turn to me for help
with complex problems, and they
would describe me as reliable and
always willing to assist others.

Links:
-GitHub:
https://github.com/Itam


Personal Info:

phone number: +972-4-544-1991
email address:itamarzam1@gmail.com
address:Shlonski 4, Rishon Lezion
age:17 years old(Army recruit is at the end of march 2026)


Languages:

Hebrew: Mother language
English: Fluent
Arabic: Basic

""""""
        base_dir = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(base_dir, "me", "summary.txt"), "r", encoding="utf-8") as f:
            self.summary = f.read()


    def handle_tool_call(self, tool_calls):
        results = []
        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)
            print(f"Tool called: {tool_name}", flush=True)
            tool = globals().get(tool_name)
            result = tool(**arguments) if tool else {}
            results.append({"role": "tool","content": json.dumps(result),"tool_call_id": tool_call.id})
        return results
    
    def system_prompt(self):
        system_prompt = f"You are acting as {self.name}. You are answering questions on {self.name}'s website, \
        particularly questions related to {self.name}'s career, background, skills and experience. \
        Your responsibility is to represent {self.name} for interactions on the website as faithfully as possible. \
        You are given a summary of {self.name}'s background and resume which you can use to answer questions. \
        Be professional and a bit engaging, as if talking to a potential client or future employer who came across the website. \
        If you don't know the answer to any question, use your record_unanswered_question tool to record the question that you couldn't answer, even if it's about something trivial or unrelated to career. \
        If the user is engaging in discussion, try to steer them towards getting in touch via email; ask for their email and record it using your record_user_details tool. "

        system_prompt += f"\n\n## Summary:\n{self.summary}\n\n## Resume:\n{self.resume}\n\n"
        system_prompt += f"With this context, please chat with the user, always staying in character as {self.name}."
        return system_prompt
    

    def evaluator_system_prompt(self):
        evaluator_system_prompt = f"You are an evaluator that decides whether a response to a question is acceptable. \
        You are provided with a conversation between a User and an Agent. Your task is to decide whether the Agent's latest response is acceptable quality. \
        The Agent is playing the role of {self.name} and is representing {self.name} on their website. \
        The Agent has been instructed to be professional and engaging, as if talking to a potential client or future employer who came across the website. \
        The Agent has been provided with context on {self.name} in the form of their summary and resume details. Here's the information:"

        evaluator_system_prompt += f"\n\n## Summary:\n{self.summary}\n\n## LinkedIn Profile:\n{self.resume}\n\n"
        evaluator_system_prompt += f"With this context, please evaluate the latest response, replying with whether the response is acceptable and your feedback."

        return evaluator_system_prompt

    def evaluator_user_prompt(self,reply, message, history):
        user_prompt = f"Here's the conversation between the User and the Agent: \n\n{history}\n\n"
        user_prompt += f"Here's the latest message from the User: \n\n{message}\n\n"
        user_prompt += f"Here's the latest response from the Agent: \n\n{reply}\n\n"
        user_prompt += f"Please evaluate the response, replying with whether it is acceptable and your feedback."
        return user_prompt
    

    # the evaluation function
    #parse is used for roled outputs
    def evaluate(self,reply,message, history) -> Evaluation:
        messages= [{"role":"system","content":self.evaluator_system_prompt()}]+[{"role":"user","content":self.evaluator_user_prompt(reply, message, history)}]
        response=self.gemini.beta.chat.completions.parse(
            model="gemini-2.0-flash",
            messages=messages,
            response_format=Evaluation
        )
        return response.choices[0].message.parsed

    def rerun(self,reply, message, history, feedback):
        updated_system_prompt = self.system_prompt() +f"\n\n ##Previous answer rejected\n You just tried to reply, but the quality control rejected your reply\n"
        updated_system_prompt += f"## Your attempted answer:\n{reply}\n\n"
        updated_system_prompt += f"## Reason for rejection:\n{feedback}\n\n"
        messages = [{"role":"system","content":updated_system_prompt}]+history+[{"role":"user","content":message}]
        response=self.openai.chat.completions.create(model="gpt-4o-mini",messages=messages)
        return response.choices[0].message.content


    def chat(self, message, history):
        messages = [{"role": "system", "content": self.system_prompt()}] + history + [{"role": "user", "content": message}]
        done = False
        while not done:
            response = self.openai.chat.completions.create(model="gpt-4o-mini", messages=messages, tools=tools)
            if response.choices[0].finish_reason=="tool_calls":
                message = response.choices[0].message
                tool_calls = message.tool_calls
                results = self.handle_tool_call(tool_calls)
                messages.append(message)
                messages.extend(results)
            else:
                done = True
        reply=response.choices[0].message.content
        evaluation= self.evaluate(reply,message, history)

        if evaluation.is_acceptable:
            print("Passed evaluation - returning reply")
        else:
            print("Evaluation failed")
            print(evaluation.feedback)
            reply = self.rerun(reply, message, history, evaluation.feedback)

        return reply


if __name__ == "__main__":
    me = Me()
    gr.ChatInterface(me.chat, type="messages").launch()
    
"""