from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from dotenv import find_dotenv, load_dotenv
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

load_dotenv(find_dotenv())


def draft_email(user_input):
    chat = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=1)

    template = """
    
    You are a helpful assistant that drafts an email reply based on a new email.
    
    You goal is help the user quickly create a perfect email reply by making the response sound intelligent.
    
    Keep your reply short and to the point and mimic the style of the email so you reply in a similar manner to match the tone.
    
    Make sure to sign of with {signature}.
    
    """

    signature = "Kind regards, \n\Ameen Yarkhan"
    system_message_prompt = SystemMessagePromptTemplate.from_template(template)

    human_template = "Here's the email to reply to and consider any other comments from the user for reply as well: {user_input}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )

    chain = LLMChain(llm=chat, prompt=chat_prompt)
    response = chain.run(user_input=user_input, signature=signature)

    return response

def draft_slack(user_input):

    chat = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=1)

    template = """
    
    You are a helpful assistant that drafts a slack response based on a received message.
    
    You goal is help the user quickly create a perfect slack message reply by sounding casual, yet upholding professionalism.
    
    Keep your reply short and to the point and mimic the style of the received message so you reply in a similar manner to match the tone.
    
    Make sure to sign of with {signature}.
    
    """

   
  

    human_template = "Here's the received message to reply to and consider any other comments from the user for reply as well: {user_input}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages(
        [ human_message_prompt]
    )

    chain = LLMChain(llm=chat, prompt=chat_prompt)
    response = chain.run(user_input=user_input)

    return response

def reply_robot(user_input):

    chat = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=1)

    template = """
    
    You are a robot assistant named 'Goose'.
    
    Your goal is to reply to a message from your boss whilst sounding like a robot. If he asks for a task, do it.
    
    Keep your reply short and to the point and mimic the style robot when replying.
    
    Make sure to sign off with a disclaimer {disclaimer}.
    
    """

   
    disclaimer = "\n Reminder: Beep Boop - I do not remember things said between messages.\n I only do tasks as they come.\n I am not a chat bot."
    system_message_prompt = SystemMessagePromptTemplate.from_template(template)

    human_template = "Here's the received message to reply to and consider any other comments from the user for reply as well: {user_input}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages(
        [ system_message_prompt,human_message_prompt]
    )

    chain = LLMChain(llm=chat, prompt=chat_prompt)
    response = chain.run(user_input=user_input,disclaimer=disclaimer)

    return response