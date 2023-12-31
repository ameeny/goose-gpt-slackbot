import os
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from slack_bolt.adapter.flask import SlackRequestHandler
from slack_bolt import App
from dotenv import find_dotenv, load_dotenv
from flask import Flask, request
import logging
import ssl
from functions import draft_email, draft_slack,reply_robot
# ssl._create_default_https_context = ssl._create_unverified_context
# logging.basicConfig(level=logging.DEBUG)


# Load environment variables from .env file
load_dotenv()

# Set Slack API credentials
SLACK_BOT_TOKEN = os.environ["SLACK_BOT_TOKEN"]
SLACK_SIGNING_SECRET = os.environ["SLACK_SIGNING_SECRET"]
SLACK_BOT_USER_ID = os.environ["SLACK_BOT_USER_ID"]

# Initialize the Slack app
app = App(token=SLACK_BOT_TOKEN,signing_secret=SLACK_SIGNING_SECRET)

# Initialize the Flask app
flask_app = Flask(__name__)
flask_app.config['WTF_CSRF_ENABLED'] = False
handler = SlackRequestHandler(app)


def get_bot_user_id():
    """
    Get the bot user ID using the Slack API.
    Returns:
        str: The bot user ID.
    """
    try:
        # Initialize the Slack client with your bot token
        slack_client = WebClient(token=os.environ["SLACK_BOT_TOKEN"])
        response = slack_client.auth_test()
        return response["user_id"]
    except SlackApiError as e:
        print(f"Error: {e}")


def my_function(text):
    """
    Custom function to process the text and return a response.
    In this example, the function converts the input text to uppercase.

    Args:
        text (str): The input text to process.

    Returns:
        str: The processed text.
    """
    response = text.upper()
    return response


@app.event("app_mention")
def handle_mentions(body, say):
    """
    Event listener for mentions in Slack.
    When the bot is mentioned, this function processes the text and sends a response.

    Args:
        body (dict): The event data received from Slack.
        say (callable): A function for sending a response to the channel.
    """
    text = body["event"]["text"]

    mention = f"<@{SLACK_BOT_USER_ID}>"
    text = text.replace(mention, "").strip()

    say("Beep boop - Goose Bot Activated 🤖 \n ")
    # response = my_function(text)
    response = reply_robot(text)
    say(response)

@app.command("/reply_email")
def reply_email(ack, say, command):
    """
    Handle the /reply_email slash command.
    
    Args:
        ack (callable): A function to acknowledge the incoming command.
        say (callable): A function for sending a response to the channel.
        command (dict): Data received from the slash command.
    """
    # Acknowledge the command
    ack()
    say("Beep boop - I'll get right on drafing an email reply for you 🤖.. ")
    # Process the command's text
    text = command.get('text') # The text after the slash command
    response = draft_email(text)

    # Send a response back to the channel
    say(response)

@app.command("/reply_slack")
def reply_slack(ack, say, command):
    """
    Handle the /reply_slack slash command.
    
    Args:
        ack (callable): A function to acknowledge the incoming command.
        say (callable): A function for sending a response to the channel.
        command (dict): Data received from the slash command.
    """
    # Acknowledge the command
    ack()
    say("Beep boop - I'll get right on drafing a slack reply for you 🤖.. ")
    # Process the command's text
    text = command.get('text') # The text after the slash command
    response = draft_slack(text)

    # Send a response back to the channel
    say(response)



@flask_app.route("/slack/events", methods=["POST"])
def slack_events():
    """
    Route for handling Slack events.
    This function passes the incoming HTTP request to the SlackRequestHandler for processing.
    """
    # data = request.json
    # print(f"\n\nIncoming data: {data}\n\n")

    # # If it's a URL verification challenge
    # if data.get("type") == "url_verification":
    #     return data.get("challenge")

    return handler.handle(request)




# Run the Flask app
if __name__ == "__main__":
    flask_app.run(port=5002)