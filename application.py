from flask import Flask, render_template, request, url_for
import pandas as pd
import chatbot
from difflib import SequenceMatcher


app = Flask(__name__)

welcome = "Hi! I'm Srilatha's Resume Chatbot. How can I help you?"
nomatch = "Sorry, I cannot help with that, I will note your question for her reference."
qafile = "srilatha-resume.csv"
thankyou = "Thank you for contacting me."
exitmessage = "Please type 'bye' to exit the chat."

bot = chatbot.Chatbot(nomatch, qafile, thankyou)
bot.run_bot()

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/resume", methods=['GET'])
def resume():
    return render_template("resume.html")
	
@app.route("/resume/getresponse")
def get_bot_response():
    userText = request.args.get('msg')
    return str(bot.get_response(userText))


if __name__ == "__main__":
    app.run()