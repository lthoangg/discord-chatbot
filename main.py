import os
import re
from dotenv import load_dotenv
load_dotenv()
import discord
from tools.api import get_weather_data, get_joke_data
import random
import json
import torch
from tools.model import NeuralNet
from tools.nltk_utils import bag_of_words, tokenize
from datetime import datetime

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('data/intents.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "data/data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

def save_msg(msg, tag):
    with open("data/dialog.txt", "a+") as f:
        f.write(f"{msg}\t{tag}\n")

def get_response(sentence):
    temp = sentence
    sentence = tokenize(sentence)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]
    save_msg(temp, tag)
    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.85:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                return f"{random.choice(intent['responses'])}", intent["entities"]
    else:
        return "I do not understand...", []

def get_weather_response(**kwargs):
    city = kwargs.get('city') or "Hanoi"
    data = get_weather_data(city)
    if data:
        return f"\nThe temperature in {city.capitalize()} is {'%.2f' % (data['temp'] - 270)}{DEGREE_SIGN}C and It's {data['weather']}."
    else:
        return f"{city} is not valid. Please try to correct the city's name!"

def get_joke_response(**kwargs):
    kind = kwargs.get('kind') or None
    data = get_joke_data(kind)

    return f"{data['setup']}\n{data['punchline']}"

def get_user_nickname(**kwargs):
    nick = kwargs.get('user').nick or kwargs.get('user')
    return f'***{nick}***'

def get_time_response(**kwargs):
    return datetime.now().strftime("%H:%M")

def get_date_response(**kwargs):
    return datetime.now().strftime("%A, %d/%m/%Y")

def get_do_math_response(problem):
    try:
        msg = re.findall("\d+\ +?[\+\-\*\/]\ +?\d+", problem)[0]
        msg = (random.choice(["It's so easy: ", "Too easy: ", "Try harder: "]) + "`{} = {:,}`").format(' '.join(msg.split(' ')), eval(msg))
    except Exception as e:
        msg = "I can't do this problem."
        msg = repr(e)
    return msg

client = discord.Client()

DEGREE_SIGN = u"\N{DEGREE SIGN}"

@client.event
async def on_ready():
    print('We have logged in as {0.user}'.format(client))

responses = {
    "<user>": get_user_nickname,
    "<weather>": get_weather_response,
    "<joke>": get_joke_response,
    "<time>": get_time_response,
    "<date>": get_date_response,
}

def get_msg(msg, ent, **kwargs):
    return msg.replace(ent, responses[ent](**kwargs))

@client.event
async def on_message(message):

    if message.author == client.user:
        return

    if message.content.startswith('/tb'):
        msg_inp = message.content.replace("/tb", "")

        if "cal" in msg_inp:
            msg = get_do_math_response(problem=msg_inp)
        else:
            msg, entities = get_response(msg_inp)
            for ent in entities:
                msg = get_msg(msg, ent, user=message.author)

        await message.channel.send(msg)
if __name__ == "__main__":
    my_secret = os.environ['TOKEN']
    client.run(os.getenv('TOKEN', my_secret))
    print("server is down!")



