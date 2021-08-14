from tools.api import get_weather_data, get_joke_data
import random
import json
import torch
from tools.model import ChatBotModule
from tools.nltk_utils import bag_of_words, tokenize
from datetime import datetime

with open('data/intents.json', 'r') as json_data:
    intents = json.load(json_data)

data = torch.load("data/params.pth")

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']

model = ChatBotModule.load_from_checkpoint(checkpoint_path="data/model.ckpt")

model.eval()

def save_msg(msg, tag):
    with open("data/dialog.txt", "a+") as f:
        f.write(f"{msg}\t{tag}\n")

def predict(sentence):
    temp = sentence
    sentence = tokenize(sentence)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X)

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
    try:
        nick = kwargs.get('user').nick or kwargs.get('user')
    except Exception:
        nick = kwargs.get('user')['nick']
    return f'***{nick}***'

def get_time_response(**kwargs):
    return datetime.now().strftime("%H:%M")

def get_date_response(**kwargs):
    return datetime.now().strftime("%A, %d/%m/%Y")
    
DEGREE_SIGN = u"\N{DEGREE SIGN}"

responses = {
    "<user>": get_user_nickname,
    "<weather>": get_weather_response,
    "<joke>": get_joke_response,
    "<time>": get_time_response,
    "<date>": get_date_response,
    # "<knowledge>": None,
    # "<song>": None
}

def get_msg(msg, ent, **kwargs):
    return msg.replace(ent, responses[ent](**kwargs))