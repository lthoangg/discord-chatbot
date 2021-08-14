from dotenv import load_dotenv
load_dotenv()
from tools.response import *

responses = {
    "<user>": get_user_nickname,
    "<weather>": get_weather_response,
    "<joke>": get_joke_response,
    "<time>": get_time_response,
    "<date>": get_date_response,
}

DEGREE_SIGN = u"\N{DEGREE SIGN}"

if __name__ == "__main__":
    while True:
        me = input("Me: ")

        lth = {"nick": 'user'}
        if "quit" in me:
            break
        else:
            msg, entities = predict(me)
            for ent in entities:
                msg = get_msg(msg, ent, user=lth)
        
        print("bot:", msg)
        
    print("server is down!")