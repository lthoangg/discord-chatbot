from dotenv import load_dotenv
load_dotenv()
import discord

from tools.response import *

client = discord.Client()
@client.event
async def on_ready():
    print('We have logged in as {0.user}'.format(client))

@client.event
async def on_message(message):

    if str(message.channel) in ["testing", "test-bot"]:
        if message.author == client.user:
            return

        if message.content.startswith('/b '):
            user_msg = message.content.replace("/b", "")

            msg, entities = get_response(user_msg)
            for ent in entities:
                msg = get_msg(msg, ent, user=message.author)

            await message.channel.send(msg)
if __name__ == "__main__":
    my_secret = os.environ['TOKEN']
    client.run(os.getenv('TOKEN', my_secret))
    print("server is down!")



