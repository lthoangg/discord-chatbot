import requests
import os

def get_data(url):
  return requests.get(url, headers = {'User-agent': 'Mozilla/5.0'}).json()

def get_weather_data(city):
  # city_name
  weather = "https://api.openweathermap.org/data/2.5/weather?q={}&appid={}"
  try:
    weather_api = os.environ['TOKEN']
    data = get_data(weather.format(city, os.getenv("weather_api", weather_api)))

    temp = data['main']['temp']
    weather = data['weather'][0]['description']
    return {'temp': temp, 'weather': weather}
  except Exception:
    return None

def get_joke_data(kind = None):
  if kind:
    api = f"https://official-joke-api.appspot.com/jokes/{kind}/random"
  else:
    api = "https://official-joke-api.appspot.com/jokes/random"

  return get_data(api)

# print(get_weather('hanoi'))