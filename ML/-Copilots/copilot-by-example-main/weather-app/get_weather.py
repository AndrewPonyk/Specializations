import requests

api_key = '<your api key here>'

# A function that uses the requests library to make a GET
# request to the OpenWeatherMap API for a certain location

def get_weather(location):
    url = f'http://api.openweathermap.org/data/2.5/weather?q={location}&appid={api_key}'
    response = requests.get(url)
    weather_data = response.json()
    return weather_data