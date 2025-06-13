from get_weather import get_weather

# Ask user for their desired location to lookup the weather for
location = input('Enter the city: ')

# Use the function in get_weather.py to get the weather for the location
weather_data = get_weather(location)

def kelvin_to_fahrenheit(kelvin_temp):
    return (kelvin_temp - 273.15) * 9/5 + 32

# Print the high and low temperatures in fahrenheit for the location
high_temp = kelvin_to_fahrenheit(weather_data['main']['temp_max'])
low_temp = kelvin_to_fahrenheit(weather_data['main']['temp_min'])
print(f'High: {high_temp}°F')
print(f'Low: {low_temp}°F')
