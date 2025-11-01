import os
import requests
from datetime import date, timedelta
from dotenv import load_dotenv
from dataclasses import dataclass
from functools import lru_cache

load_dotenv()

@dataclass
class WeatherData:
    main: str
    description: str
    icon: str
    temperature: float
    city: str
    country: str
    feels_like: str
    date: date

@lru_cache(maxsize=128)
def get_lat_lon(city):
    try:
        params = {
            'q': city,
            'appid': os.getenv('OPENWEATHERMAP_API_KEY')
        }
        response = requests.get(os.getenv('LAT_LON_URL'), params=params)
        response.raise_for_status()
        data = response.json()[0]
        return data
    except requests.exceptions.RequestException as e:
        print(f"An error occurred while fetching geo data: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

@lru_cache(maxsize=128)
def get_weather_data(lat, lon, date):
    try:
        params = {
            'lat': lat,
            'lon': lon,
            'date': date,
            'appid': os.getenv('OPENWEATHERMAP_API_KEY'),
            'units': 'metric'
        }
        response = requests.get(os.getenv('WEATHER_URL'), params=params)
        response.raise_for_status()

        data = response.json()
        weather_data = WeatherData(
            main=data['weather'][0]['main'],
            description=data['weather'][0]['description'].capitalize(),
            icon=data['weather'][0]['icon'],
            temperature=data['main']['temp'],
            city=data['name'],
            country=data['sys']['country'],
            feels_like=data['main']['feels_like'],
            date=date
        )
        return weather_data   
    except requests.exceptions.RequestException as e:
        print(f"An error occurred while fetching weather data: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

def get_three_days_weather(city):
    geo_data = get_lat_lon(city)
    weather_data = []
    for i in range(3):
        data = get_weather_data(geo_data['lat'], geo_data['lon'], date.today() + timedelta(days=i))
        weather_data.append(data)
    return weather_data
