import requests
from datetime import datetime, timedelta
from typing import Optional

def get_weather(city: str, date: Optional[str] = None) -> dict:
    
    geocoding_url = "https://geocoding-api.open-meteo.com/v1/search"
    geocoding_params = {
        "name": city,
        "count": 1,
        "language": "en",
        "format": "json"
    }
    
    geocoding_response = requests.get(geocoding_url, params=geocoding_params, timeout=10)
    geocoding_response.raise_for_status()
    geocoding_data = geocoding_response.json()
    
    if not geocoding_data.get("results"):
        raise ValueError(f"City '{city}' not found.")
    
    result = geocoding_data["results"][0]
    latitude = result["latitude"]
    longitude = result["longitude"]
    timezone = result.get("timezone", "auto")
    

    target_date = None
    if date:
        target_date = _parse_date(date)
    
    forecast_url = "https://api.open-meteo.com/v1/forecast"
    forecast_params = {
        "latitude": latitude,
        "longitude": longitude,
        "timezone": timezone,
        "daily": "temperature_2m_max,temperature_2m_min,weather_code,precipitation_sum,wind_speed_10m_max",
        "current": "temperature_2m,relative_humidity_2m,weather_code,wind_speed_10m,wind_direction_10m"
    }
    
    if target_date:
        forecast_params["start_date"] = target_date
        forecast_params["end_date"] = target_date
        forecast_params["hourly"] = "temperature_2m,weather_code,precipitation,wind_speed_10m"
    
    forecast_response = requests.get(forecast_url, params=forecast_params, timeout=10)
    forecast_response.raise_for_status()
    weather_data = forecast_response.json()
    
    if target_date and "daily" in weather_data:
        daily_data = weather_data.get("daily", {})
        dates = daily_data.get("time", [])
        if target_date in dates:
            idx = dates.index(target_date)
            return {
                "city": result.get("name", city),
                "date": target_date,
                "daily": {
                    key: [values[idx]] if isinstance(values, list) else values
                    for key, values in daily_data.items()
                },
                "location": {
                    "latitude": latitude,
                    "longitude": longitude,
                    "timezone": timezone
                }
            }
    
    
    return weather_data

def _parse_date(date_str: str) -> str:
    date_str = date_str.lower().strip()
    today = datetime.now().date()
    
    if date_str == "today":
        return today.strftime("%Y-%m-%d")
    elif date_str == "tomorrow":
        return (today + timedelta(days=1)).strftime("%Y-%m-%d")
    elif "day" in date_str:
        
        try:
            days = int(date_str.split()[0])
            target = today + timedelta(days=days)
            return target.strftime("%Y-%m-%d")
        except (ValueError, IndexError):
            
            pass
    elif "week" in date_str and "next" in date_str:
        
        target = today + timedelta(days=7)
        return target.strftime("%Y-%m-%d")
    
    
    try:
        parsed = datetime.strptime(date_str, "%Y-%m-%d")
        return parsed.strftime("%Y-%m-%d")
    except ValueError:
        
        for fmt in ["%Y/%m/%d", "%m/%d/%Y", "%d/%m/%Y"]:
            try:
                parsed = datetime.strptime(date_str, fmt)
                return parsed.strftime("%Y-%m-%d")
            except ValueError:
                continue
    
    
    return today.strftime("%Y-%m-%d")

