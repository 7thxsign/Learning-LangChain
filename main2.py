# main2_fixed.py
from dataclasses import dataclass
import requests
from dotenv import load_dotenv
import os
import sys

load_dotenv()

@dataclass
class Context:
    user_id: str

@dataclass
class ResponseFormat:
    summary: str
    temperature_celcius: float
    temperature_fahrenheit: float
    humidity: float

def get_weather(city: str):
    """Gets the weather for a given city using wttr.in (returns JSON or raises)."""
    try:
        # wttr.in JSON format: https://wttr.in/:help
        resp = requests.get(f"https://wttr.in/{city}?format=j1", timeout=10)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Error getting weather: {e}")

def locate_user(context: Context) -> str:
    """Simple location lookup based on user_id."""
    match context.user_id:
        case 'ABC123':
            return 'Mangaluru'
        case 'XYZ789':
            return 'Bengaluru'
        case 'LMN456':
            return 'Chennai'
        case _:
            return 'Unknown'

def c_to_f(c: float) -> float:
    return c * 9.0 / 5.0 + 32.0

def build_response_for_user(context: Context) -> ResponseFormat:
    city = locate_user(context)
    if city == 'Unknown':
        summary = "I couldn't determine your location from the provided context."
        return ResponseFormat(summary=summary, temperature_celcius=0.0, temperature_fahrenheit=0.0, humidity=0.0)

    data = get_weather(city)

    # wttr.in JSON layout: current_condition is a list with first element having 'temp_C', 'humidity', 'weatherDesc'
    try:
        current = data['current_condition'][0]
        temp_c = float(current.get('temp_C'))
        humidity = float(current.get('humidity'))
        weather_desc_list = current.get('weatherDesc', [])
        weather_desc = weather_desc_list[0].get('value') if weather_desc_list else "No description"
    except Exception as e:
        raise RuntimeError(f"Unexpected weather data format: {e}")

    temp_f = c_to_f(temp_c)
    # Compose a light-hearted summary (you can replace this with LLM-generated text)
    summary = (
        f"Right now in {city}: {weather_desc}. "
        f"Temperature is {temp_c:.1f}°C ({temp_f:.1f}°F) with humidity around {humidity:.0f}%. "
        "Stay hydrated — I'm just a bot but even I feel the humidity! 😄"
    )

    return ResponseFormat(
        summary=summary,
        temperature_celcius=temp_c,
        temperature_fahrenheit=temp_f,
        humidity=humidity
    )

if __name__ == "__main__":
    # Example invocation (mirrors how you called agent.invoke)
    ctx = Context(user_id="ABC123")
    try:
        resp = build_response_for_user(ctx)
    except RuntimeError as e:
        print("Error:", e, file=sys.stderr)
        sys.exit(1)

    print("\n--- Agent's Final Answer (simulated) ---\n")
    print("Structured response object:", resp)
    print("\nSummary:\n", resp.summary)
    print("\nTemperature (°C):", resp.temperature_celcius)
    print("Temperature (°F):", resp.temperature_fahrenheit)
    print("Humidity (%):", resp.humidity)
