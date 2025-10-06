# tests/test_tools.py
import pytest
import requests
from agent import fetch_weather

def test_fetch_weather_success(mocker):
    """Tests the fetch_weather function for a successful API call."""
    # Mock the requests.get call to simulate a successful response
    mock_response = mocker.Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"main": {"temp": 25}, "weather": [{"description": "haze"}]}
    mocker.patch("requests.get", return_value=mock_response)

    # Call the function
    result = fetch_weather("Bengaluru")

    # Check if the result is what we expect
    assert result["main"]["temp"] == 25
    assert result["weather"][0]["description"] == "haze"

def test_fetch_weather_city_not_found(mocker):
    """Tests the fetch_weather function for a 404 'Not Found' error."""
    # Mock the requests.get call to simulate an error
    mock_response = mocker.Mock()
    mock_response.status_code = 404
    mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError
    mocker.patch("requests.get", return_value=mock_response)

    # Call the function with a city that doesn't exist
    result = fetch_weather("Atlantis")

    # Check if the error is handled correctly
    assert "error" in result
    assert "City 'Atlantis' not found" in result["error"]