# IT-Project-3


## Python environment

python -m venv .venv_it_proj3

source .venv_it_proj3/bin/activate

## Explore relationships

- Temperature and Dew Point Temperature:
    - Relationship: Investigate how the current temperature (temp_c) is related to the past dew point temperature (dew_point_temp_c).
    - Lags to test: 1 hour, 3 hours, or even up to 24 hours to see if past dew point temperatures influence the current temperature.

- Real Humidity Percentage and Dew Point Temperature:
    - Relationship: Examine how the real humidity percentage (real_hum_pct) is related to the past dew point temperature (dew_point_temp_c).
    - Lags to test: Test different lags to capture potential dependencies.
    
- Weather Conditions and Temperature/Real Humidity Percentage:
    - Relationship: Investigate how weather conditions (e.g., Fog, Freezing Drizzle, etc.) are related to the current temperature (temp_c) or real humidity percentage (real_hum_pct).
    - Lags to test: Test the relationship with the current weather condition. Lag it by 1 hour to explore potential effects on temperature or humidity in the following hour.

- Wind Speed and Visibility:
    - Relationship: Explore how the wind speed (wind_speed_kmh) affects the visibility (visibility_km).
    - Lags to test: Test different lags to understand if wind speed influences visibility immediately or with a delay.

- Pressure and Weather Conditions:
    - Relationship: Examine how atmospheric pressure (press_kpa) is associated with different weather conditions.
    - Lags to test: Test the relationship with the current pressure. Consider different lags to identify potential patterns.

- Temperature and Weather Conditions:
    - Relationship: Investigate how the current temperature (temp_c) is related to specific weather conditions (e.g., Fog, Freezing Drizzle, etc.).
    - Lags to test: Test different lags to see how temperature relates to the occurrence of particular weather conditions.
