import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Configuration
API_KEY = os.getenv('FOOTBALL_DATA_API_KEY', '0eabd7aff1954618968f10525f1c1c1d')
BASE_URL = 'https://api.football-data.org/v4/'
HEADERS = {'X-Auth-Token': API_KEY}

# Competition and Season
DEFAULT_COMPETITION = 'PL'  # Premier League
CURRENT_SEASON = 2025       # 2024-2025 season

# Team Name Mappings (to handle different name formats across APIs)
TEAM_NAME_MAPPING = {
    'Manchester United': 'Manchester United FC',
    'Arsenal': 'Arsenal FC',
    # Add more mappings as needed
}

def normalize_team_name(team_name: str) -> str:
    """Normalize team name to a standard format"""
    return TEAM_NAME_MAPPING.get(team_name, team_name)

def safe_request(url: str, params=None, max_retries: int = 3) -> dict:
    """Make a rate-limited request with retries"""
    import requests
    import time
    
    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=HEADERS, params=params)
            
            # Check rate limits
            if 'X-Requests-Available' in response.headers:
                remaining = int(response.headers['X-Requests-Available'])
                if remaining < 3:  # Leave buffer
                    wait_time = int(response.headers.get('Retry-After', 60))
                    print(f"Approaching rate limit. Waiting {wait_time} seconds...")
                    time.sleep(wait_time)
                    continue
            
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:  # Rate limited
                retry_after = int(e.response.headers.get('Retry-After', 60))
                print(f"Rate limited. Waiting {retry_after} seconds...")
                time.sleep(retry_after)
                continue
            print(f"HTTP Error: {e}")
            return {}
            
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
                continue
            return {}
    
    return {}