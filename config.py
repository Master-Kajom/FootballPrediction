import os
from dotenv import load_dotenv
from typing import Dict, Tuple, Optional
import requests
import time

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

def safe_requests(url: str, params=None, max_retries: int = 3) -> dict:
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

def safe_request(url: str, params=None, max_retries: int = 3) -> Optional[dict]:
    """Make a rate-limited request with retries"""
    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=HEADERS, params=params)
            
            # Check rate limits
            if 'X-Requests-Available' in response.headers:
                remaining = int(response.headers['X-Requests-Available'])
                if remaining < 3:  # Leave buffer
                    wait_time = 60  # Default wait time in seconds
                    if 'Retry-After' in response.headers:
                        wait_time = int(response.headers['Retry-After'])
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
            return None
            
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
                continue
            return None
    
    return None

def get_competition_id(competition_code: str) -> Optional[int]:
    """
    Get the competition ID for a given competition code.
    
    Args:
        competition_code: The competition code (e.g., 'PL' for Premier League)
        
    Returns:
        int: The competition ID if found, None otherwise
    """
    if not competition_code:
        return None
        
    try:
        url = f"{BASE_URL}competitions"
        response = requests.get(url, headers=HEADERS)
        response.raise_for_status()
        
        competitions = response.json().get('competitions', [])
        
        for comp in competitions:
            if str(comp.get('code', '')).upper() == str(competition_code).upper():
                return comp.get('id')
                
        print(f"No competition found with code: {competition_code}")
        return None
        
    except requests.exceptions.RequestException as e:
        print(f"Error fetching competitions: {e}")
        return None

def get_team_id(team_name: str, competition_code: str = 'PL', competitionId: int = 2021) -> Optional[int]:
    """Get team ID by name using competition-based search first"""
    # Try to find in specified competition first (e.g., PL for Premier League)
    if competition_code:
        comp_teams_url = f"{BASE_URL}competitions/{competition_code}/teams"
        data = safe_request(comp_teams_url)
        
        if data and 'teams' in data:
            for team in data['teams']:
                if team['name'].lower() == team_name.lower():
                    return team['id']
    
    # If not found in specified competition, try direct search
    search_url = f"{BASE_URL}teams?name={team_name}"
    data = safe_request(search_url)
    
    if data and 'teams' in data and data['teams']:
        for team in data['teams']:
            if team['name'].lower() == team_name.lower():
                return team['id']
    
    print(f"Team '{team_name}' not found in competition '{competition_code}' or general search.")
    return None

