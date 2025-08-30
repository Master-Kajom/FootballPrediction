import requests
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import time
import traceback

# Configuration
API_KEY = '0eabd7aff1954618968f10525f1c1c1d'
BASE_URL = 'https://api.football-data.org/v4/'
HEADERS = {'X-Auth-Token': API_KEY}

def safe_request(url: str, max_retries: int = 3, params: dict = None) -> Optional[dict]:
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

def get_team_id(team_name: str, competition_code: str = 'PL') -> Optional[int]:
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

def get_head_to_head(home_team: str, away_team: str) -> Dict:
    """
    Get head-to-head statistics between two teams
    
    Args:
        home_team: Name of the home team
        away_team: Name of the away team
        
    Returns:
        Dictionary containing head-to-head statistics
    """
    try:
        # First, get team IDs
        home_id = get_team_id(home_team)
        away_id = get_team_id(away_team)
        
        if not home_id or not away_id:
            print(f"Could not find IDs for one or both teams: {home_team}, {away_team}")
            return {}
            
        # Get the most recent match between these teams to get the match ID
        recent_matches_url = f"{BASE_URL}teams/{home_id}/matches"
        params = {
            'dateFrom': '2024-01-01',
            'dateTo': '2025-08-29'
        }
        
        print(f"Requesting URL: {recent_matches_url}")
        response = safe_request(recent_matches_url, params=params)
        
        # if response.status_code != 200:
        #     print(f"Error fetching recent matches: {response.text if response else 'No response'}")
        #     return {}
            
        matches = response.get('matches', [])
        match_id = None
        
        # Find the most recent match between these two teams
        for match in matches:
            if (match.get('homeTeam', {}).get('id') == away_id or 
                match.get('awayTeam', {}).get('id') == away_id):
                match_id = match.get('id')
                break
                
        if not match_id:
            print(f"No recent matches found between {home_team} and {away_team}")
            return {}
            
        # Get head-to-head data
        print("\n=== DEBUG: Getting head-to-head data ===")
        h2h_url = f"{BASE_URL}matches/{match_id}/head2head"
        h2h_params = {
            'limit': 10,
            'competitions': '2021'  # Get last 10 meetings
        }
        
        print(f"Requesting URL: {h2h_url}")
        h2h_response = safe_request(h2h_url, params=h2h_params)
        
        if not h2h_response:
            print("No response received from head-to-head endpoint")
            return {}
            
        #print(f"Response status: {h2h_response.status_code}")
        
        # if h2h_response.status_code != 200:
        #     print(f"Error fetching head-to-head data: {h2h_response.text}")
        #     return {}
            
        h2h_data = h2h_response
        
        # Process head-to-head data
        home_team_key = home_team.lower().replace(' ', '_').replace('_fc', '').replace('_afc', '')
        away_team_key = away_team.lower().replace(' ', '_').replace('_fc', '').replace('_afc', '')
        
        result = {
            f"{home_team_key}_home_h2h": [],
            f"{home_team_key}_home_h2h_avg": 0.0,
            f"{away_team_key}_away_h2h": [],
            f"{away_team_key}_away_h2h_avg": 0.0
        }
        
        # Check if we have matches in the response
        if 'matches' not in h2h_data or not isinstance(h2h_data['matches'], list):
            print("No match data found in head-to-head response")
            return result
            
        matches = h2h_data['matches']
        
        if not matches:
            print("No matches found in head-to-head data")
            return result
            
        # Process each match
        home_results = []
        away_results = []
        
        for match in matches:
            home_team_match = match.get('homeTeam', {}).get('name', '').lower()
            away_team_match = match.get('awayTeam', {}).get('name', '').lower()
            score = match.get('score', {})
            
            if not all([home_team_match, away_team_match, score]):
                continue
                
            # Determine if home team in this match is our home team
            is_home_match = (home_team_match == home_team.lower())
            winner = score.get('winner')
            
            if winner == 'DRAW':
                home_results.append(0.5)
                away_results.append(0.5)
            elif winner == 'HOME_TEAM':
                if is_home_match:
                    home_results.append(1.0)  # Our home team won at home
                    away_results.append(0.0)
                else:
                    away_results.append(1.0)  # Our away team won at home
                    home_results.append(0.0)
            elif winner == 'AWAY_TEAM':
                if is_home_match:
                    home_results.append(0.0)  # Our home team lost at home
                    away_results.append(1.0)
                else:
                    away_results.append(0.0)  # Our away team lost away
                    home_results.append(1.0)
        
        # Calculate averages
        if home_results:
            result[f"{home_team_key}_home_h2h"] = home_results
            result[f"{home_team_key}_home_h2h_avg"] = sum(home_results) / len(home_results)
            
        if away_results:
            result[f"{away_team_key}_away_h2h"] = away_results
            result[f"{away_team_key}_away_h2h_avg"] = sum(away_results) / len(away_results)
        
        return result
        
    except Exception as e:
        print(f"Error in get_head_to_head: {e}")
        import traceback
        traceback.print_exc()
        return {}

if __name__ == "__main__":
    # Example usage
    home_team = "Manchester United FC"
    away_team = "Arsenal FC"
    
    h2h_stats = get_head_to_head(home_team, away_team)
    
    if h2h_stats:
        for key, value in h2h_stats.items():
            if isinstance(value, float):
                print(f"{key} = {value:.2f}  # Average result (1=Win, 0.5=Draw, 0=Loss)")
            else:
                print(f"{key} = {value}  # Last {len(value)} matches (1=Win, 0.5=Draw, 0=Loss)")
    else:
        print("Could not fetch head-to-head statistics.")
