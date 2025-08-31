import requests
import pandas as pd
from datetime import datetime, timedelta
import time
from typing import Optional
import numpy as np

# Configuration
API_KEY = '0eabd7aff1954618968f10525f1c1c1d'  # Replace with your actual API key from football-data.org
BASE_URL = 'https://api.football-data.org/v4/'
HEADERS = {'X-Auth-Token': API_KEY}

def safe_request(url: str, max_retries: int = 3) -> Optional[dict]:
    """Make a rate-limited request with retries"""
    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=HEADERS)
            
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

def get_team_matches(team_id, status='FINISHED', limit=10, competitionId: int = 2021):
    """Get recent matches for a team"""
    url = f"{BASE_URL}teams/{team_id}/matches"
    print(f"\n=== DEBUG: Fetching matches from URL: {url} ===")
    params = {
        'competitions': competitionId,
        'dateFrom': '2025-03-01',
        'dateTo': '2025-08-29',
        #'limit': 15
    }
    print(f"=== DEBUG: Request params: {params} ===")
    
    try:
        response = requests.get(url, headers=HEADERS, params=params)
        #print(f"=== DEBUG: Response status code: {response.status_code} ===")
        response.raise_for_status()
        matches = response.json()
        #print(f"=== DEBUG: Raw matches response: ===\n{matches}")
        return matches.get('matches', [])
    except requests.exceptions.RequestException as e:
        print(f"Error fetching matches: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Response content: {e.response.text}")
        return []

def analyze_matches(matches, team_id, is_home_team=True):
    """Analyze match statistics for a team, separating home and away stats"""
    print(f"\n=== DEBUG: Analyzing matches for team_id: {team_id} (is_home_team: {is_home_team}) ===")
    print(f"=== DEBUG: Number of matches to analyze: {len(matches) if matches else 0} ===")
    
    stats = {
        'goals_scored': [],
        'goals_conceded': [],
        'results': []  # 1=Win, 0.5=Draw, 0=Loss
    }
    
    for match in matches:
        # print(f"\n=== DEBUG: Processing match ===")
        # print(f"Match ID: {match.get('id')}")
        # print(f"Home Team: {match.get('homeTeam', {}).get('name')} (ID: {match.get('homeTeam', {}).get('id')})")
        # print(f"Away Team: {match.get('awayTeam', {}).get('name')} (ID: {match.get('awayTeam', {}).get('id')})")
        # print(f"Score: {match.get('score', {}).get('fullTime', {})}")
        
        team_is_home = match['homeTeam']['id'] == team_id
        opponent_is_home = not team_is_home
        
        # Get scores
        home_goals = match['score']['fullTime'].get('home', 0)
        away_goals = match['score']['fullTime'].get('away', 0)
        
        # Determine if this is a home or away match for our team
        if team_is_home:
            goals_for = home_goals
            goals_against = away_goals
        else:
            goals_for = away_goals
            goals_against = home_goals
        
        # Calculate result (1=Win, 0.5=Draw, 0=Loss)
        if goals_for > goals_against:
            result = 1.0  # Win
        elif goals_for == goals_against:
            result = 0.5  # Draw
        else:
            result = 0.0  # Loss
            
        #print(f"Team goals: {goals_for}, Opponent goals: {goals_against}, Result: {result}")
        
        stats['goals_scored'].append(goals_for)
        stats['goals_conceded'].append(goals_against)
        stats['results'].append(result)
    
    print(f"\n=== DEBUG: Final stats for team_id {team_id} ===")
    print(f"Goals scored: {stats['goals_scored']}")
    print(f"Goals conceded: {stats['goals_conceded']}")
    print(f"Results: {stats['results']}")
    print(f"Average goals scored: {np.mean(stats['goals_scored']) if stats['goals_scored'] else 0:.2f}")
    print(f"Average goals conceded: {np.mean(stats['goals_conceded']) if stats['goals_conceded'] else 0:.2f}")
    print(f"Form average: {np.mean(stats['results']) if stats['results'] else 0.5:.2f}")
    
    return stats

def get_match_stats(home_team: str, away_team: str, competition_code: str = 'PL', competitionId: int = 2021):
    """Get match statistics for two teams in a specific competition"""
    print(f"Fetching data for {home_team} (home) vs {away_team} (away) in competition {competition_code}...\n")
    
    # Get team IDs with competition code
    home_id = get_team_id(home_team, competition_code,competitionId)
    away_id = get_team_id(away_team, competition_code,competitionId)
    
    if not home_id or not away_id:
        return None, None
    
    # Get recent matches for home team (home matches only)
    home_team_matches = get_team_matches(home_id, limit=20, competitionId=competitionId)  # Get more matches to find 5 home games
    home_stats = analyze_matches(home_team_matches, home_id, is_home_team=True)
    
    # Get recent matches for away team (away matches only)
    away_team_matches = get_team_matches(away_id, limit=20,competitionId=competitionId)  # Get more matches to find 5 away games
    away_stats = analyze_matches(away_team_matches, away_id, is_home_team=False)
    
    return home_stats, away_stats

def format_stats(home_team, away_team, home_stats, away_stats):
    """Format the statistics for display"""
    if not home_stats or not away_stats:
        return "Error: Could not fetch statistics for one or both teams."
    
    # Calculate averages
    home_avg_scored = sum(home_stats['goals_scored']) / len(home_stats['goals_scored'])
    home_avg_conceded = sum(home_stats['goals_conceded']) / len(home_stats['goals_conceded'])
    away_avg_scored = sum(away_stats['goals_scored']) / len(away_stats['goals_scored'])
    away_avg_conceded = sum(away_stats['goals_conceded']) / len(away_stats['goals_conceded'])
    
    # Format the output
    output = f"""--- {home_team} vs {away_team} Statistics ---

{home_team} (Last 5 home matches):
  • Goals Scored: {home_stats['goals_scored']}
  • Goals Conceded: {home_stats['goals_conceded']}
  • Results: {['Win' if x == 1 else 'Draw' if x == 0.5 else 'Loss' for x in home_stats['results']]}
  • Avg. Goals Scored: {home_avg_scored:.1f}
  • Avg. Goals Conceded: {home_avg_conceded:.1f}

{away_team} (Last 5 away matches):
  • Goals Scored: {away_stats['goals_scored']}
  • Goals Conceded: {away_stats['goals_conceded']}
  • Results: {['Win' if x == 1 else 'Draw' if x == 0.5 else 'Loss' for x in away_stats['results']]}
  • Avg. Goals Scored: {away_avg_scored:.1f}
  • Avg. Goals Conceded: {away_avg_conceded:.1f}
"""
    return output

if __name__ == "__main__":
    # Example usage with competition code
    COMPETITION_CODE = 'PL'  # Premier League
    HOME_TEAM = "Manchester United FC"
    AWAY_TEAM = "Burnley FC"
    
    # Get the statistics with competition code
    home_stats, away_stats = get_match_stats(HOME_TEAM, AWAY_TEAM, COMPETITION_CODE)
    
    # Format and print the results
    if home_stats and away_stats:
        # Generate variable names from team names
        home_var = HOME_TEAM.lower().replace(' ', '_').replace('_fc', '').replace('_afc', '')
        away_var = AWAY_TEAM.lower().replace(' ', '_').replace('_fc', '').replace('_afc', '')
        
        # Format the output as requested
        print(f"# {HOME_TEAM} vs {AWAY_TEAM} Statistics\n")
        print(f"# Last 5 home matches for {HOME_TEAM}")
        print(f"{home_var}_home_goals = {[round(x, 1) for x in home_stats['goals_scored']]}  # Last 5 home matches")
        print(f"{home_var}_home_conceded = {[round(x, 1) for x in home_stats['goals_conceded']]}  # Last 5 home matches\n")
        
        print(f"# Last 5 away matches for {AWAY_TEAM}")
        print(f"{away_var}_away_goals = {[round(x, 1) for x in away_stats['goals_scored']]}  # Last 5 away matches")
        print(f"{away_var}_away_conceded = {[round(x, 1) for x in away_stats['goals_conceded']]}  # Last 5 away matches")
        
        # Also print form for reference
        print(f"\n# Form (1=Win, 0.5=Draw, 0=Loss)")
        print(f"{home_var}_form = {home_stats['results']}  # Last 5 matches")
        print(f"{away_var}_form = {away_stats['results']}  # Last 5 matches")
