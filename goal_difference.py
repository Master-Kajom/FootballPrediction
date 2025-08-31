import os
from typing import Dict, Tuple, Optional
import requests
#from dotenv import load_dotenv

# Load environment variables
#load_dotenv()

# Configuration
#API_KEY = os.getenv('FOOTBALL_DATA_API_KEY')
API_KEY = '0eabd7aff1954618968f10525f1c1c1d'
BASE_URL = 'https://api.football-data.org/v4/'
HEADERS = {'X-Auth-Token': API_KEY}

def get_team_goal_differences(competition_code: str, season: int, competitionId: int) -> Dict[str, Dict[str, int]]:
    """
    Calculate goal differences for all teams in a specific league and season.
    
    Args:
        competition_code: League code (e.g., 'PL' for Premier League)
        season: End year of the season (e.g., 2024 for 2023/24 season)
        
    Returns:
        Dictionary with team names as keys and their goal difference stats
    """
    print(f"\n=== DEBUG: Getting goal differences for {competition_code} season {season-1}-{season%100} ===")
    
    # Get all matches for the competition and season
    competition_id = competitionId #get_competition_id(competition_code)
    matches_url = f"{BASE_URL}competitions/{competition_id}/matches"
    params = {
        'dateFrom': '2025-08-01',
        'dateTo': '2025-08-29',
        'status': 'FINISHED'
    }
    
    print(f"Requesting URL: {matches_url}")
    print(f"With params: {params}")
    
    try:
        response = requests.get(matches_url, headers=HEADERS, params=params)
        #print(f"Response status: {response.status_code}")
        
        if response.status_code != 200:
            print(f"Error fetching matches: {response.text}")
            return {}
            
        data = response.json()
        #print(f"Raw response received. Found {len(data.get('matches', []))} matches.")
        
        # Initialize team stats
        team_stats = {}
        
        # Process each match
        for i, match in enumerate(data.get('matches', []), 1):
            home_team = match.get('homeTeam', {}).get('name', 'Unknown')
            away_team = match.get('awayTeam', {}).get('name', 'Unknown')
            
            #print(f"\nMatch {i}: {home_team} vs {away_team}")
            
            # Skip if score is not available
            if 'fullTime' not in match.get('score', {}):
                print("  No score available, skipping...")
                continue
                
            home_goals = match['score']['fullTime'].get('home', 0)
            away_goals = match['score']['fullTime'].get('away', 0)
            
            #print(f"  Score: {home_goals}-{away_goals}")
            
            # Update home team stats
            if home_team not in team_stats:
                team_stats[home_team] = {'goalsFor': 0, 'goalsAgainst': 0, 'matches': 0}
            team_stats[home_team]['goalsFor'] += home_goals
            team_stats[home_team]['goalsAgainst'] += away_goals
            team_stats[home_team]['matches'] += 1
            
            # Update away team stats
            if away_team not in team_stats:
                team_stats[away_team] = {'goalsFor': 0, 'goalsAgainst': 0, 'matches': 0}
            team_stats[away_team]['goalsFor'] += away_goals
            team_stats[away_team]['goalsAgainst'] += home_goals
            team_stats[away_team]['matches'] += 1
        
        # print("\n=== DEBUG: Team Statistics ===")
        # for team, stats in team_stats.items():
        #     print(f"{team}: {stats['goalsFor']}F {stats['goalsAgainst']}A "
        #           f"(GD: {stats['goalsFor'] - stats['goalsAgainst']}) in {stats['matches']} matches")
        
        # Calculate goal differences and format results
        result = {}
        for team, stats in team_stats.items():
            result[team] = {
                'goalDifference': stats['goalsFor'] - stats['goalsAgainst'],
                'goalsFor': stats['goalsFor'],
                'goalsAgainst': stats['goalsAgainst'],
                'matchesPlayed': stats['matches'],
                'goalsPerMatch': round(stats['goalsFor'] / stats['matches'], 2) if stats['matches'] > 0 else 0
            }
            
        # print("\n=== DEBUG: Final Goal Difference Stats ===")
        # for team, stats in result.items():
        #     print(f"{team}: GD={stats['goalDifference']} "
        #           f"({stats['goalsFor']}F, {stats['goalsAgainst']}A) "
        #           f"in {stats['matchesPlayed']} matches")
            
        return result
        
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Response content: {e.response.text}")
        return {}
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return {}

def get_teams_goal_difference(competition_code: str, season: int, 
                             home_team: str, away_team: str, competitionId: int = 2021) -> Tuple[Dict, Dict]:
    """
    Get goal difference for specific home and away teams.
    
    Args:
        competition_code: League code (e.g., 'PL' for Premier League)
        season: End year of the season (e.g., 2024 for 2023/24 season)
        home_team: Name of the home team
        away_team: Name of the away team
        
    Returns:
        Tuple of (home_team_stats, away_team_stats)
    """
    print(f"\n=== DEBUG: Getting goal differences for {home_team} (H) vs {away_team} (A) ===")
    
    team_stats = get_team_goal_differences(competition_code, season, competitionId)
    
    # Get stats for the requested teams
    home_stats = team_stats.get(home_team, {'goalDifference': 0, 'goalsFor': 0, 
                                          'goalsAgainst': 0, 'matchesPlayed': 0})
    away_stats = team_stats.get(away_team, {'goalDifference': 0, 'goalsFor': 0, 
                                          'goalsAgainst': 0, 'matchesPlayed': 0})
    
    print(f"\n=== DEBUG: Final Team Stats ===")
    print(f"{home_team} (Home): GD={home_stats['goalDifference']} "
          f"({home_stats['goalsFor']}F, {home_stats['goalsAgainst']}A) "
          f"in {home_stats['matchesPlayed']} matches")
    print(f"{away_team} (Away): GD={away_stats['goalDifference']} "
          f"({away_stats['goalsFor']}F, {away_stats['goalsAgainst']}A) "
          f"in {away_stats['matchesPlayed']} matches")
    
    return home_stats, away_stats

# Example usage
if __name__ == "__main__":
    # Example: Get goal differences for Premier League 2023/24 season
    competition = "PL"
    season = 2025
    
    # Example: Get stats for specific teams
    home_team = "Manchester United FC"
    away_team = "Arsenal FC"
    
    home_stats, away_stats = get_teams_goal_difference(competition, season, home_team, away_team)
    
    print(f"\n{home_team} (Home) - {season-1}/{season} Season:")
    print(f"Goal Difference: {home_stats['goalDifference']:+d}")
    print(f"Goals Scored: {home_stats['goalsFor']}")
    print(f"Goals Conceded: {home_stats['goalsAgainst']}")
    print(f"Matches Played: {home_stats['matchesPlayed']}")
    
    print(f"\n{away_team} (Away) - {season-1}/{season} Season:")
    print(f"Goal Difference: {away_stats['goalDifference']:+d}")
    print(f"Goals Scored: {away_stats['goalsFor']}")
    print(f"Goals Conceded: {away_stats['goalsAgainst']}")
    print(f"Matches Played: {away_stats['matchesPlayed']}")
