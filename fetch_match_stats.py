import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional
import numpy as np
from config import (
    DEFAULT_COMPETITION, CURRENT_SEASON,
    normalize_team_name, safe_request, get_competition_id, get_team_id, API_KEY, BASE_URL, HEADERS
)


def get_team_matches(team_id, status='FINISHED', limit=10, competitionId: int = 2021):
    """Get recent matches for a team"""
    url = f"{BASE_URL}teams/{team_id}/matches"
    print(f"\n=== DEBUG: Fetching matches from URL: {url} ===")
    params = {
        'competitions': competitionId,
        'dateFrom': '2025-03-01',
        'dateTo': '2025-09-27',
        'status': 'FINISHED',  # ensure only completed matches are returned
        #'limit': 15
    }
    print(f"=== DEBUG: Request params: {params} ===")
    
    try:
        #use safe_request
        matches = safe_request(url, params=params)
        if not matches or not isinstance(matches, dict):
            print("=== DEBUG: No matches payload or malformed response from API ===")
            return []
        #print(f"=== DEBUG: Raw matches response: ===\n{matches}")
        return matches.get('matches', []) or []
    except requests.exceptions.RequestException as e:
        print(f"Error fetching matches: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Response content: {e.response.text}")
        return []

def analyze_matches(matches, team_id, is_home_team=True):
    """Analyze match statistics for a team, filtering by venue and limiting to last 15.
    - If is_home_team is True: only include matches where the team played at home.
    - If is_home_team is False: only include matches where the team played away.
    - Use the 15 most recent matches for the specified venue.
    """
    print(f"\n=== DEBUG: Analyzing matches for team_id: {team_id} (is_home_team: {is_home_team}) ===")
    print(f"=== DEBUG: Number of matches fetched (pre-filter): {len(matches) if matches else 0} ===")
    
    stats = {
        'goals_scored': [],
        'goals_conceded': [],
        'results': [],  # 1=Win, 0.5=Draw, 0=Loss
        'opponents': []
    }

    if not matches:
        print("=== DEBUG: No matches provided to analyze ===")
        return stats

    # Sort by utcDate descending to get most recent first (ISO strings sort lexicographically)
    matches_sorted = sorted(matches, key=lambda m: m.get('utcDate', ''), reverse=True)

    count = 0
    for match in matches_sorted:
        # Only consider finished matches
        if match.get('status') and match.get('status') != 'FINISHED':
            continue
        team_is_home = match.get('homeTeam', {}).get('id') == team_id

        # Venue filter: include only home or away depending on is_home_team
        if is_home_team and not team_is_home:
            continue
        if not is_home_team and team_is_home:
            continue

        # Get scores safely
        full_time = match.get('score', {}).get('fullTime', {})
        # Coerce None to 0 to avoid NoneType comparisons
        home_goals = full_time.get('home', 0) or 0
        away_goals = full_time.get('away', 0) or 0

        if team_is_home:
            goals_for = home_goals
            goals_against = away_goals
        else:
            goals_for = away_goals
            goals_against = home_goals

        # Calculate result (1=Win, 0.5=Draw, 0=Loss)
        if goals_for > goals_against:
            result = 1.0
        elif goals_for == goals_against:
            result = 0.5
        else:
            result = 0.0

        stats['goals_scored'].append(goals_for)
        stats['goals_conceded'].append(goals_against)
        stats['results'].append(result)
        opponent_name = (
            match.get('awayTeam', {}).get('name') if team_is_home else match.get('homeTeam', {}).get('name')
        )
        stats['opponents'].append(opponent_name)

        count += 1
        if count >= 15:
            break

    print(f"\n=== DEBUG: Final stats for team_id {team_id} (venue-filtered, last {count}) ===")
    print(f"Goals scored: {stats['goals_scored']}")
    print(f"Goals conceded: {stats['goals_conceded']}")
    print(f"Results: {stats['results']}")
    print(f"Opponents: {stats['opponents']}")
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
  • Opponents: {home_stats.get('opponents', [])}
  • Goals Scored: {home_stats['goals_scored']}
  • Goals Conceded: {home_stats['goals_conceded']}
  • Results: {['Win' if x == 1 else 'Draw' if x == 0.5 else 'Loss' for x in home_stats['results']]}
  • Avg. Goals Scored: {home_avg_scored:.1f}
  • Avg. Goals Conceded: {home_avg_conceded:.1f}

{away_team} (Last 5 away matches):
  • Opponents: {away_stats.get('opponents', [])}
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
