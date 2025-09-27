import requests
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import time
import traceback
from config import (
    DEFAULT_COMPETITION, CURRENT_SEASON,
    normalize_team_name, safe_request, get_competition_id, get_team_id, API_KEY, BASE_URL, HEADERS
)


def get_head_to_head(home_team: str, away_team: str, competition: str,competitionId: int) -> Dict:
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
        home_id = get_team_id(home_team, competition)
        away_id = get_team_id(away_team, competition)
        
        if not home_id or not away_id:
            print(f"Could not find IDs for one or both teams: {home_team}, {away_team}")
            return {}
            
        # Get the most recent match between these teams to get the match ID
        recent_matches_url = f"{BASE_URL}teams/{home_id}/matches"
        params = {
            'dateFrom': '2024-01-01',
            'dateTo': '2025-09-27'
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
            'limit': 50,  # fetch more to allow venue filtering down to last 7
            'competitions': competitionId  # restrict to competition
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
            
        # Sort matches by date descending to take most recent first
        matches_sorted = sorted(matches, key=lambda m: m.get('utcDate', ''), reverse=True)

        # Filter to venue-specific H2H
        home_home_results = []  # home team playing at home vs this opponent
        away_away_results = []  # away team playing away vs this opponent

        for match in matches_sorted:
            m_home = match.get('homeTeam', {})
            m_away = match.get('awayTeam', {})
            home_team_match = m_home.get('name', '')
            away_team_match = m_away.get('name', '')
            score = match.get('score', {})
            winner = score.get('winner')

            if not all([home_team_match, away_team_match, winner]):
                continue

            # Case 1: home team at home vs away team
            if home_team_match.lower() == home_team.lower() and away_team_match.lower() == away_team.lower():
                if winner == 'DRAW':
                    home_home_results.append(0.5)
                elif winner == 'HOME_TEAM':
                    home_home_results.append(1.0)
                elif winner == 'AWAY_TEAM':
                    home_home_results.append(0.0)

            # Case 2: away team away vs home team
            if away_team_match.lower() == away_team.lower() and home_team_match.lower() == home_team.lower():
                if winner == 'DRAW':
                    away_away_results.append(0.5)
                elif winner == 'AWAY_TEAM':
                    away_away_results.append(1.0)
                elif winner == 'HOME_TEAM':
                    away_away_results.append(0.0)

            # Stop if we have enough for both
            if len(home_home_results) >= 7 and len(away_away_results) >= 7:
                break

        # Trim to last 7 (already in most-recent-first order)
        home_home_results = home_home_results[:5]
        away_away_results = away_away_results[:5]

        # Calculate venue-specific averages
        if home_home_results:
            result[f"{home_team_key}_home_h2h"] = home_home_results
            result[f"{home_team_key}_home_h2h_avg"] = sum(home_home_results) / len(home_home_results)

        if away_away_results:
            result[f"{away_team_key}_away_h2h"] = away_away_results
            result[f"{away_team_key}_away_h2h_avg"] = sum(away_away_results) / len(away_away_results)
        
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
