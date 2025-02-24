#!/usr/bin/env python3
"""
monty.py

Runs a Monte Carlo analysis using a tennis point-by-point betting system.
By default, it performs 10,000 simulations, each randomly sampling 50,000 games
from 'combined.csv'. It then summarizes the results and produces a
'monte_carlo_results.png' report.

Usage:
    python monty.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from datetime import datetime
import random
from tqdm import tqdm  # progress bar library

# --------------------------------------------------------------------------------
# Global Configuration
# --------------------------------------------------------------------------------
INITIAL_BANKROLL = 1500.0
# Initial stake is 1/1000th of the bankroll
INITIAL_STAKE = INITIAL_BANKROLL / 1000  
SERVER_ODDS = -250
RECEIVER_ODDS = 135
MAX_BET_ABSOLUTE = 40000.0

# Number of Monte Carlo simulations and games per simulation
MONTE_CARLO_SIMULATIONS = 1000
GAMES_PER_SIMULATION = 5000

# --------------------------------------------------------------------------------
# Core Classes
# --------------------------------------------------------------------------------
class BetRecord:
    """Tracks complete information about a single bet including context and result."""
    def __init__(self, bet_number, stake, pick, game_score, game_sequence, 
                 bankroll, won, strategy_type='Progression', profit=0):
        self.bet_number = bet_number
        self.stake = stake
        self.pick = pick
        self.game_score = game_score
        self.game_sequence = game_sequence
        self.bankroll = bankroll
        self.won = won
        self.strategy_type = strategy_type
        self.profit = profit
        self.timestamp = datetime.now()

class GameState:
    """Manages the state and scoring of a tennis game."""
    def __init__(self):
        self.server_points = 0
        self.receiver_points = 0
        self.points_sequence = []
        
    def add_point(self, point):
        """Records a point and updates score."""
        self.points_sequence.append(point)
        if point in ['S', 'A']:
            self.server_points += 1
        elif point in ['R', 'D']:
            self.receiver_points += 1
            
    def is_score_tied(self):
        """Checks if game score is tied."""
        return self.server_points == self.receiver_points
        
    def get_advantage_pick(self):
        """Determines which player to bet on based on score."""
        if self.server_points > self.receiver_points:
            return 'S'
        elif self.receiver_points > self.server_points:
            return 'R'
        return 'S'  # Default to server when tied
        
    def get_score_summary(self):
        """Provides formatted game score."""
        return f"Server {self.server_points} - {self.receiver_points} Receiver"

class SessionAnalytics:
    """Manages comprehensive session statistics and analytics."""
    def __init__(self):
        self.bets = []
        self.max_bet = None
        self.bankroll_history = []
        self.bet_sizes = []
        self.strategy_outcomes = {
            'Progression': {'wins': 0, 'total': 0, 'profit': 0},
            'Score-Based': {'wins': 0, 'total': 0, 'profit': 0}
        }
        
    def record_bet(self, bet_record):
        """Records bet details and updates analytics."""
        self.bets.append(bet_record)
        self.bankroll_history.append(bet_record.bankroll)
        self.bet_sizes.append(bet_record.stake)
        
        # Update strategy statistics
        strategy = bet_record.strategy_type
        self.strategy_outcomes[strategy]['total'] += 1
        self.strategy_outcomes[strategy]['profit'] += bet_record.profit
        if bet_record.won:
            self.strategy_outcomes[strategy]['wins'] += 1
        
        # Track maximum bet
        if self.max_bet is None or bet_record.stake > self.max_bet.stake:
            self.max_bet = bet_record
            
    def verify_statistics(self):
        """Verifies consistency of tracking statistics."""
        total_profit = sum(bet.profit for bet in self.bets)
        total_stakes = sum(bet.stake for bet in self.bets)
        strategy_total = sum(s['total'] for s in self.strategy_outcomes.values())
        
        verification = {
            'total_bets_match': len(self.bets) == strategy_total,
            'bankroll_consistent': len(self.bankroll_history) == len(self.bets),
            'total_profit': total_profit,
            'total_stakes': total_stakes
        }
        return verification

# --------------------------------------------------------------------------------
# Utility Functions
# --------------------------------------------------------------------------------
def format_currency(amount):
    """
    Formats currency with color-coding to improve readability of financial outcomes.
    Green indicates positive amounts, red indicates negative amounts.
    """
    if amount > 0:
        return f"\033[92m${amount:,.2f}\033[0m"  # Green for profits
    elif amount < 0:
        return f"\033[91m${amount:,.2f}\033[0m"  # Red for losses
    return f"${amount:,.2f}"  # Normal for zero

def calculate_next_bet(current_bet, next_pick, previous_pick, bankroll):
    """
    Determines the next bet size following the progression rules:
    - Triple the previous bet when betting on Server
    - Double the previous bet when betting on Receiver
    - Never exceed half of current bankroll or maximum bet limit
    """
    if next_pick == 'S':
        next_bet = current_bet * 3  # Triple for Server bets
    else:  # next_pick == 'R'
        next_bet = current_bet * 2  # Double for Receiver bets
        
    max_allowed = min(bankroll / 2, MAX_BET_ABSOLUTE)
    return min(next_bet, max_allowed)

def is_tiebreak(game_str):
    """
    Identifies tiebreak games by checking for the presence of service changes (/).
    These games are skipped in our betting strategy.
    """
    return '/' in game_str

def evaluate_point(point, pick):
    """
    Determines if a bet wins based on the point outcome.
    Server bets win on 'S' (service winner) or 'A' (ace).
    Receiver bets win on 'R' (return winner) or 'D' (double fault).
    """
    if pick == 'S':
        return point in ['S', 'A']
    else:
        return point in ['R', 'D']

def process_game_points(game_str):
    """
    Extracts valid points from a game string.
    Filters out tiebreaks and ensures only valid point markers are included.
    """
    if not game_str or is_tiebreak(game_str):
        return []
    return [p for p in game_str if p in ['S', 'A', 'R', 'D']]

def calculate_profit(stake, odds):
    """
    Calculates potential profit based on stake and American odds.
    """
    if odds > 0:
        return stake * (odds / 100)
    else:
        return stake * (100 / abs(odds))

def verify_bet_calculation(stake, pick, won_bet):
    """
    Verifies bet calculations to ensure accuracy of financial tracking.
    Returns a tuple of (profit, verification_data).
    """
    odds = SERVER_ODDS if pick == 'S' else RECEIVER_ODDS
    if won_bet:
        profit = calculate_profit(stake, odds)
        verification = {
            'stake': stake,
            'odds': odds,
            'result': 'win',
            'profit': profit
        }
    else:
        profit = -stake
        verification = {
            'stake': stake,
            'odds': odds,
            'result': 'loss',
            'profit': profit
        }
    return profit, verification

# --------------------------------------------------------------------------------
# Monte Carlo: Turn Off Interactive Mode
# --------------------------------------------------------------------------------
AUTO_MODE = True  # Disable interactive prompts for Monte Carlo runs

# --------------------------------------------------------------------------------
# Main Simulation Logic (non-interactive)
# --------------------------------------------------------------------------------
def simulate_interactive_betting(df, initial_stake=INITIAL_STAKE):
    """
    Simulates betting on tennis points with detailed tracking and verification.
    In Monte Carlo mode (AUTO_MODE = True), interactive prompts are skipped.
    Returns final bankroll, peak bankroll, max drawdown, session_stats, and analytics.
    """
    bankroll = INITIAL_BANKROLL
    peak = bankroll
    max_drawdown = 0
    bet_number = 0
    analytics = SessionAnalytics()
    
    session_stats = {
        'matches_processed': 0,
        'total_games': 0,
        'actual_games': 0,
        'total_bets': 0,
        'wins': 0,
        'total_profit': 0,
        'total_stakes': 0,
        'progression_wins': 0,
        'score_based_wins': 0,
        'tiebreaks_skipped': 0
    }
    
    for _, match in df.iterrows():
        if bankroll <= 0:
            break
            
        session_stats['matches_processed'] += 1
        match_info = {
            'tournament': match.get('tny_name', 'Unknown'),
            'date': match.get('date', 'Unknown'),
            'server1': match.get('server1', 'Unknown'),
            'server2': match.get('server2', 'Unknown')
        }
        
        pbp_data = str(match.get('pbp', ''))
        sets = [s.strip() for s in pbp_data.split('.') if s.strip()]
        for set_str in sets:
            games = [g.strip() for g in set_str.split(';') if g.strip()]
            session_stats['actual_games'] += len(games)
            
            for game_str in games:
                if is_tiebreak(game_str):
                    session_stats['tiebreaks_skipped'] += 1
                    continue
                    
                points = process_game_points(game_str)
                if not points:
                    continue
                
                session_stats['total_games'] += 1
                
                game_state = GameState()
                current_stake = initial_stake
                previous_pick = None
                points_played = 0
                using_score_strategy = False
                strategy_picks = ['S', 'S', 'S', 'R', 'R', 'R']
                current_sequence = 0
                
                while points_played < len(points):
                    current_point = points[points_played]
                    game_state.add_point(current_point)
                    
                    if not using_score_strategy:
                        if current_sequence >= len(strategy_picks):
                            using_score_strategy = True
                        else:
                            pick = strategy_picks[current_sequence]
                    
                    if using_score_strategy:
                        if game_state.is_score_tied():
                            pick = 'S'
                        else:
                            pick = game_state.get_advantage_pick()
                    
                    if previous_pick is not None:
                        current_stake = calculate_next_bet(current_stake, pick, previous_pick, bankroll)
                    
                    stake = min(current_stake, bankroll / 2, MAX_BET_ABSOLUTE)
                    bet_number += 1
                    strategy_type = "Progression" if not using_score_strategy else "Score-Based"
                    
                    won_bet = evaluate_point(current_point, pick)
                    
                    session_stats['total_bets'] += 1
                    session_stats['total_stakes'] += stake
                    
                    if won_bet:
                        profit = calculate_profit(stake, SERVER_ODDS if pick == 'S' else RECEIVER_ODDS)
                        bankroll += profit
                        session_stats['wins'] += 1
                        session_stats['total_profit'] += profit
                        if using_score_strategy:
                            session_stats['score_based_wins'] += 1
                        else:
                            session_stats['progression_wins'] += 1
                        
                        bet_record = BetRecord(
                            bet_number=bet_number,
                            stake=stake,
                            pick=pick,
                            game_score=game_state.get_score_summary(),
                            game_sequence=game_str,
                            bankroll=bankroll,
                            won=True,
                            strategy_type=strategy_type,
                            profit=profit
                        )
                        analytics.record_bet(bet_record)
                        
                        peak = max(peak, bankroll)
                        max_drawdown = max(max_drawdown, peak - bankroll)
                        break
                    else:
                        profit = -stake
                        bankroll -= stake
                        session_stats['total_profit'] += profit
                        
                        bet_record = BetRecord(
                            bet_number=bet_number,
                            stake=stake,
                            pick=pick,
                            game_score=game_state.get_score_summary(),
                            game_sequence=game_str,
                            bankroll=bankroll,
                            won=False,
                            strategy_type=strategy_type,
                            profit=profit
                        )
                        analytics.record_bet(bet_record)
                        
                        if not using_score_strategy:
                            current_sequence += 1
                        
                        peak = max(peak, bankroll)
                        max_drawdown = max(max_drawdown, peak - bankroll)
                    
                    previous_pick = pick
                    points_played += 1
                    
                    if bankroll <= 0:
                        break
                if bankroll <= 0:
                    break
        if bankroll <= 0:
            break
    
    return bankroll, peak, max_drawdown, session_stats, analytics

def generate_detailed_analytics_report(analytics, session_stats, output_file="betting_analytics.png"):
    """
    Generates comprehensive visual analytics of betting performance including
    bankroll evolution, bet distribution, and strategy effectiveness.
    """
    try:
        plt.style.use('default')
        fig = plt.figure(figsize=(20, 24))
        
        # 1. Bankroll Evolution Over Time
        plt.subplot(5, 1, 1)
        plt.plot(analytics.bankroll_history, label='Bankroll', color='blue', linewidth=2)
        plt.axhline(y=INITIAL_BANKROLL, color='red', linestyle='--', label='Initial Bankroll')
        plt.title('Bankroll Evolution Over Time', fontsize=14, pad=20)
        plt.xlabel('Bet Number')
        plt.ylabel('Bankroll ($)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # 2. Bet Size Distribution
        plt.subplot(5, 1, 2)
        if len(analytics.bet_sizes) > 0:
            bins = np.linspace(min(analytics.bet_sizes), max(analytics.bet_sizes), 51)
            plt.hist(analytics.bet_sizes, bins=bins, color='green', alpha=0.7)
            mean_bet = np.mean(analytics.bet_sizes)
            plt.axvline(mean_bet, color='red', linestyle='--', label=f'Mean Bet: {format_currency(mean_bet)}')
        plt.title('Bet Size Distribution', fontsize=14, pad=20)
        plt.xlabel('Bet Size ($)')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # 3. Strategy Performance Comparison
        plt.subplot(5, 1, 3)
        strategies = list(analytics.strategy_outcomes.keys())
        x = np.arange(len(strategies))
        width = 0.35
        
        win_rates = [
            (analytics.strategy_outcomes[s]['wins'] / analytics.strategy_outcomes[s]['total'] * 100)
            if analytics.strategy_outcomes[s]['total'] > 0 else 0
            for s in strategies
        ]
        
        rois = []
        for s in strategies:
            strategy_stakes = sum(bet.stake for bet in analytics.bets if bet.strategy_type == s)
            strategy_profit = analytics.strategy_outcomes[s]['profit']
            if strategy_stakes > 0:
                rois.append((strategy_profit / strategy_stakes) * 100)
            else:
                rois.append(0)
        
        plt.bar(x - width/2, win_rates, width, label='Win Rate %', color='blue')
        plt.bar(x + width/2, rois, width, label='ROI %', color='green')
        
        plt.title('Strategy Performance Metrics', fontsize=14, pad=20)
        plt.xticks(x, strategies)
        plt.ylabel('Percentage')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # 4. Win Rate by Bet Size Range
        plt.subplot(5, 1, 4)
        if len(analytics.bet_sizes) > 0:
            bet_data = pd.DataFrame({
                'stake': [bet.stake for bet in analytics.bets],
                'won': [bet.won for bet in analytics.bets]
            })
            bet_data['size_group'] = pd.qcut(bet_data['stake'], 10, labels=False, duplicates='drop')
            win_rates_by_group = bet_data.groupby('size_group')['won'].mean() * 100
            
            plt.bar(range(len(win_rates_by_group)), win_rates_by_group, color='purple', alpha=0.7)
        plt.title('Win Rate by Bet Size Range', fontsize=14, pad=20)
        plt.xlabel('Bet Size Decile (0=Smallest, 9=Largest)')
        plt.ylabel('Win Rate (%)')
        plt.grid(True, alpha=0.3)
        
        # 5. Maximum Bet Details and Key Statistics
        plt.subplot(5, 1, 5)
        plt.axis('off')
        
        stats_text = (
            f"Session Overview:\n\n"
            f"Initial Stake: {format_currency(INITIAL_STAKE)}\n"
            f"Total Games Processed: {session_stats['total_games']:,}\n"
            f"Total Bets Placed: {session_stats['total_bets']:,}\n"
            f"Overall Win Rate: {(session_stats['wins']/session_stats['total_bets']*100):.1f}%\n"
            f"Total Amount Staked: {format_currency(session_stats['total_stakes'])}\n"
            f"Net Profit/Loss: {format_currency(session_stats['total_profit'])}\n"
            f"ROI: {(session_stats['total_profit']/session_stats['total_stakes']*100):.1f}%\n"
            f"Maximum Drawdown: {format_currency(max_drawdowns_mean := 'N/A' if session_stats['total_profit'] == 0 else session_stats['total_stakes'] - session_stats['total_profit'])}\n\n"
        )
        
        if analytics.max_bet is not None:
            stats_text += (
                f"Maximum Bet Details:\n"
                f"Amount: {format_currency(analytics.max_bet.stake)}\n"
                f"Bet Number: {analytics.max_bet.bet_number}\n"
                f"Strategy: {analytics.max_bet.strategy_type}\n"
                f"Game Score: {analytics.max_bet.game_score}\n"
                f"Point Sequence: {analytics.max_bet.game_sequence}\n"
                f"Result: {'Won' if analytics.max_bet.won else 'Lost'}\n"
                f"Timestamp: {analytics.max_bet.timestamp}\n"
            )
        
        plt.text(0.1, 0.5, stats_text, fontsize=12, verticalalignment='center')
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Detailed analytics report saved as '{output_file}'")
    except Exception as e:
        print(f"\nWarning: Could not generate visual analytics due to: {str(e)}")
        print("Continuing with text-based results...")

# --------------------------------------------------------------------------------
# Monte Carlo Logic
# --------------------------------------------------------------------------------
def run_one_monte_carlo(df, n_games=GAMES_PER_SIMULATION):
    """
    Runs a single Monte Carlo simulation by sampling 'n_games' from 'df' (with replacement),
    then running the betting simulation. Returns final bankroll, total profit, ROI, and max drawdown.
    """
    sample_df = df.sample(n=n_games, replace=True)
    final_bankroll, peak, max_drawdown, session_stats, analytics = simulate_interactive_betting(sample_df, initial_stake=INITIAL_STAKE)
    
    total_stakes = session_stats['total_stakes']
    total_profit = session_stats['total_profit']
    roi = (total_profit / total_stakes * 100) if total_stakes > 0 else 0
    
    return final_bankroll, total_profit, roi, max_drawdown

def main():
    """
    Main function for running a Monte Carlo analysis of the betting system.
    """
    print("Loading combined.csv for Monte Carlo analysis...")
    df = pd.read_csv('combined.csv')
    # Specify the date format to match expected data (e.g., "28-Jul-11")
    df['date'] = pd.to_datetime(df['date'], format="%d-%b-%y", dayfirst=True, errors='coerce')
    
    print(f"\nStarting Monte Carlo with {MONTE_CARLO_SIMULATIONS} simulations "
          f"of {GAMES_PER_SIMULATION} games each...\n")
    
    final_bankrolls = []
    total_profits = []
    rois = []
    max_drawdowns = []
    
    for sim_index in tqdm(range(1, MONTE_CARLO_SIMULATIONS + 1), desc="Simulations"):
        fb, tp, roi, md = run_one_monte_carlo(df, n_games=GAMES_PER_SIMULATION)
        final_bankrolls.append(fb)
        total_profits.append(tp)
        rois.append(roi)
        max_drawdowns.append(md)
    
    final_bankrolls = np.array(final_bankrolls)
    total_profits = np.array(total_profits)
    rois = np.array(rois)
    max_drawdowns = np.array(max_drawdowns)
    
    print("\n=== Monte Carlo Results Summary ===")
    print(f"Simulations: {MONTE_CARLO_SIMULATIONS}")
    print(f"Games per Simulation: {GAMES_PER_SIMULATION}")
    print(f"Initial Stake: {format_currency(INITIAL_STAKE)}")
    
    print(f"\nFinal Bankroll:")
    print(f"  Mean:  {final_bankrolls.mean():.2f}")
    print(f"  Median: {np.median(final_bankrolls):.2f}")
    print(f"  Std:   {final_bankrolls.std():.2f}")
    print(f"  Min:   {final_bankrolls.min():.2f}")
    print(f"  Max:   {final_bankrolls.max():.2f}")
    
    print(f"\nTotal Profit:")
    print(f"  Mean:  {total_profits.mean():.2f}")
    print(f"  Median: {np.median(total_profits):.2f}")
    print(f"  Std:   {total_profits.std():.2f}")
    print(f"  Min:   {total_profits.min():.2f}")
    print(f"  Max:   {total_profits.max():.2f}")
    
    print(f"\nROI (%):")
    print(f"  Mean:  {rois.mean():.2f}%")
    print(f"  Median: {np.median(rois):.2f}%")
    print(f"  Std:   {rois.std():.2f}%")
    print(f"  Min:   {rois.min():.2f}%")
    print(f"  Max:   {rois.max():.2f}%")
    
    print(f"\nMaximum Drawdown:")
    print(f"  Mean:  {max_drawdowns.mean():.2f}")
    print(f"  Median: {np.median(max_drawdowns):.2f}")
    print(f"  Std:   {max_drawdowns.std():.2f}")
    print(f"  Min:   {max_drawdowns.min():.2f}")
    print(f"  Max:   {max_drawdowns.max():.2f}")
    
    try:
        plt.style.use('default')
        fig = plt.figure(figsize=(18, 6))
        
        # Histogram of Final Bankroll
        plt.subplot(1, 3, 1)
        plt.hist(final_bankrolls, bins=50, color='blue', alpha=0.7)
        plt.axvline(final_bankrolls.mean(), color='red', linestyle='--', 
                    label=f"Mean = {final_bankrolls.mean():.2f}")
        plt.title('Distribution of Final Bankroll')
        plt.xlabel('Final Bankroll')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Histogram of ROI
        plt.subplot(1, 3, 2)
        plt.hist(rois, bins=50, color='green', alpha=0.7)
        plt.axvline(rois.mean(), color='red', linestyle='--', 
                    label=f"Mean = {rois.mean():.2f}%")
        plt.title('Distribution of ROI (%)')
        plt.xlabel('ROI (%)')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Histogram of Maximum Drawdown
        plt.subplot(1, 3, 3)
        plt.hist(max_drawdowns, bins=50, color='purple', alpha=0.7)
        plt.axvline(max_drawdowns.mean(), color='red', linestyle='--', 
                    label=f"Mean = {max_drawdowns.mean():.2f}")
        plt.title('Distribution of Maximum Drawdown')
        plt.xlabel('Maximum Drawdown')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig("monte_carlo_results.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print("\nMonte Carlo distribution report saved as 'monte_carlo_results.png'")
    except Exception as e:
        print(f"\nWarning: Could not generate Monte Carlo report due to: {str(e)}")
    
    print("\nMonte Carlo Analysis Complete!")

if __name__ == "__main__":
    main()
