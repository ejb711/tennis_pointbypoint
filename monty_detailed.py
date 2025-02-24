#!/usr/bin/env python3
"""
monty_detailed.py

Runs a Monte Carlo analysis using a tennis point-by-point betting system.
This version additionally computes the ROI of each bet in the progression,
saves a CSV breakdown of all bets from the final simulation, and supports
interactive step-by-step user input (or auto-run) for each bet.

Key correction: if the game is tied (server_points == receiver_points),
the bet should favor the server when using the score-based strategy.
Otherwise, bet on the player with the advantage.

By default (in this version), it performs 200 simulations,
each randomly sampling 500 games from 'combined.csv'. It then summarizes
the results and produces a 'monte_carlo_results.png' report, plus a
'bet_breakdown.csv' for the final run.
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
INITIAL_BANKROLL = 2200.0
# Initial stake for each new game
INITIAL_STAKE = 1  
SERVER_ODDS = -265
RECEIVER_ODDS = 125
MAX_BET_ABSOLUTE = 40000

# Number of Monte Carlo simulations and games per simulation
MONTE_CARLO_SIMULATIONS = 200
GAMES_PER_SIMULATION = 1000

# If AUTO_MODE is False, we prompt the user for each bet.
AUTO_MODE = False

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
        # Compute ROI for this individual bet:
        # (profit / stake * 100). For a losing bet, ROI can be -100%.
        self.roi = (profit / stake * 100) if stake != 0 else 0
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
        # If still tied, default to server
        return 'S'
        
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
    Remove \033[...] codes if you prefer no ANSI color (avoiding glyph warnings).
    """
    if amount > 0:
        return f"\033[92m${amount:,.2f}\033[0m"
    elif amount < 0:
        return f"\033[91m${amount:,.2f}\033[0m"
    return f"${amount:,.2f}"

def calculate_next_bet(current_bet, next_pick, previous_pick, bankroll):
    """
    Determines the next bet size following the progression rules:
    - Triple the previous bet when betting on Server
    - Double the previous bet when betting on Receiver
    - Never exceed half of current bankroll or maximum bet limit
    """
    if next_pick == 'S':
        next_bet = current_bet * 3
    else:  # next_pick == 'R'
        next_bet = current_bet * 2
        
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

# --------------------------------------------------------------------------------
# Interactive Helpers
# --------------------------------------------------------------------------------
def check_user_input():
    """
    Handles user interaction during simulation.
    If AUTO_MODE is False, we prompt:
      - Enter to proceed
      - 'q' to quit
      - 'auto' to switch to automatic mode
    """
    global AUTO_MODE
    if AUTO_MODE:
        return False  # no prompt if in auto mode
    
    user_input = input("\nPress Enter to continue, 'q' to quit, or 'auto' for automatic: ")
    if user_input.lower() == 'q':
        return True
    elif user_input.lower() == 'auto':
        print("\nSwitching to automatic mode...")
        AUTO_MODE = True
    return False

def display_bet_info(bet_number, match_info, game_str, current_point, points_played, 
                     stake, pick, bankroll, game_state, strategy_type):
    """
    Displays comprehensive information about the current bet situation.
    Skipped if AUTO_MODE is True.
    """
    if AUTO_MODE:
        return
    
    print("\n" + "="*80)
    print(f"Bet #{bet_number}")
    print("="*80)
    
    print(f"\nMatch Information:")
    print(f"Tournament: {match_info.get('tournament','Unknown')}")
    print(f"Date: {match_info.get('date','Unknown')}")
    print(f"Server: {match_info.get('server1','Unknown')} vs {match_info.get('server2','Unknown')}")
    
    print(f"\nGame Information:")
    print(f"Full points sequence: {game_str}")
    print(f"Current score: {game_state.get_score_summary()}")
    print(f"Current point: {current_point} (Point {points_played} of this game)")
    print(f"Points played so far: {' '.join(game_state.points_sequence)}")
    
    print(f"\nBetting Information:")
    print(f"Strategy: {strategy_type}")
    print(f"Stake Amount: {format_currency(stake)}")
    print(f"Betting On: {'Server' if pick == 'S' else 'Receiver'}")
    print(f"Odds: {SERVER_ODDS if pick == 'S' else RECEIVER_ODDS}")
    
    print(f"\nBankroll Status:")
    print(f"Current Bankroll: {format_currency(bankroll)}")
    print(f"Maximum allowed bet: {format_currency(min(bankroll / 2, MAX_BET_ABSOLUTE))}")
    
    input("\nPress Enter to see the result...")

def display_bet_result(won_bet, profit, bankroll, peak, max_drawdown, move_to_next_game):
    """
    Shows the outcome of a bet and updates key statistics.
    Skipped if AUTO_MODE is True.
    """
    if AUTO_MODE:
        return
    
    print("\nBet Result:")
    print("="*40)
    if won_bet:
        print(f"\033[92mWIN!\033[0m")
        print(f"Profit: {format_currency(profit)}")
    else:
        print(f"\033[91mLOSS\033[0m")
        print(f"Loss: {format_currency(-profit)}")
    
    print(f"\nUpdated Statistics:")
    print(f"Current Bankroll: {format_currency(bankroll)}")
    print(f"Peak Bankroll: {format_currency(peak)}")
    print(f"Maximum Drawdown: {format_currency(max_drawdown)}")
    
    if move_to_next_game:
        print("\n\033[93mMoving to next game\033[0m")

# --------------------------------------------------------------------------------
# Main Simulation Logic (non-interactive, but with optional user prompts)
# --------------------------------------------------------------------------------
def simulate_interactive_betting(df, initial_stake=INITIAL_STAKE):
    """
    Simulates betting on tennis points with detailed tracking and verification.
    If AUTO_MODE is False, the user is prompted for each bet. Otherwise it runs automatically.
    
    Returns final bankroll, peak bankroll, max_drawdown, session_stats, analytics.
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
    
    # Process each match
    for _, match in df.iterrows():
        if bankroll <= 0:
            break
        session_stats['matches_processed'] += 1
        
        match_info = {
            'tournament': match.get('tny_name','Unknown'),
            'date': match.get('date','Unknown'),
            'server1': match.get('server1','Unknown'),
            'server2': match.get('server2','Unknown')
        }
        
        pbp_data = str(match.get('pbp',''))
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
                
                # Initialize a new game
                game_state = GameState()
                current_stake = initial_stake
                previous_pick = None
                points_played = 0
                using_score_strategy = False
                strategy_picks = ['S','S','S','R','R','R']
                current_sequence = 0
                
                while points_played < len(points):
                    current_point = points[points_played]
                    game_state.add_point(current_point)
                    
                    # Decide pick
                    if not using_score_strategy:
                        if current_sequence >= len(strategy_picks):
                            using_score_strategy = True
                        else:
                            pick = strategy_picks[current_sequence]
                    
                    # If we're in score-based strategy
                    if using_score_strategy:
                        if game_state.is_score_tied():
                            # If tied, always pick server
                            pick = 'S'
                        else:
                            # Otherwise pick whoever has advantage
                            pick = game_state.get_advantage_pick()
                    
                    if previous_pick is not None:
                        current_stake = calculate_next_bet(current_stake, pick, previous_pick, bankroll)
                    
                    stake = min(current_stake, bankroll/2, MAX_BET_ABSOLUTE)
                    bet_number += 1
                    strategy_type = "Progression" if not using_score_strategy else "Score-Based"
                    
                    # Display info (if not AUTO_MODE)
                    display_bet_info(
                        bet_number, match_info, game_str, current_point,
                        points_played+1, stake, pick, bankroll,
                        game_state, strategy_type
                    )
                    
                    won_bet = evaluate_point(current_point, pick)
                    
                    # Update stats
                    session_stats['total_bets'] += 1
                    session_stats['total_stakes'] += stake
                    
                    if won_bet:
                        profit = calculate_profit(stake, SERVER_ODDS if pick=='S' else RECEIVER_ODDS)
                        bankroll += profit
                        session_stats['wins'] += 1
                        session_stats['total_profit'] += profit
                        
                        if using_score_strategy:
                            session_stats['score_based_wins'] += 1
                        else:
                            session_stats['progression_wins'] += 1
                        
                        # Record winning bet
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
                        
                        # Show result
                        display_bet_result(True, profit, bankroll, peak, max_drawdown, True)
                        break
                    else:
                        profit = -stake
                        bankroll += profit
                        session_stats['total_profit'] += profit
                        
                        # Record losing bet
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
                        
                        # Show result
                        display_bet_result(False, profit, bankroll, peak, max_drawdown, points_played>=len(points)-1)
                    
                    previous_pick = pick
                    points_played += 1
                    
                    # If user typed 'q' to quit or 'auto' to go auto, handle it
                    if check_user_input():
                        # user typed 'q'
                        return bankroll, peak, max_drawdown, session_stats, analytics
        
        if bankroll <= 0:
            break
    
    return bankroll, peak, max_drawdown, session_stats, analytics

# --------------------------------------------------------------------------------
# Monte Carlo Logic (runs multiple simulations)
# --------------------------------------------------------------------------------
def run_one_monte_carlo(df, n_games=GAMES_PER_SIMULATION):
    """
    Runs a single Monte Carlo simulation by sampling 'n_games' from 'df' (with replacement),
    then running the betting simulation. Returns final bankroll, total profit, ROI, max drawdown,
    and the analytics object (so we can save a breakdown).
    """
    sample_df = df.sample(n=n_games, replace=True)
    final_bankroll, peak, max_drawdown, session_stats, analytics = simulate_interactive_betting(
        sample_df, initial_stake=INITIAL_STAKE
    )
    
    total_stakes = session_stats['total_stakes']
    total_profit = session_stats['total_profit']
    roi = (total_profit / total_stakes * 100) if total_stakes > 0 else 0
    
    return final_bankroll, total_profit, roi, max_drawdown, session_stats, analytics

def generate_bet_breakdown_report(analytics, output_file="bet_breakdown.csv"):
    """
    Saves a CSV breakdown of every bet placed in the final simulation.
    """
    bet_rows = []
    for bet in analytics.bets:
        bet_rows.append({
            "BetNumber": bet.bet_number,
            "Stake": bet.stake,
            "Profit": bet.profit,
            "ROI(%)": f"{bet.roi:.2f}",
            "Pick": bet.pick,
            "Won": bet.won,
            "Strategy": bet.strategy_type,
            "GameScore": bet.game_score,
            "GameSequence": bet.game_sequence,
            "BankrollAfterBet": bet.bankroll,
            "Timestamp": bet.timestamp
        })
    df_bets = pd.DataFrame(bet_rows)
    df_bets.to_csv(output_file, index=False)
    print(f"\nDetailed bet breakdown saved as '{output_file}'")

def generate_detailed_analytics_report(analytics, session_stats, output_file="monte_carlo_results.png"):
    """
    Generates comprehensive visual analytics of betting performance including
    bankroll evolution, bet distribution, strategy effectiveness, and a
    distribution of bet ROI.
    """
    import matplotlib.pyplot as plt
    try:
        plt.style.use('default')
        fig = plt.figure(figsize=(20, 30))
        
        # 1. Bankroll Evolution
        ax1 = plt.subplot(6, 1, 1)
        ax1.plot(analytics.bankroll_history, label='Bankroll', color='blue', linewidth=2)
        ax1.axhline(y=INITIAL_BANKROLL, color='red', linestyle='--', label='Initial Bankroll')
        ax1.set_title('Bankroll Evolution Over Time', fontsize=14, pad=20)
        ax1.set_xlabel('Bet Number')
        ax1.set_ylabel('Bankroll ($)')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # 2. Bet Size Distribution
        ax2 = plt.subplot(6, 1, 2)
        if len(analytics.bet_sizes) > 0:
            bins = np.linspace(min(analytics.bet_sizes), max(analytics.bet_sizes), 51)
            ax2.hist(analytics.bet_sizes, bins=bins, color='green', alpha=0.7)
            mean_bet = np.mean(analytics.bet_sizes)
            ax2.axvline(mean_bet, color='red', linestyle='--', label=f'Mean Bet: {format_currency(mean_bet)}')
        ax2.set_title('Bet Size Distribution', fontsize=14, pad=20)
        ax2.set_xlabel('Bet Size ($)')
        ax2.set_ylabel('Frequency')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # 3. Strategy Performance Comparison
        ax3 = plt.subplot(6, 1, 3)
        strategies = list(analytics.strategy_outcomes.keys())
        x = np.arange(len(strategies))
        width = 0.35
        
        win_rates = []
        rois = []
        for s in strategies:
            total_bets = analytics.strategy_outcomes[s]['total']
            wins = analytics.strategy_outcomes[s]['wins']
            profit = analytics.strategy_outcomes[s]['profit']
            
            wr = (wins / total_bets * 100) if total_bets > 0 else 0
            strategy_stakes = sum(bet.stake for bet in analytics.bets if bet.strategy_type == s)
            roi_val = (profit / strategy_stakes * 100) if strategy_stakes > 0 else 0
            
            win_rates.append(wr)
            rois.append(roi_val)
        
        ax3.bar(x - width/2, win_rates, width, label='Win Rate %', color='blue')
        ax3.bar(x + width/2, rois, width, label='ROI %', color='green')
        ax3.set_title('Strategy Performance Metrics', fontsize=14, pad=20)
        ax3.set_xticks(x)
        ax3.set_xticklabels(strategies)
        ax3.set_ylabel('Percentage')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # 4. Win Rate by Bet Size Range
        ax4 = plt.subplot(6, 1, 4)
        if len(analytics.bet_sizes) > 0:
            bet_data = pd.DataFrame({
                'stake': [bet.stake for bet in analytics.bets],
                'won': [bet.won for bet in analytics.bets]
            })
            bet_data['size_group'] = pd.qcut(bet_data['stake'], 10, labels=False, duplicates='drop')
            win_rates_by_group = bet_data.groupby('size_group')['won'].mean() * 100
            ax4.bar(range(len(win_rates_by_group)), win_rates_by_group, color='purple', alpha=0.7)
        ax4.set_title('Win Rate by Bet Size Range', fontsize=14, pad=20)
        ax4.set_xlabel('Bet Size Decile (0=Smallest, 9=Largest)')
        ax4.set_ylabel('Win Rate (%)')
        ax4.grid(True, alpha=0.3)
        
        # 5. Key Statistics Text
        ax5 = plt.subplot(6, 1, 5)
        ax5.axis('off')
        
        stats_text = (
            f"Session Overview:\n\n"
            f"Initial Stake: {format_currency(INITIAL_STAKE)}\n"
            f"Total Games Processed: {session_stats['total_games']:,}\n"
            f"Total Bets Placed: {session_stats['total_bets']:,}\n"
            f"Overall Win Rate: {(session_stats['wins']/session_stats['total_bets']*100):.1f}%\n"
            f"Total Amount Staked: {format_currency(session_stats['total_stakes'])}\n"
            f"Net Profit/Loss: {format_currency(session_stats['total_profit'])}\n"
            f"ROI: {(session_stats['total_profit']/session_stats['total_stakes']*100):.1f}%\n\n"
        )
        
        if len(analytics.bets) > 0 and analytics.max_bet is not None:
            stats_text += (
                f"Maximum Bet Details:\n"
                f"Amount: {format_currency(analytics.max_bet.stake)}\n"
                f"Bet Number: {analytics.max_bet.bet_number}\n"
                f"Strategy: {analytics.max_bet.strategy_type}\n"
                f"Game Score: {analytics.max_bet.game_score}\n"
                f"Game Sequence: {analytics.max_bet.game_sequence}\n"
                f"Result: {'Won' if analytics.max_bet.won else 'Lost'}\n"
                f"Timestamp: {analytics.max_bet.timestamp}\n"
            )
        
        ax5.text(0.1, 0.5, stats_text, fontsize=12, verticalalignment='center')
        
        # 6. Individual Bet ROI Distribution
        ax6 = plt.subplot(6, 1, 6)
        if len(analytics.bets) > 0:
            bet_rois = [bet.roi for bet in analytics.bets]
            ax6.hist(bet_rois, bins=50, color='orange', alpha=0.7)
            mean_bet_roi = np.mean(bet_rois)
            ax6.axvline(mean_bet_roi, color='red', linestyle='--', label=f"Mean ROI = {mean_bet_roi:.2f}%")
        ax6.set_title('Distribution of Individual Bet ROI', fontsize=14, pad=20)
        ax6.set_xlabel('Bet ROI (%)')
        ax6.set_ylabel('Frequency')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\nDetailed analytics report saved as '{output_file}'")
    except Exception as e:
        print(f"\nWarning: Could not generate visual analytics due to: {str(e)}")

def main():
    print("Loading combined.csv for Monte Carlo analysis...")
    df = pd.read_csv('combined.csv')
    df['date'] = pd.to_datetime(df['date'], format="%d-%b-%y", dayfirst=True, errors='coerce')
    
    print(f"\nStarting Monte Carlo with {MONTE_CARLO_SIMULATIONS} simulations "
          f"of {GAMES_PER_SIMULATION} games each...\n")
    
    final_bankrolls = []
    total_profits = []
    rois = []
    max_drawdowns = []
    
    final_analytics = None
    final_session_stats = None
    
    # Perform the Monte Carlo loop
    for sim_index in tqdm(range(1, MONTE_CARLO_SIMULATIONS + 1), desc="Simulations"):
        fb, tp, roi_val, md, session_stats, analytics = run_one_monte_carlo(df, n_games=GAMES_PER_SIMULATION)
        final_bankrolls.append(fb)
        total_profits.append(tp)
        rois.append(roi_val)
        max_drawdowns.append(md)
        
        # Keep the final analytics + session_stats from the last simulation
        final_analytics = analytics
        final_session_stats = session_stats
    
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
    
    # Now generate a detailed report + bet breakdown for the final simulation
    if final_analytics is not None and final_session_stats is not None:
        # Save the bet-by-bet breakdown
        generate_bet_breakdown_report(final_analytics, output_file="bet_breakdown.csv")
        # Generate the big analytics chart
        generate_detailed_analytics_report(final_analytics, final_session_stats, output_file="monte_carlo_results.png")
    
    print("\nMonte Carlo Analysis Complete!")

if __name__ == "__main__":
    main()
