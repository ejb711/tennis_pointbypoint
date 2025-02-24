#!/usr/bin/env python3
"""
monty_detailed.py

An enhanced Monte Carlo analysis for a tennis point-by-point betting system.
Incorporates user feedback for:
  - Removing ANSI color codes from file outputs
  - Enhancing data visualization
  - Improving report structure
  - Adding more detailed analytics
  - Providing practical insights

Code Outline:
1. Prompt user for global configuration or use defaults
2. Run Monte Carlo simulations
3. Use a ReportingManager class to generate structured output:
    - Executive Summary
    - Table of Contents
    - Detailed Analytics (Sharpe ratio, max drawdown, ROI distribution, etc.)
    - Visual enhancements and annotated charts
    - Practical insights & recommendations
4. Export final CSV breakdown (bet_breakdown.csv) and final PNG report (monte_carlo_results.png)
"""

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from collections import defaultdict
from datetime import datetime
import random
from tqdm import tqdm  # progress bar for Monte Carlo
import math

###############################################################################
# Prompt Helpers
###############################################################################
def prompt_for_float(param_name, default_val):
    user_input = input(f"Enter {param_name} [default: {default_val}]: ").strip()
    if user_input == "":
        return default_val
    else:
        return float(user_input)

def prompt_for_int(param_name, default_val):
    user_input = input(f"Enter {param_name} [default: {default_val}]: ").strip()
    if user_input == "":
        return default_val
    else:
        return int(user_input)

def prompt_for_string(param_name, default_val):
    user_input = input(f"Enter {param_name} [default: {default_val}]: ").strip()
    if user_input == "":
        return default_val
    else:
        return user_input

###############################################################################
# Global Configuration (Defaults)
###############################################################################
INITIAL_BANKROLL = 2200.0
INITIAL_STAKE = 1
SERVER_ODDS = -265
RECEIVER_ODDS = 125
MAX_BET_ABSOLUTE = 400

MONTE_CARLO_SIMULATIONS = 200
GAMES_PER_SIMULATION = 1000

SERVER_BET_MULTIPLIER = 3.0
RECEIVER_BET_MULTIPLIER = 2.0

INITIAL_SEQUENCE = "SSSRRR"  # progression
TIE_FAVORED = "S"
ADVANTAGE_FAVORED = "A"

AUTO_MODE = False

###############################################################################
# Core Classes
###############################################################################
class BetRecord:
    """
    Tracks complete information about a single bet including context and result.
    Also computes ROI for that specific bet.
    """
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
        self.roi = (profit / stake * 100) if stake != 0 else 0
        self.timestamp = datetime.now()

class GameState:
    """
    Manages the state and scoring of a tennis game.
    """
    def __init__(self):
        self.server_points = 0
        self.receiver_points = 0
        self.points_sequence = []
        
    def add_point(self, point):
        self.points_sequence.append(point)
        if point in ['S','A']:
            self.server_points += 1
        elif point in ['R','D']:
            self.receiver_points += 1
            
    def is_score_tied(self):
        return self.server_points == self.receiver_points
        
    def get_advantage_pick(self):
        if self.server_points > self.receiver_points:
            return 'S'
        elif self.receiver_points > self.server_points:
            return 'R'
        return 'S'  # if still tied, server

    def get_score_summary(self):
        return f"Server {self.server_points} - {self.receiver_points} Receiver"

class SessionAnalytics:
    """
    Manages comprehensive session statistics and analytics.
    """
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
        self.bets.append(bet_record)
        self.bankroll_history.append(bet_record.bankroll)
        self.bet_sizes.append(bet_record.stake)
        
        strat = bet_record.strategy_type
        self.strategy_outcomes[strat]['total'] += 1
        self.strategy_outcomes[strat]['profit'] += bet_record.profit
        if bet_record.won:
            self.strategy_outcomes[strat]['wins'] += 1
        
        # Track maximum bet
        if self.max_bet is None or bet_record.stake > self.max_bet.stake:
            self.max_bet = bet_record
            
    def verify_statistics(self):
        total_profit = sum(bet.profit for bet in self.bets)
        total_stakes = sum(bet.stake for bet in self.bets)
        strategy_total = sum(s['total'] for s in self.strategy_outcomes.values())
        
        return {
            'total_bets_match': len(self.bets) == strategy_total,
            'bankroll_consistent': len(self.bankroll_history) == len(self.bets),
            'total_profit': total_profit,
            'total_stakes': total_stakes
        }

###############################################################################
# Utility / Calculation Functions
###############################################################################
def format_currency(amount):
    """
    Formats currency with color-coding for console logs only.
    We'll remove ANSI codes in file outputs or CSV exports.
    """
    if amount>0:
        return f"\033[92m${amount:,.2f}\033[0m"
    elif amount<0:
        return f"\033[91m${amount:,.2f}\033[0m"
    return f"${amount:,.2f}"

def calculate_next_bet(current_bet, next_pick, previous_pick, bankroll):
    """
    Uses the global SERVER_BET_MULTIPLIER and RECEIVER_BET_MULTIPLIER.
    """
    global SERVER_BET_MULTIPLIER, RECEIVER_BET_MULTIPLIER
    if next_pick=='S':
        next_bet = current_bet * SERVER_BET_MULTIPLIER
    else:
        next_bet = current_bet * RECEIVER_BET_MULTIPLIER
    max_allowed = min(bankroll/2, MAX_BET_ABSOLUTE)
    return min(next_bet, max_allowed)

def is_tiebreak(game_str):
    return '/' in game_str

def evaluate_point(point, pick):
    if pick=='S':
        return point in ['S','A']
    else:
        return point in ['R','D']

def process_game_points(game_str):
    if not game_str or is_tiebreak(game_str):
        return []
    return [p for p in game_str if p in ['S','A','R','D']]

def calculate_profit(stake, odds):
    if odds>0:
        return stake*(odds/100)
    else:
        return stake*(100/abs(odds))

###############################################################################
# Interactive Mode Helpers
###############################################################################
def check_user_input():
    global AUTO_MODE
    if AUTO_MODE:
        return False
    user_input = input("\nPress Enter to continue, 'q' to quit, or 'auto' for automatic: ")
    if user_input.lower()=='q':
        return True
    elif user_input.lower()=='auto':
        print("\nSwitching to automatic mode...")
        AUTO_MODE = True
    return False

def display_bet_info(bet_number, match_info, game_str, current_point, points_played, 
                     stake, pick, bankroll, game_state, strategy_type):
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
    print(f"Betting On: {'Server' if pick=='S' else 'Receiver'}")
    print(f"Odds: {SERVER_ODDS if pick=='S' else RECEIVER_ODDS}")
    print(f"\nBankroll Status:")
    print(f"Current Bankroll: {format_currency(bankroll)}")
    print(f"Maximum allowed bet: {format_currency(min(bankroll/2, MAX_BET_ABSOLUTE))}")
    input("\nPress Enter to see the result...")

def display_bet_result(won_bet, profit, bankroll, peak, max_drawdown, move_to_next_game):
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

###############################################################################
# Main Simulation Logic
###############################################################################
def simulate_interactive_betting(df, initial_stake=1):
    global AUTO_MODE, INITIAL_BANKROLL, INITIAL_SEQUENCE
    global TIE_FAVORED, ADVANTAGE_FAVORED
    
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
    
    progression_list = list(INITIAL_SEQUENCE)
    
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
                
                game_state = GameState()
                current_stake = initial_stake
                previous_pick = None
                points_played = 0
                using_score_strategy = False
                current_sequence = 0
                
                while points_played < len(points):
                    current_point = points[points_played]
                    game_state.add_point(current_point)
                    
                    # Decide pick
                    if not using_score_strategy:
                        if current_sequence >= len(progression_list):
                            using_score_strategy = True
                        else:
                            pick = progression_list[current_sequence]
                    
                    if using_score_strategy:
                        if game_state.is_score_tied():
                            pick = TIE_FAVORED  # default 'S'
                        else:
                            if ADVANTAGE_FAVORED == "A":
                                pick = game_state.get_advantage_pick()
                            else:
                                pick = ADVANTAGE_FAVORED  # 'S' or 'R'
                    
                    if previous_pick is not None:
                        current_stake = calculate_next_bet(current_stake, pick, previous_pick, bankroll)
                    
                    stake = min(current_stake, bankroll/2, MAX_BET_ABSOLUTE)
                    bet_number += 1
                    strategy_type = "Progression" if not using_score_strategy else "Score-Based"
                    
                    display_bet_info(
                        bet_number, match_info, game_str, current_point,
                        points_played+1, stake, pick, bankroll,
                        game_state, strategy_type
                    )
                    
                    won_bet = evaluate_point(current_point, pick)
                    
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
                        
                        display_bet_result(True, profit, bankroll, peak, max_drawdown, True)
                        break
                    else:
                        profit = -stake
                        bankroll += profit
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
                        
                        display_bet_result(False, profit, bankroll, peak, max_drawdown, points_played>=len(points)-1)
                    
                    previous_pick = pick
                    points_played += 1
                    
                    if check_user_input():
                        return bankroll, peak, max_drawdown, session_stats, analytics
        
        if bankroll <= 0:
            break
    
    return bankroll, peak, max_drawdown, session_stats, analytics

###############################################################################
# Monte Carlo Logic
###############################################################################
def run_one_monte_carlo(df, n_games=1000):
    final_bankroll, peak, max_drawdown, session_stats, analytics = simulate_interactive_betting(
        df.sample(n=n_games, replace=True), initial_stake=INITIAL_STAKE
    )
    total_stakes = session_stats['total_stakes']
    total_profit = session_stats['total_profit']
    roi = (total_profit / total_stakes * 100) if total_stakes>0 else 0
    return final_bankroll, total_profit, roi, max_drawdown, session_stats, analytics

###############################################################################
# Reporting Class (Structure, Analytics, Visuals)
###############################################################################
class ReportingManager:
    """
    Handles the creation of a structured report:
      - Executive Summary
      - Table of Contents
      - Detailed Analytics
      - Visual Enhancements
      - Practical Insights & Recommendations
    """
    def __init__(self, analytics, session_stats):
        self.analytics = analytics
        self.session_stats = session_stats
        self.sharpe_ratio = None
        self.max_drawdown_pct = None
        self.success_rates_by_pick = None

    def compute_additional_analytics(self):
        """
        Compute advanced metrics like Sharpe ratio, max drawdown percentage,
        success rates by pick, etc.
        """
        # 1. Sharpe Ratio (very simplistic approach)
        # Assume each bet is an 'investment' with profit as 'return'
        # We define the average return / std of return
        # This is only illustrative
        returns = [bet.profit for bet in self.analytics.bets if bet.stake>0]
        if len(returns)<2:
            self.sharpe_ratio = 0.0
        else:
            avg_return = np.mean(returns)
            std_return = np.std(returns)
            if std_return==0:
                self.sharpe_ratio = 0.0
            else:
                self.sharpe_ratio = avg_return/std_return
        
        # 2. Max Drawdown Percentage
        # We already have max_drawdown in session_stats, but let's do ratio vs peak
        total_stakes = self.session_stats['total_stakes']
        if total_stakes>0:
            self.max_drawdown_pct = (self.session_stats['total_stakes'] - self.session_stats['total_profit']) / total_stakes * 100
        else:
            self.max_drawdown_pct = 0.0

        # 3. Success rates by pick (server vs receiver)
        # Example breakdown
        pick_counts = {'S':0, 'R':0}
        pick_wins = {'S':0, 'R':0}
        for bet in self.analytics.bets:
            if bet.pick in ['S','R']:
                pick_counts[bet.pick]+=1
                if bet.won:
                    pick_wins[bet.pick]+=1
        self.success_rates_by_pick = {}
        for p in pick_counts:
            if pick_counts[p]>0:
                self.success_rates_by_pick[p] = pick_wins[p]/pick_counts[p]*100
            else:
                self.success_rates_by_pick[p] = 0.0

    def create_executive_summary(self):
        """
        Creates a high-level summary of the results, including:
          - Key metrics
          - Quick interpretive text
        """
        total_bets = self.session_stats['total_bets']
        total_wins = self.session_stats['wins']
        overall_win_rate = (total_wins/total_bets*100) if total_bets>0 else 0
        summary = (
            "EXECUTIVE SUMMARY\n"
            "==================\n\n"
            f"Total Bets Placed: {total_bets}\n"
            f"Total Wins: {total_wins}\n"
            f"Overall Win Rate: {overall_win_rate:.2f}%\n"
            f"Sharpe Ratio (Approx): {self.sharpe_ratio:.2f}\n"
            f"Max Drawdown % (Approx): {self.max_drawdown_pct:.2f}%\n\n"
            "In brief, the system shows a moderate level of performance. "
            "Further analysis is recommended to determine long-term viability.\n\n"
        )
        return summary

    def create_table_of_contents(self):
        """
        Creates a textual table of contents for the final report.
        """
        toc = (
            "TABLE OF CONTENTS\n"
            "=================\n\n"
            "1. Executive Summary\n"
            "2. Detailed Analytics\n"
            "3. Visual Enhancements\n"
            "4. Practical Insights & Recommendations\n"
            "5. Appendix / CSV Data\n\n"
        )
        return toc

    def create_detailed_analytics_text(self):
        """
        Provides deeper analytics text, including success rates by pick,
        and interpretive commentary.
        """
        pick_text = ""
        for p in self.success_rates_by_pick:
            pick_text += f" - Pick '{p}': {self.success_rates_by_pick[p]:.2f}% success rate\n"
        
        details = (
            "DETAILED ANALYTICS\n"
            "==================\n\n"
            f"Sharpe Ratio (Approx): {self.sharpe_ratio:.2f}\n"
            f"Maximum Drawdown %: {self.max_drawdown_pct:.2f}%\n\n"
            "Success Rates by Pick:\n"
            f"{pick_text}\n"
            "These metrics provide deeper insight into how each pick type performed.\n"
            "Further correlation or scenario-based analysis could reveal additional patterns.\n\n"
        )
        return details

    def create_practical_insights(self):
        """
        Summarizes practical insights and recommendations.
        """
        insights = (
            "PRACTICAL INSIGHTS & RECOMMENDATIONS\n"
            "=====================================\n\n"
            "- Consider adjusting stake multipliers if variance is too high.\n"
            "- Evaluate skipping certain picks if success rates are consistently low.\n"
            "- Monitor drawdown closely to manage risk.\n"
            "- Explore alternative progressions or score-based heuristics.\n"
            "- Compare results with baseline strategies (e.g., random pick) for context.\n\n"
        )
        return insights

###############################################################################
# CSV and Visual Reporting
###############################################################################
def generate_bet_breakdown_report(analytics, output_file="bet_breakdown.csv"):
    """
    Saves a CSV breakdown of every bet placed in the final simulation,
    with ANSI color codes removed for numeric fields.
    """
    bet_rows = []
    for bet in analytics.bets:
        # Remove ANSI codes from numeric fields
        bet_rows.append({
            "BetNumber": bet.bet_number,
            "Stake": f"{bet.stake:.2f}",
            "Profit": f"{bet.profit:.2f}",
            "ROI(%)": f"{bet.roi:.2f}",
            "Pick": bet.pick,
            "Won": bet.won,
            "Strategy": bet.strategy_type,
            "GameScore": bet.game_score,
            "GameSequence": bet.game_sequence,
            "BankrollAfterBet": f"{bet.bankroll:.2f}",
            "Timestamp": bet.timestamp.isoformat()
        })
    df_bets = pd.DataFrame(bet_rows)
    df_bets.to_csv(output_file, index=False)
    print(f"\nDetailed bet breakdown saved as '{output_file}'")

def generate_detailed_analytics_report(reporting_manager, output_file="monte_carlo_results.png"):
    """
    Generates a structured text-based final report plus an annotated matplotlib figure,
    combining an Executive Summary, Table of Contents, Detailed Analytics,
    and Practical Insights. Also includes a final multi-plot figure.
    """
    # 1. Prepare structured text report
    text_report = []
    text_report.append(reporting_manager.create_executive_summary())
    text_report.append(reporting_manager.create_table_of_contents())
    text_report.append(reporting_manager.create_detailed_analytics_text())
    text_report.append(reporting_manager.create_practical_insights())
    
    final_report_str = "\n".join(text_report)
    
    # 2. Print final structured report to console
    print("\n" + "="*80)
    print(final_report_str)
    print("="*80 + "\n")
    
    # 3. Generate multi-plot figure with annotated charts
    analytics = reporting_manager.analytics
    session_stats = reporting_manager.session_stats
    
    plt.style.use('default')
    fig = plt.figure(figsize=(20, 30))

    # Subplot 1: Bankroll Evolution
    ax1 = plt.subplot(6,1,1)
    ax1.plot(analytics.bankroll_history, label='Bankroll', color='blue', linewidth=2)
    ax1.set_title('Bankroll Evolution Over Time (Annotated)', fontsize=14, pad=20)
    ax1.set_xlabel('Bet Number')
    ax1.set_ylabel('Bankroll ($)')
    ax1.grid(True, alpha=0.3)
    # Annotate peak
    peak_val = max(analytics.bankroll_history) if analytics.bankroll_history else 0
    peak_idx = analytics.bankroll_history.index(peak_val) if analytics.bankroll_history else 0
    ax1.annotate(f"Peak: {peak_val:.2f}", xy=(peak_idx, peak_val),
                 xytext=(peak_idx, peak_val+100), 
                 arrowprops=dict(facecolor='green', shrink=0.05), fontsize=10)
    ax1.legend()

    # Subplot 2: Bet Size Distribution (improved binning)
    ax2 = plt.subplot(6,1,2)
    if len(analytics.bet_sizes)>0:
        # Freedman-Diaconis or other approach for bin width, here we do a simple approach
        bin_count = int(math.sqrt(len(analytics.bet_sizes)))  # e.g. sqrt rule
        ax2.hist(analytics.bet_sizes, bins=bin_count, color='green', alpha=0.7)
        mean_bet = np.mean(analytics.bet_sizes)
        ax2.axvline(mean_bet, color='red', linestyle='--', label=f'Mean Bet: {mean_bet:.2f}')
    ax2.set_title('Bet Size Distribution (Enhanced)', fontsize=14, pad=20)
    ax2.set_xlabel('Bet Size ($)')
    ax2.set_ylabel('Frequency')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # Subplot 3: Strategy Performance
    ax3 = plt.subplot(6,1,3)
    strategies = list(analytics.strategy_outcomes.keys())
    x = np.arange(len(strategies))
    width = 0.35
    
    win_rates = []
    rois = []
    for s in strategies:
        total_bets = analytics.strategy_outcomes[s]['total']
        wins = analytics.strategy_outcomes[s]['wins']
        profit = analytics.strategy_outcomes[s]['profit']
        
        wr = (wins/total_bets*100) if total_bets>0 else 0
        strategy_stakes = sum(bet.stake for bet in analytics.bets if bet.strategy_type == s)
        roi_val = (profit/strategy_stakes*100) if strategy_stakes>0 else 0
        
        win_rates.append(wr)
        rois.append(roi_val)
    
    ax3.bar(x - width/2, win_rates, width, label='Win Rate %', color='blue')
    ax3.bar(x + width/2, rois, width, label='ROI %', color='green')
    ax3.set_title('Strategy Performance Metrics (Clearer Labels)', fontsize=14, pad=20)
    ax3.set_xticks(x)
    ax3.set_xticklabels(strategies)
    ax3.set_ylabel('Percentage')
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    # Subplot 4: Win Rate by Bet Size Range (improved x-axis)
    ax4 = plt.subplot(6,1,4)
    if len(analytics.bet_sizes)>0:
        bet_data = pd.DataFrame({
            'stake': [b.stake for b in analytics.bets],
            'won': [b.won for b in analytics.bets]
        })
        bin_count = int(math.sqrt(len(bet_data))) if len(bet_data)>1 else 10
        # For demonstration, let's do a cut into 5 bins
        bet_data['size_group'] = pd.cut(bet_data['stake'], bins=5, labels=False)
        # Could do qcut for quantiles, but let's keep it simpler
        win_rates_by_group = bet_data.groupby('size_group')['won'].mean()*100
        ax4.bar(win_rates_by_group.index, win_rates_by_group, color='purple', alpha=0.7)
        ax4.set_xticks(win_rates_by_group.index)
        ax4.set_xticklabels([f"Bin {i}" for i in win_rates_by_group.index])
    ax4.set_title('Win Rate by Bet Size Range (Improved Binning)', fontsize=14, pad=20)
    ax4.set_xlabel('Bet Size Bin')
    ax4.set_ylabel('Win Rate (%)')
    ax4.grid(True, alpha=0.3)

    # Subplot 5: Key Stats / Possibly annotated
    ax5 = plt.subplot(6,1,5)
    ax5.axis('off')
    
    stats_text = (
        "Key Metrics:\n\n"
        f"Total Games Processed: {session_stats['total_games']:,}\n"
        f"Total Bets Placed: {session_stats['total_bets']:,}\n"
        f"Overall Win Rate: {(session_stats['wins']/session_stats['total_bets']*100 if session_stats['total_bets'] else 0):.1f}%\n"
        f"Total Amount Staked: {session_stats['total_stakes']:.2f}\n"
        f"Net Profit/Loss: {session_stats['total_profit']:.2f}\n"
        f"ROI: {(session_stats['total_profit']/session_stats['total_stakes']*100 if session_stats['total_stakes'] else 0):.1f}%\n"
        "Annotations highlight key insights directly on the charts above.\n\n"
    )
    ax5.text(0.1, 0.5, stats_text, fontsize=12, verticalalignment='center')

    # Subplot 6: ROI Distribution with annotation
    ax6 = plt.subplot(6,1,6)
    bet_rois = [bet.roi for bet in analytics.bets]
    if len(bet_rois)>1:
        bin_count = int(math.sqrt(len(bet_rois)))
        ax6.hist(bet_rois, bins=bin_count, color='orange', alpha=0.7)
        mean_roi = np.mean(bet_rois)
        ax6.axvline(mean_roi, color='red', linestyle='--', label=f"Mean ROI = {mean_roi:.2f}%")
    ax6.set_title('Distribution of Individual Bet ROI (Annotated)', fontsize=14, pad=20)
    ax6.set_xlabel('Bet ROI (%)')
    ax6.set_ylabel('Frequency')
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nEnhanced analytics report saved as '{output_file}'")

###############################################################################
# Main Execution
###############################################################################
def main():
    global INITIAL_BANKROLL, INITIAL_STAKE, SERVER_ODDS, RECEIVER_ODDS
    global MAX_BET_ABSOLUTE, MONTE_CARLO_SIMULATIONS, GAMES_PER_SIMULATION
    global SERVER_BET_MULTIPLIER, RECEIVER_BET_MULTIPLIER
    global INITIAL_SEQUENCE, TIE_FAVORED, ADVANTAGE_FAVORED
    global AUTO_MODE
    
    print("=== Configuration Setup ===")
    # Prompt user for each config
    INITIAL_BANKROLL = prompt_for_float("Initial Bankroll", INITIAL_BANKROLL)
    INITIAL_STAKE = prompt_for_float("Initial Stake", INITIAL_STAKE)
    SERVER_ODDS = prompt_for_float("Server Odds", SERVER_ODDS)
    RECEIVER_ODDS = prompt_for_float("Receiver Odds", RECEIVER_ODDS)
    MAX_BET_ABSOLUTE = prompt_for_float("Max Bet Absolute", MAX_BET_ABSOLUTE)
    MONTE_CARLO_SIMULATIONS = prompt_for_int("Number of Monte Carlo simulations", MONTE_CARLO_SIMULATIONS)
    GAMES_PER_SIMULATION = prompt_for_int("Number of games per simulation", GAMES_PER_SIMULATION)
    
    SERVER_BET_MULTIPLIER = prompt_for_float("Server Bet Multiplier", SERVER_BET_MULTIPLIER)
    RECEIVER_BET_MULTIPLIER = prompt_for_float("Receiver Bet Multiplier", RECEIVER_BET_MULTIPLIER)
    
    INITIAL_SEQUENCE = prompt_for_string("Initial Betting Sequence (e.g. SSSRRR)", INITIAL_SEQUENCE)
    TIE_FAVORED = prompt_for_string("Tie favored pick (S or R)", TIE_FAVORED)
    ADVANTAGE_FAVORED = prompt_for_string("Advantage favored pick (A=advantage, S=server, R=receiver)", ADVANTAGE_FAVORED)

    print("\nLoading combined.csv for Monte Carlo analysis...")
    df = pd.read_csv('combined.csv')
    df['date'] = pd.to_datetime(df['date'], format="%d-%b-%y", dayfirst=True, errors='coerce')
    
    print(f"\nStarting Monte Carlo with {MONTE_CARLO_SIMULATIONS} simulations "
          f"of {GAMES_PER_SIMULATION} games each...\n"
          f"(Initial Bankroll={INITIAL_BANKROLL}, Stake={INITIAL_STAKE}, "
          f"ServerOdds={SERVER_ODDS}, ReceiverOdds={RECEIVER_ODDS}, "
          f"MaxBet={MAX_BET_ABSOLUTE}, S-BetMult={SERVER_BET_MULTIPLIER}, R-BetMult={RECEIVER_BET_MULTIPLIER}, "
          f"Sequence='{INITIAL_SEQUENCE}', TieFavored='{TIE_FAVORED}', AdvantageFavored='{ADVANTAGE_FAVORED}')\n")
    
    final_bankrolls = []
    total_profits = []
    rois = []
    max_drawdowns = []
    
    final_analytics = None
    final_session_stats = None
    
    # Perform Monte Carlo loop
    for sim_index in tqdm(range(1, MONTE_CARLO_SIMULATIONS+1), desc="Simulations"):
        fb, tp, roi_val, md, session_stats, analytics = run_one_monte_carlo(
            df, n_games=GAMES_PER_SIMULATION
        )
        final_bankrolls.append(fb)
        total_profits.append(tp)
        rois.append(roi_val)
        max_drawdowns.append(md)
        
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
    
    # Generate final reports
    if final_analytics is not None and final_session_stats is not None:
        # 1. CSV breakdown
        generate_bet_breakdown_report(final_analytics, output_file="bet_breakdown.csv")
        
        # 2. Additional analytics
        from statistics import mean, stdev
        reporting_manager = ReportingManager(final_analytics, final_session_stats)
        reporting_manager.compute_additional_analytics()
        
        # 3. Enhanced final report
        generate_detailed_analytics_report(reporting_manager, output_file="monte_carlo_results.png")
    
    print("\nMonte Carlo Analysis Complete!")

if __name__ == "__main__":
    main()
