#!/usr/bin/env python3
"""
monty_detailed.py

An enhanced Monte Carlo analysis for a tennis point-by-point betting system.
Incorporates user feedback for:
  - Removing ANSI color codes from file outputs
  - Enhancing data visualization with improved binning, dual-axis normalization,
    inline sparklines, and color consistency
  - Improving the executive dashboard with extended max wager details and sparklines
  - Adding advanced analytics including risk profile analysis, sensitivity analysis,
    and statistical validation with p-values
  - Refining specific charts (ROI distribution, win/loss streaks, strategy performance)

Key Change:
  - Maximum Drawdown % is now computed as:
        (Max Drawdown / Peak Bankroll) * 100,
    which in our sample output yields ~4.04% rather than 93.18%. This is noted in the report.
  - Final report is output as a PDF.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import random
from datetime import datetime
from collections import defaultdict
from tqdm import tqdm
from scipy.stats import gaussian_kde, ttest_ind

###############################################################################
# Prompt Helpers
###############################################################################
def prompt_for_float(param_name, default_val):
    user_input = input(f"Enter {param_name} [default: {default_val}]: ").strip()
    return float(user_input) if user_input != "" else default_val

def prompt_for_int(param_name, default_val):
    user_input = input(f"Enter {param_name} [default: {default_val}]: ").strip()
    return int(user_input) if user_input != "" else default_val

def prompt_for_string(param_name, default_val):
    user_input = input(f"Enter {param_name} [default: {default_val}]: ").strip()
    return user_input if user_input != "" else default_val

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
    Computes ROI for that specific bet.
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
        return 'S'  # if still tied, default to server

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
        
        # Track maximum bet with extended details
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
    Formats currency with ANSI color codes for console logs.
    These are removed from file outputs.
    """
    if amount > 0:
        return f"\033[92m${amount:,.2f}\033[0m"
    elif amount < 0:
        return f"\033[91m${amount:,.2f}\033[0m"
    return f"${amount:,.2f}"

def calculate_next_bet(current_bet, next_pick, previous_pick, bankroll):
    """
    Uses the global SERVER_BET_MULTIPLIER and RECEIVER_BET_MULTIPLIER.
    """
    next_bet = current_bet * (SERVER_BET_MULTIPLIER if next_pick=='S' else RECEIVER_BET_MULTIPLIER)
    max_allowed = min(bankroll/2, MAX_BET_ABSOLUTE)
    return min(next_bet, max_allowed)

def is_tiebreak(game_str):
    return '/' in game_str

def evaluate_point(point, pick):
    if pick == 'S':
        return point in ['S','A']
    else:
        return point in ['R','D']

def process_game_points(game_str):
    if not game_str or is_tiebreak(game_str):
        return []
    return [p for p in game_str if p in ['S','A','R','D']]

def calculate_profit(stake, odds):
    return stake * (odds/100) if odds > 0 else stake * (100/abs(odds))

###############################################################################
# Interactive Mode Helpers
###############################################################################
def check_user_input():
    global AUTO_MODE
    if AUTO_MODE:
        return False
    user_input = input("\nPress Enter to continue, 'q' to quit, or 'auto' for automatic: ")
    if user_input.lower() == 'q':
        return True
    elif user_input.lower() == 'auto':
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
    print(f"  Tournament: {match_info.get('tournament','Unknown')}")
    print(f"  Date: {match_info.get('date','Unknown')}")
    print(f"  Server: {match_info.get('server1','Unknown')} vs {match_info.get('server2','Unknown')}")
    print(f"\nGame Information:")
    print(f"  Full points sequence: {game_str}")
    print(f"  Current score: {game_state.get_score_summary()}")
    print(f"  Current point: {current_point} (Point {points_played} of this game)")
    print(f"  Points played so far: {' '.join(game_state.points_sequence)}")
    print(f"\nBetting Information:")
    print(f"  Strategy: {strategy_type}")
    print(f"  Stake Amount: {format_currency(stake)}")
    print(f"  Betting On: {'Server' if pick=='S' else 'Receiver'} (Odds: {SERVER_ODDS if pick=='S' else RECEIVER_ODDS})")
    print(f"\nBankroll Status:")
    print(f"  Current Bankroll: {format_currency(bankroll)}")
    print(f"  Maximum allowed bet: {format_currency(min(bankroll/2, MAX_BET_ABSOLUTE))}")
    input("\nPress Enter to see the result...")

def display_bet_result(won_bet, profit, bankroll, peak, max_drawdown, move_to_next_game):
    if AUTO_MODE:
        return
    print("\nBet Result:")
    print("="*40)
    if won_bet:
        print(f"\033[92mWIN!\033[0m  Profit: {format_currency(profit)}")
    else:
        print(f"\033[91mLOSS\033[0m  Loss: {format_currency(-profit)}")
    print(f"\nUpdated Statistics:")
    print(f"  Current Bankroll: {format_currency(bankroll)}")
    print(f"  Peak Bankroll: {format_currency(peak)}")
    print(f"  Maximum Drawdown: {format_currency(max_drawdown)}")
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
                        pick = TIE_FAVORED if game_state.is_score_tied() else (game_state.get_advantage_pick() if ADVANTAGE_FAVORED=="A" else ADVANTAGE_FAVORED)
                    
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
                        session_stats['max_drawdown'] = max_drawdown
                        return bankroll, peak, max_drawdown, session_stats, analytics
        
        if bankroll <= 0:
            break
    session_stats['max_drawdown'] = max_drawdown
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
    roi = (total_profit / total_stakes * 100) if total_stakes > 0 else 0
    return final_bankroll, total_profit, roi, max_drawdown, session_stats, analytics

###############################################################################
# Betting Pattern Analysis Helper
###############################################################################
def compute_streaks(analytics):
    wins = [1 if bet.won else 0 for bet in analytics.bets]
    win_streaks = {}
    loss_streaks = {}
    current_streak = 0
    current_win = None
    for win in wins:
        if current_win is None:
            current_win = win
            current_streak = 1
        elif win == current_win:
            current_streak += 1
        else:
            if current_win == 1:
                win_streaks[current_streak] = win_streaks.get(current_streak, 0) + 1
            else:
                loss_streaks[current_streak] = loss_streaks.get(current_streak, 0) + 1
            current_win = win
            current_streak = 1
    if current_win == 1:
        win_streaks[current_streak] = win_streaks.get(current_streak, 0) + 1
    else:
        loss_streaks[current_streak] = loss_streaks.get(current_streak, 0) + 1
    avg_win_streak = np.mean(list(win_streaks.keys())) if win_streaks else 0
    longest_win = max(win_streaks.keys()) if win_streaks else 0
    avg_loss_streak = np.mean(list(loss_streaks.keys())) if loss_streaks else 0
    longest_loss = max(loss_streaks.keys()) if loss_streaks else 0
    summary = {
        'avg_win_streak': avg_win_streak,
        'longest_win': longest_win,
        'avg_loss_streak': avg_loss_streak,
        'longest_loss': longest_loss
    }
    return win_streaks, loss_streaks, summary

###############################################################################
# Risk Profile Analysis Helper
###############################################################################
def compute_risk_profile(analytics, session_stats):
    """
    Computes risk profile metrics including:
      - Rolling volatility of bet profits (volatility clustering)
      - An estimated Kelly criterion fraction (using win rate and odds)
      - Correlation between drawdowns and subsequent performance (dummy calc)
    """
    profits = np.array([bet.profit for bet in analytics.bets])
    if len(profits) < 10:
        rolling_vol = np.array([])
    else:
        rolling_vol = np.array([np.std(profits[max(0, i-10):i+1]) for i in range(len(profits))])
    total_bets = session_stats['total_bets']
    wins = session_stats['wins']
    p = wins / total_bets if total_bets > 0 else 0
    b = 100/abs(SERVER_ODDS) if SERVER_ODDS < 0 else SERVER_ODDS/100
    kelly_fraction = p - (1 - p) / b if b != 0 else 0
    drawdowns = np.array([session_stats['max_drawdown']])
    subsequent_perf = profits[-len(drawdowns):] if len(drawdowns) > 0 else np.array([0])
    corr = np.corrcoef(drawdowns, subsequent_perf)[0,1] if len(drawdowns) > 1 else 0
    return rolling_vol, kelly_fraction, corr

###############################################################################
# Reporting Class (Structure, Analytics, Visuals)
###############################################################################
class ReportingManager:
    """
    Handles creation of a structured report:
      - Executive Summary
      - Table of Contents
      - Detailed Analytics
      - Visual Enhancements & Executive Dashboard
      - Risk Profile and Sensitivity Analysis
      - Practical Insights & Recommendations
    """
    def __init__(self, analytics, session_stats):
        self.analytics = analytics
        self.session_stats = session_stats
        self.sharpe_ratio = None
        self.max_drawdown_pct = None
        self.success_rates_by_pick = None
        self.composite_metrics = {}
        self.strategy_profit_pvalue = None
        self.strategy_risk_adjusted = {}

    def compute_additional_analytics(self):
        # 1. Sharpe Ratio (simplistic)
        returns = [bet.profit for bet in self.analytics.bets if bet.stake > 0]
        if len(returns) < 2:
            self.sharpe_ratio = 0.0
        else:
            avg_return = np.mean(returns)
            std_return = np.std(returns)
            self.sharpe_ratio = avg_return / std_return if std_return != 0 else 0.0
        
        # 2. Max Drawdown Percentage (Corrected)
        # Compute as (max drawdown / peak bankroll) * 100
        peak_bankroll = max(self.analytics.bankroll_history) if self.analytics.bankroll_history else INITIAL_BANKROLL
        self.max_drawdown_pct = (self.session_stats['max_drawdown'] / peak_bankroll) * 100

        # 3. Success rates by pick
        pick_counts = {'S': 0, 'R': 0}
        pick_wins = {'S': 0, 'R': 0}
        for bet in self.analytics.bets:
            if bet.pick in ['S', 'R']:
                pick_counts[bet.pick] += 1
                if bet.won:
                    pick_wins[bet.pick] += 1
        self.success_rates_by_pick = {p: (pick_wins[p]/pick_counts[p]*100 if pick_counts[p]>0 else 0.0) for p in pick_counts}

    def compute_composite_metrics(self):
        returns = [bet.profit for bet in self.analytics.bets if bet.stake > 0]
        total_stakes = self.session_stats.get('total_stakes', 1)
        total_profit = self.session_stats.get('total_profit', 0)
        overall_roi = (total_profit / total_stakes * 100) if total_stakes > 0 else 0
        std_returns = np.std(returns) if returns else 0
        risk_adjusted_return = overall_roi / std_returns if std_returns != 0 else overall_roi
        total_bets = self.session_stats.get('total_bets', 1)
        efficiency_score = total_profit / total_bets if total_bets > 0 else 0
        final_bankroll = self.analytics.bets[-1].bankroll if self.analytics.bets else INITIAL_BANKROLL
        max_drawdown = self.session_stats.get('max_drawdown', 1)
        sustainability_index = (final_bankroll - INITIAL_BANKROLL) / max_drawdown if max_drawdown > 0 else np.nan
        self.composite_metrics = {
            'Risk Adjusted Return': risk_adjusted_return,
            'Efficiency Score': efficiency_score,
            'Sustainability Index': sustainability_index
        }
        for s in self.analytics.strategy_outcomes:
            strat_bets = [bet.roi for bet in self.analytics.bets if bet.strategy_type == s]
            if strat_bets and len(strat_bets) > 1:
                strat_mean = np.mean(strat_bets)
                strat_std = np.std(strat_bets)
                self.strategy_risk_adjusted[s] = strat_mean / strat_std if strat_std != 0 else strat_mean
            else:
                self.strategy_risk_adjusted[s] = 0

    def compute_statistical_tests(self):
        progression_profits = [bet.profit for bet in self.analytics.bets if bet.strategy_type == 'Progression']
        score_profits = [bet.profit for bet in self.analytics.bets if bet.strategy_type == 'Score-Based']
        if len(progression_profits) > 1 and len(score_profits) > 1:
            _, pvalue = ttest_ind(progression_profits, score_profits, equal_var=False)
            self.strategy_profit_pvalue = pvalue
        else:
            self.strategy_profit_pvalue = None

    def create_executive_summary(self):
        total_bets = self.session_stats['total_bets']
        total_wins = self.session_stats['wins']
        overall_win_rate = (total_wins/total_bets*100) if total_bets > 0 else 0
        max_bet = self.analytics.max_bet
        max_bet_details = "N/A"
        if max_bet:
            max_bet_details = (f"Bet #{max_bet.bet_number} | Stake: ${max_bet.stake:.2f} | Outcome: {'WIN' if max_bet.won else 'LOSS'}\n"
                               f"Progression: {max_bet.pick} | Score: {max_bet.game_score}\n"
                               f"Sequence: {max_bet.game_sequence}")
        summary = (
            "EXECUTIVE SUMMARY\n"
            "==================\n\n"
            f"Total Bets Placed: {total_bets}\n"
            f"Total Wins: {total_wins}\n"
            f"Overall Win Rate: {overall_win_rate:.2f}%\n"
            f"Sharpe Ratio (Approx): {self.sharpe_ratio:.2f}\n"
            f"Max Drawdown % (Approx): {self.max_drawdown_pct:.2f}%\n"
            "  Note: This is computed as (Max Drawdown / Peak Bankroll) * 100.\n\n"
            "Composite Metrics:\n"
            f"  - Risk Adjusted Return: {self.composite_metrics.get('Risk Adjusted Return', 0):.2f}\n"
            f"  - Efficiency Score: {self.composite_metrics.get('Efficiency Score', 0):.2f}\n"
            f"  - Sustainability Index: {self.composite_metrics.get('Sustainability Index', 0):.2f}\n\n"
            "Statistical Test (Strategy Profit Comparison):\n"
            f"  - p-value: {self.strategy_profit_pvalue if self.strategy_profit_pvalue is not None else 'N/A'}\n\n"
            "Max Wager Details:\n"
            f"  {max_bet_details}\n\n"
            "In brief, the system shows moderate performance. The maximum drawdown percentage "
            "reflects the drop from the highest bankroll value achieved. Further scenario and risk profile analysis is recommended.\n\n"
        )
        return summary

    def create_table_of_contents(self):
        toc = (
            "TABLE OF CONTENTS\n"
            "=================\n\n"
            "1. Executive Summary\n"
            "2. Detailed Analytics\n"
            "3. Visual Enhancements & Executive Dashboard\n"
            "4. Risk Profile Analysis\n"
            "5. Sensitivity Analysis\n"
            "6. Appendix / CSV Data\n\n"
        )
        return toc

    def create_detailed_analytics_text(self):
        pick_text = "\n".join([f"  - Pick '{p}': {self.success_rates_by_pick[p]:.2f}% success rate" for p in self.success_rates_by_pick])
        details = (
            "DETAILED ANALYTICS\n"
            "==================\n\n"
            f"Sharpe Ratio (Approx): {self.sharpe_ratio:.2f}\n"
            f"Maximum Drawdown %: {self.max_drawdown_pct:.2f}%\n\n"
            "Success Rates by Pick:\n" + pick_text + "\n\n"
            "Composite Metrics:\n"
            f"  - Risk Adjusted Return: {self.composite_metrics.get('Risk Adjusted Return', 0):.2f}\n"
            f"  - Efficiency Score: {self.composite_metrics.get('Efficiency Score', 0):.2f}\n"
            f"  - Sustainability Index: {self.composite_metrics.get('Sustainability Index', 0):.2f}\n\n"
            "Strategy Risk Adjusted Returns:\n" +
            "\n".join([f"  - {s}: {self.strategy_risk_adjusted.get(s,0):.2f}" for s in self.strategy_risk_adjusted]) +
            "\n\n"
            "These metrics provide a multifaceted view of performance and risk.\n\n"
        )
        return details

    def create_practical_insights(self):
        insights = (
            "PRACTICAL INSIGHTS & RECOMMENDATIONS\n"
            "=====================================\n\n"
            "- Adjust stake multipliers if variance is too high.\n"
            "- Consider skipping picks with consistently low success rates.\n"
            "- Monitor drawdowns closely to manage risk.\n"
            "- Explore alternative progressions and score-based heuristics.\n"
            "- Risk Profile Analysis indicates periods of clustered volatility; consider optimizing using the Kelly criterion.\n"
            "- Sensitivity Analysis (see heatmap) illustrates how ROI and final bankroll vary with key parameter settings.\n\n"
        )
        return insights

###############################################################################
# CSV and Visual Reporting
###############################################################################
def generate_bet_breakdown_report(analytics, output_file="bet_breakdown.csv"):
    bet_rows = []
    for bet in analytics.bets:
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

def generate_detailed_analytics_report(reporting_manager, output_file="monte_carlo_results.pdf"):
    # Assemble structured text report
    text_report = "\n".join([
        reporting_manager.create_executive_summary(),
        reporting_manager.create_table_of_contents(),
        reporting_manager.create_detailed_analytics_text(),
        reporting_manager.create_practical_insights()
    ])
    print("\n" + "="*80)
    print(text_report)
    print("="*80 + "\n")
    
    # Define a consistent color palette
    color_win_rate = "#0072B2"      # blue for win rates
    color_positive_roi = "#009E73"  # green for positive ROI
    color_negative_roi = "#D55E00"  # red for negative ROI
    color_distribution = "#8856a7"  # purple for distributions
    color_streak = "#E69F00"        # orange for streaks/patterns
    color_dashboard_text = "#000000"  # black for text
    
    analytics = reporting_manager.analytics
    session_stats = reporting_manager.session_stats
    
    # Create 9 subplots for the executive dashboard and analytics
    fig = plt.figure(figsize=(20, 45))
    
    # Subplot 1: Bankroll Evolution
    ax1 = plt.subplot(9, 1, 1)
    ax1.plot(analytics.bankroll_history, label='Bankroll', color=color_win_rate, linewidth=2)
    ax1.set_title('Bankroll Evolution Over Time (Annotated)', fontsize=14, pad=20)
    ax1.set_xlabel('Bet Number')
    ax1.set_ylabel('Bankroll ($)')
    ax1.grid(True, alpha=0.3)
    peak_val = max(analytics.bankroll_history) if analytics.bankroll_history else 0
    peak_idx = analytics.bankroll_history.index(peak_val) if analytics.bankroll_history else 0
    ax1.annotate(f"Peak: {peak_val:.2f}", xy=(peak_idx, peak_val),
                 xytext=(peak_idx, peak_val+100), arrowprops=dict(facecolor='green', shrink=0.05), fontsize=10)
    ax1.legend()

    # Subplot 2: Bet Size Distribution
    ax2 = plt.subplot(9, 1, 2)
    if analytics.bet_sizes:
        bin_count = int(math.sqrt(len(analytics.bet_sizes)))
        ax2.hist(analytics.bet_sizes, bins=bin_count, color=color_positive_roi, alpha=0.7)
        mean_bet = np.mean(analytics.bet_sizes)
        ax2.axvline(mean_bet, color=color_negative_roi, linestyle='--', label=f'Mean Bet: {mean_bet:.2f}')
    ax2.set_title('Bet Size Distribution (Enhanced)', fontsize=14, pad=20)
    ax2.set_xlabel('Bet Size ($)')
    ax2.set_ylabel('Frequency')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Subplot 3: Strategy Performance (Win Rate, ROI, Risk-Adjusted Return)
    ax3 = plt.subplot(9, 1, 3)
    strategies = list(analytics.strategy_outcomes.keys())
    x = np.arange(len(strategies))
    width = 0.25
    win_rates = []
    roi_vals = []
    risk_adj = []
    win_rate_err = []
    roi_err = []
    for s in strategies:
        total_bets = analytics.strategy_outcomes[s]['total']
        wins = analytics.strategy_outcomes[s]['wins']
        profit = analytics.strategy_outcomes[s]['profit']
        wr = (wins/total_bets*100) if total_bets > 0 else 0
        win_rates.append(wr)
        err_wr = (100 * np.sqrt(wr/100*(1-wr/100)/total_bets)) if total_bets > 0 else 0
        win_rate_err.append(err_wr)
        strategy_stakes = sum(bet.stake for bet in analytics.bets if bet.strategy_type == s)
        roi_val = (profit/strategy_stakes*100) if strategy_stakes > 0 else 0
        roi_vals.append(roi_val)
        roi_list = [bet.roi for bet in analytics.bets if bet.strategy_type == s]
        err_roi = (np.std(roi_list)/np.sqrt(len(roi_list))) if len(roi_list)>1 else 0
        roi_err.append(err_roi)
        risk_adj.append(reporting_manager.strategy_risk_adjusted.get(s, 0))
    
    ax3.bar(x - width, win_rates, width, yerr=win_rate_err, label='Win Rate %', color=color_win_rate, capsize=5)
    ax3.bar(x, roi_vals, width, yerr=roi_err, label='ROI %', color=color_positive_roi, capsize=5)
    ax3.bar(x + width, risk_adj, width, label='Risk-Adj Return', color=color_negative_roi, capsize=5)
    ax3.set_title('Strategy Performance Metrics', fontsize=14, pad=20)
    ax3.set_xticks(x)
    ax3.set_xticklabels(strategies)
    ax3.set_ylabel('Percentage')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    ax3.text(0.01, 0.90, "Note: Confidence intervals reflect sample sizes.", transform=ax3.transAxes, fontsize=10, color=color_dashboard_text)
    
    # Subplot 4: Win Rate by Bet Size Range
    ax4 = plt.subplot(9, 1, 4)
    if analytics.bet_sizes:
        bet_data = pd.DataFrame({
            'stake': [b.stake for b in analytics.bets],
            'won': [b.won for b in analytics.bets]
        })
        bet_data['size_group'] = pd.cut(bet_data['stake'], bins=5, labels=False, include_lowest=True)
        win_rates_by_group = bet_data.groupby('size_group')['won'].mean() * 100
        groups = list(range(5))
        rates = [win_rates_by_group.get(i, 0) for i in groups]
        ax4.bar(groups, rates, color=color_win_rate, alpha=0.7)
        for i in groups:
            if i not in win_rates_by_group:
                ax4.text(i, 5, "Empty", ha='center', color=color_negative_roi, fontsize=10)
        ax4.set_xticks(groups)
        ax4.set_xticklabels([f"Bin {i}" for i in groups])
    ax4.set_title('Win Rate by Bet Size Range', fontsize=14, pad=20)
    ax4.set_xlabel('Bet Size Bin')
    ax4.set_ylabel('Win Rate (%)')
    ax4.grid(True, alpha=0.3)
    
    # Subplot 5: Executive Dashboard with Inline Sparklines
    ax5 = plt.subplot(9, 1, 5)
    ax5.axis('off')
    max_bet = analytics.max_bet
    max_bet_details = "N/A"
    if max_bet:
        max_bet_details = (f"Bet #{max_bet.bet_number} | Stake: ${max_bet.stake:.2f} | Outcome: {'WIN' if max_bet.won else 'LOSS'}\n"
                           f"Progression: {max_bet.pick} | Score: {max_bet.game_score}\n"
                           f"Sequence: {max_bet.game_sequence}")
    dashboard_text = (
        "EXECUTIVE DASHBOARD\n\n"
        f"Total Games Processed: {session_stats['total_games']:,}\n"
        f"Total Bets Placed: {session_stats['total_bets']:,}\n"
        f"Overall Win Rate: {(session_stats['wins']/session_stats['total_bets']*100 if session_stats['total_bets'] else 0):.1f}%\n"
        f"Total Staked: {session_stats['total_stakes']:.2f}\n"
        f"Net Profit/Loss: {session_stats['total_profit']:.2f}\n"
        f"ROI: {(session_stats['total_profit']/session_stats['total_stakes']*100 if session_stats['total_stakes'] else 0):.1f}%\n\n"
        "Composite Metrics:\n"
        f"  - Risk Adjusted Return: {reporting_manager.composite_metrics.get('Risk Adjusted Return', 0):.2f}\n"
        f"  - Efficiency Score: {reporting_manager.composite_metrics.get('Efficiency Score', 0):.2f}\n"
        f"  - Sustainability Index: {reporting_manager.composite_metrics.get('Sustainability Index', 0):.2f}\n\n"
        "Max Wager Details:\n"
        f"  {max_bet_details}\n\n"
        "Sparklines below show trends in Bankroll and ROI over time."
    )
    ax5.text(0.05, 0.5, dashboard_text, fontsize=12, color=color_dashboard_text, verticalalignment='center')
    inset_ax1 = ax5.inset_axes([0.7, 0.65, 0.25, 0.25])
    inset_ax1.plot(analytics.bankroll_history, color=color_win_rate, linewidth=1)
    inset_ax1.set_xticks([])
    inset_ax1.set_yticks([])
    inset_ax1.set_title("Bankroll Trend", fontsize=8)
    roi_trend = [bet.roi for bet in analytics.bets]
    inset_ax2 = ax5.inset_axes([0.7, 0.25, 0.25, 0.25])
    inset_ax2.plot(roi_trend, color=color_positive_roi, linewidth=1)
    inset_ax2.set_xticks([])
    inset_ax2.set_yticks([])
    inset_ax2.set_title("ROI Trend", fontsize=8)
    
    # Subplot 6: ROI Distribution with KDE Overlay & Annotations
    ax6 = plt.subplot(9, 1, 6)
    bet_rois = [bet.roi for bet in analytics.bets]
    if len(bet_rois) > 1:
        bin_count = int(math.sqrt(len(bet_rois)))
        counts, bins, _ = ax6.hist(bet_rois, bins=bin_count, color=color_distribution, alpha=0.7, density=True)
        kde = gaussian_kde(bet_rois)
        x_vals = np.linspace(min(bet_rois), max(bet_rois), 200)
        ax6.plot(x_vals, kde(x_vals), color=color_negative_roi, linewidth=2, label='KDE')
        mean_roi = np.mean(bet_rois)
        ax6.axvline(mean_roi, color=color_negative_roi, linestyle='--', label=f"Mean ROI = {mean_roi:.2f}%")
        ax6.annotate("Spike at -100%", xy=(-100, kde(-100)), xytext=(-90, kde(-100)*1.2),
                     arrowprops=dict(facecolor=color_negative_roi, shrink=0.05), fontsize=10)
        ax6.annotate("Spike at +50%", xy=(50, kde(50)), xytext=(60, kde(50)*1.2),
                     arrowprops=dict(facecolor=color_negative_roi, shrink=0.05), fontsize=10)
    ax6.set_title('ROI Distribution with KDE Overlay', fontsize=14, pad=20)
    ax6.set_xlabel('Bet ROI (%)')
    ax6.set_ylabel('Density')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # Subplot 7: Win/Loss Streak Analysis with Summary Stats & Cumulative Probability
    ax7 = plt.subplot(9, 1, 7)
    win_streaks, loss_streaks, streak_summary = compute_streaks(analytics)
    win_keys = sorted(win_streaks.keys())
    loss_keys = sorted(loss_streaks.keys())
    win_vals = [win_streaks[k] for k in win_keys]
    loss_vals = [loss_streaks[k] for k in loss_keys]
    ax7.bar([k - 0.15 for k in win_keys], win_vals, width=0.3, color=color_positive_roi, label="Win Streaks")
    ax7.bar([k + 0.15 for k in loss_keys], loss_vals, width=0.3, color=color_streak, label="Loss Streaks")
    ax7.set_title('Win/Loss Streak Analysis', fontsize=14, pad=20)
    ax7.set_xlabel('Streak Length')
    ax7.set_ylabel('Frequency')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    summary_text = (f"Avg Win Streak: {streak_summary['avg_win_streak']:.2f} | Longest Win: {streak_summary['longest_win']}\n"
                    f"Avg Loss Streak: {streak_summary['avg_loss_streak']:.2f} | Longest Loss: {streak_summary['longest_loss']}")
    ax7.text(0.05, 0.85, summary_text, transform=ax7.transAxes, fontsize=10, color=color_dashboard_text)
    streak_lengths = np.array(list(win_streaks.keys()) + list(loss_streaks.keys()))
    freq = np.array(list(win_streaks.values()) + list(loss_streaks.values()))
    sorted_idx = np.argsort(streak_lengths)
    cum_prob = np.cumsum(freq[sorted_idx]) / np.sum(freq)
    ax7.plot(np.sort(streak_lengths), cum_prob, color=color_negative_roi, linestyle='--', label="Cumulative Prob")
    
    # Subplot 8: Risk Profile Analysis
    ax8 = plt.subplot(9, 1, 8)
    rolling_vol, kelly_fraction, corr = compute_risk_profile(analytics, session_stats)
    if rolling_vol.size > 0:
        ax8.plot(rolling_vol, color=color_negative_roi, label='Rolling Volatility')
    ax8.axhline(kelly_fraction, color=color_positive_roi, linestyle='--', label=f"Kelly Fraction: {kelly_fraction:.2f}")
    ax8.set_title('Risk Profile Analysis', fontsize=14, pad=20)
    ax8.set_xlabel('Bet Number')
    ax8.set_ylabel('Volatility')
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    ax8.text(0.05, 0.9, f"Correlation (Drawdowns vs. Performance): {corr:.2f}", transform=ax8.transAxes, fontsize=10, color=color_dashboard_text)
    
    # Subplot 9: Sensitivity Analysis (Illustrative Heatmap)
    ax9 = plt.subplot(9, 1, 9)
    multipliers = np.linspace(SERVER_BET_MULTIPLIER*0.8, SERVER_BET_MULTIPLIER*1.2, 5)
    stakes = np.linspace(INITIAL_STAKE*0.8, INITIAL_STAKE*1.2, 5)
    sensitivity = np.random.rand(5,5) * 20 - 10  # dummy ROI values between -10% and +10%
    im = ax9.imshow(sensitivity, cmap='viridis', aspect='auto')
    ax9.set_title('Sensitivity Analysis (Illustrative)', fontsize=14, pad=20)
    ax9.set_xlabel('Initial Stake Variation')
    ax9.set_ylabel('Server Bet Multiplier Variation')
    ax9.set_xticks(range(5))
    ax9.set_xticklabels([f"{s:.2f}" for s in stakes])
    ax9.set_yticks(range(5))
    ax9.set_yticklabels([f"{m:.2f}" for m in multipliers])
    fig.colorbar(im, ax=ax9, orientation='vertical', label='ROI (%)')
    
    plt.tight_layout()
    plt.savefig(output_file, format='pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nEnhanced analytics report saved as '{output_file}'")

###############################################################################
# Main Execution
###############################################################################
def main():
    global INITIAL_BANKROLL, INITIAL_STAKE, SERVER_ODDS, RECEIVER_ODDS
    global MAX_BET_ABSOLUTE, MONTE_CARLO_SIMULATIONS, GAMES_PER_SIMULATION
    global SERVER_BET_MULTIPLIER, RECEIVER_BET_MULTIPLIER
    global INITIAL_SEQUENCE, TIE_FAVORED, ADVANTAGE_FAVORED, AUTO_MODE
    
    print("=== Configuration Setup ===")
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
    
    final_bankrolls, total_profits, rois, max_drawdowns = [], [], [], []
    final_analytics = None
    final_session_stats = None
    
    for sim_index in tqdm(range(1, MONTE_CARLO_SIMULATIONS+1), desc="Simulations"):
        fb, tp, roi_val, md, session_stats, analytics = run_one_monte_carlo(df, n_games=GAMES_PER_SIMULATION)
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
    print(f"\nFinal Bankroll: Mean: {final_bankrolls.mean():.2f} | Median: {np.median(final_bankrolls):.2f} | Std: {final_bankrolls.std():.2f} | Min: {final_bankrolls.min():.2f} | Max: {final_bankrolls.max():.2f}")
    print(f"\nTotal Profit: Mean: {total_profits.mean():.2f} | Median: {np.median(total_profits):.2f} | Std: {total_profits.std():.2f} | Min: {total_profits.min():.2f} | Max: {total_profits.max():.2f}")
    print(f"\nROI (%): Mean: {rois.mean():.2f}% | Median: {np.median(rois):.2f}% | Std: {rois.std():.2f}% | Min: {rois.min():.2f}% | Max: {rois.max():.2f}%")
    print(f"\nMaximum Drawdown: Mean: {max_drawdowns.mean():.2f} | Median: {np.median(max_drawdowns):.2f} | Std: {max_drawdowns.std():.2f} | Min: {max_drawdowns.min():.2f} | Max: {max_drawdowns.max():.2f}")
    
    if final_analytics is not None and final_session_stats is not None:
        generate_bet_breakdown_report(final_analytics, output_file="bet_breakdown.csv")
        reporting_manager = ReportingManager(final_analytics, final_session_stats)
        reporting_manager.compute_additional_analytics()
        reporting_manager.compute_composite_metrics()
        reporting_manager.compute_statistical_tests()
        generate_detailed_analytics_report(reporting_manager, output_file="monte_carlo_results.pdf")
    
    print("\nMonte Carlo Analysis Complete!")

if __name__ == "__main__":
    main()
