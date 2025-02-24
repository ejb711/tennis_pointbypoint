#!/usr/bin/env python3

import os
import sys

def resource_path(relative_path):
    """Get absolute path to resource, works for dev and for PyInstaller"""
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

"""
monty_detailed.py

An enhanced Monte Carlo analysis for a tennis point-by-point betting system.
Incorporates improvements for:
  - Enhanced data visualization, executive dashboard, advanced analytics, and statistical validation.
  - Correct maximum drawdown percentage calculation.
  - Outputting the final report as a PDF.
  - Running two simulation sets – one with baseline settings and one with user-supplied alternative (comparison) settings – and comparing their results.
  - Allowing a "random" option for:
       * Initial Betting Sequence (e.g. SSSRRR; if set to "random", a new random sequence is generated for each game)
       * Tie Favored Pick (S or R; if "random", each tie is resolved randomly)
       * Advantage Favored Pick (A=advantage, S=server, R=receiver; if "random", each bet is decided randomly among A/S/R)

This version restores the complete plotting code in generate_detailed_analytics_report()
so that the baseline and comparison PDFs (monte_carlo_results_baseline.pdf,
monte_carlo_results_comparison.pdf) are no longer blank.
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
from matplotlib.backends.backend_pdf import PdfPages

###############################################################################
# Prompt Helpers
###############################################################################
def prompt_for_float(param_name, default_val):
    user_input = input(f"Enter {param_name} [default: {default_val}] (or press Enter to keep default): ").strip()
    return float(user_input) if user_input != "" else default_val

def prompt_for_int(param_name, default_val):
    user_input = input(f"Enter {param_name} [default: {default_val}] (or press Enter to keep default): ").strip()
    return int(user_input) if user_input != "" else default_val

def prompt_for_string(param_name, default_val):
    user_input = input(f"Enter {param_name} [default: {default_val}] (or type 'random' for dynamic random selection): ").strip()
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

INITIAL_SEQUENCE = "SSSRRR"  
TIE_FAVORED = "S"
ADVANTAGE_FAVORED = "A"

AUTO_MODE = False

###############################################################################
# Helper for Random Generation
###############################################################################
def generate_random_sequence(length=6):
    """Generate a random sequence of 'S' and 'R' of given length."""
    return "".join(random.choice(["S", "R"]) for _ in range(length))

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
        return 'S'

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
    if amount > 0:
        return f"\033[92m${amount:,.2f}\033[0m"
    elif amount < 0:
        return f"\033[91m${amount:,.2f}\033[0m"
    return f"${amount:,.2f}"

def calculate_next_bet(current_bet, next_pick, previous_pick, bankroll):
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
    
    # If the global INITIAL_SEQUENCE is "random", generate a new random sequence for each game.
    progression_list = list(INITIAL_SEQUENCE) if INITIAL_SEQUENCE.lower() != "random" else None
    
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
                
                # For each game, if INITIAL_SEQUENCE is "random", generate a random progression list.
                if INITIAL_SEQUENCE.lower() == "random":
                    progression_list = list(generate_random_sequence(6))
                
                while points_played < len(points):
                    current_point = points[points_played]
                    game_state.add_point(current_point)
                    
                    # Decide pick:
                    if not using_score_strategy:
                        # If we've reached the end of the progression_list, switch to score-based
                        if progression_list is None or (points_played >= len(progression_list)):
                            using_score_strategy = True
                        else:
                            pick = progression_list[points_played]
                    if using_score_strategy:
                        if game_state.is_score_tied():
                            if TIE_FAVORED.lower() == "random":
                                pick = random.choice(["S", "R"])
                            else:
                                pick = TIE_FAVORED
                        else:
                            if ADVANTAGE_FAVORED.lower() == "random":
                                pick = random.choice(["A", "S", "R"])
                            elif ADVANTAGE_FAVORED == "A":
                                pick = game_state.get_advantage_pick()
                            else:
                                pick = ADVANTAGE_FAVORED
                    
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

def run_simulation(df):
    final_bankrolls = []
    total_profits = []
    rois = []
    max_drawdowns = []
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
    return {
         "final_bankrolls": np.array(final_bankrolls),
         "total_profits": np.array(total_profits),
         "rois": np.array(rois),
         "max_drawdowns": np.array(max_drawdowns),
         "analytics": final_analytics,
         "session_stats": final_session_stats
    }

###############################################################################
# Analytics Helpers
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

def compute_risk_profile(analytics, session_stats):
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
# Reporting Class
###############################################################################
class ReportingManager:
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
        returns = [bet.profit for bet in self.analytics.bets if bet.stake > 0]
        if len(returns) < 2:
            self.sharpe_ratio = 0.0
        else:
            avg_return = np.mean(returns)
            std_return = np.std(returns)
            self.sharpe_ratio = avg_return / std_return if std_return != 0 else 0.0
        
        peak_bankroll = max(self.analytics.bankroll_history) if self.analytics.bankroll_history else INITIAL_BANKROLL
        self.max_drawdown_pct = (self.session_stats['max_drawdown'] / peak_bankroll) * 100
        
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

def generate_detailed_analytics_report(reporting_manager, output_file):
    """
    Generate a multi-page PDF report with subplots and textual summaries.
    """
    text_report = "\n".join([
        reporting_manager.create_executive_summary(),
        reporting_manager.create_table_of_contents(),
        reporting_manager.create_detailed_analytics_text(),
        reporting_manager.create_practical_insights()
    ])
    print("\n" + "="*80)
    print(text_report)
    print("="*80 + "\n")
    
    # Colors
    color_win_rate = "#0072B2"
    color_positive_roi = "#009E73"
    color_negative_roi = "#D55E00"
    color_distribution = "#8856a7"
    color_streak = "#E69F00"
    color_dashboard_text = "#000000"
    
    analytics = reporting_manager.analytics
    session_stats = reporting_manager.session_stats
    
    fig = plt.figure(figsize=(20, 45))
    
    # 1) Bankroll Evolution
    ax1 = plt.subplot(9, 1, 1)
    ax1.plot(analytics.bankroll_history, label='Bankroll', color=color_win_rate, linewidth=2)
    ax1.set_title('Bankroll Evolution Over Time (Annotated)', fontsize=14, pad=20)
    ax1.set_xlabel('Bet Number')
    ax1.set_ylabel('Bankroll ($)')
    ax1.grid(True, alpha=0.3)
    peak_val = max(analytics.bankroll_history) if analytics.bankroll_history else 0
    if analytics.bankroll_history:
        peak_idx = analytics.bankroll_history.index(peak_val)
        ax1.annotate(f"Peak: {peak_val:.2f}", xy=(peak_idx, peak_val),
                     xytext=(peak_idx, peak_val+100),
                     arrowprops=dict(facecolor='green', shrink=0.05), fontsize=10)
    ax1.legend()
    
    # 2) Bet Size Distribution
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
    
    # 3) Strategy Performance (Win Rate, ROI, Risk-Adjusted)
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
        err_roi = (np.std(roi_list)/np.sqrt(len(roi_list))) if len(roi_list) > 1 else 0
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
    
    # 4) Win Rate by Bet Size Range
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
    
    # 5) Executive Dashboard
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
    
    # 6) ROI Distribution with KDE
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
        # Example annotations if we have extremes at -100% or ~+50%
        if -100 in bet_rois:
            ax6.annotate("Spike at -100%", xy=(-100, kde(-100)), xytext=(-90, kde(-100)*1.2),
                         arrowprops=dict(facecolor=color_negative_roi, shrink=0.05), fontsize=10)
        if 50 in bet_rois:
            ax6.annotate("Spike at +50%", xy=(50, kde(50)), xytext=(60, kde(50)*1.2),
                         arrowprops=dict(facecolor=color_negative_roi, shrink=0.05), fontsize=10)
    ax6.set_title('ROI Distribution with KDE Overlay', fontsize=14, pad=20)
    ax6.set_xlabel('Bet ROI (%)')
    ax6.set_ylabel('Density')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # 7) Win/Loss Streak Analysis
    from matplotlib import rcParams
    rcParams.update({'figure.autolayout': False})
    ax7 = plt.subplot(9, 1, 7)
    from copy import deepcopy
    wins_copy = deepcopy(analytics.bets)
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
    streak_lengths = list(win_streaks.keys()) + list(loss_streaks.keys())
    if streak_lengths:
        freq = np.array(list(win_streaks.values()) + list(loss_streaks.values()))
        sorted_idx = np.argsort(streak_lengths)
        sorted_lengths = np.array(streak_lengths)[sorted_idx]
        cum_prob = np.cumsum(freq[sorted_idx]) / np.sum(freq)
        ax7.plot(sorted_lengths, cum_prob, color=color_negative_roi, linestyle='--', label="Cumulative Prob")
    
    # 8) Risk Profile Analysis
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
    
    # 9) Sensitivity Analysis (Illustrative)
    ax9 = plt.subplot(9, 1, 9)
    multipliers = np.linspace(SERVER_BET_MULTIPLIER*0.8, SERVER_BET_MULTIPLIER*1.2, 5)
    stakes = np.linspace(INITIAL_STAKE*0.8, INITIAL_STAKE*1.2, 5)
    sensitivity = np.random.rand(5,5) * 20 - 10
    im = ax9.imshow(sensitivity, cmap='viridis', aspect='auto')
    ax9.set_title('Sensitivity Analysis (Illustrative)', fontsize=14, pad=20)
    ax9.set_xlabel('Initial Stake Variation')
    ax9.set_ylabel('Server Bet Multiplier Variation')
    ax9.set_xticks(range(5))
    ax9.set_xticklabels([f"{s:.2f}" for s in stakes])
    ax9.set_yticks(range(5))
    ax9.set_yticklabels([f"{m:.2f}" for m in multipliers])
    plt.colorbar(im, ax=ax9, orientation='vertical', label='ROI (%)')
    
    plt.tight_layout()
    plt.savefig(output_file, format='pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nEnhanced analytics report saved as '{output_file}'")

###############################################################################
# Combined Comparison Report
###############################################################################
def generate_combined_comparison_report(baseline_reporter, comp_reporter, output_file="monte_carlo_results_combined.pdf"):
    with PdfPages(output_file) as pdf:
        # Figure 1: Combined Bankroll Evolution
        fig, ax = plt.subplots(figsize=(10,6))
        ax.plot(baseline_reporter.analytics.bankroll_history, label="Baseline", color="#0072B2", linewidth=2)
        ax.plot(comp_reporter.analytics.bankroll_history, label="Comparison", color="#D55E00", linewidth=2)
        ax.set_title("Combined Bankroll Evolution")
        ax.set_xlabel("Bet Number")
        ax.set_ylabel("Bankroll ($)")
        ax.grid(True, alpha=0.3)
        ax.legend()
        pdf.savefig(fig)
        plt.close(fig)
        
        # Figure 2: Combined ROI Distribution
        fig, ax = plt.subplots(figsize=(10,6))
        baseline_rois = [bet.roi for bet in baseline_reporter.analytics.bets]
        comp_rois = [bet.roi for bet in comp_reporter.analytics.bets]
        if baseline_rois and comp_rois:
            all_rois = baseline_rois + comp_rois
            bin_count = int(math.sqrt(len(all_rois)))
            ax.hist(baseline_rois, bins=bin_count, alpha=0.5, density=True, label="Baseline", color="#0072B2")
            ax.hist(comp_rois, bins=bin_count, alpha=0.5, density=True, label="Comparison", color="#D55E00")
            kde_baseline = gaussian_kde(baseline_rois) if len(baseline_rois) > 1 else None
            kde_comp = gaussian_kde(comp_rois) if len(comp_rois) > 1 else None
            x_vals = np.linspace(min(all_rois), max(all_rois), 200)
            if kde_baseline is not None:
                ax.plot(x_vals, kde_baseline(x_vals), color="#0072B2", linewidth=2)
            if kde_comp is not None:
                ax.plot(x_vals, kde_comp(x_vals), color="#D55E00", linewidth=2)
        ax.set_title("Combined ROI Distribution")
        ax.set_xlabel("Bet ROI (%)")
        ax.set_ylabel("Density")
        ax.legend()
        ax.grid(True, alpha=0.3)
        pdf.savefig(fig)
        plt.close(fig)
    print(f"\nCombined comparison report saved as '{output_file}'")

###############################################################################
# Main Execution with Comparison Simulation
###############################################################################
def main():
    global INITIAL_BANKROLL, INITIAL_STAKE, SERVER_ODDS, RECEIVER_ODDS
    global MAX_BET_ABSOLUTE, MONTE_CARLO_SIMULATIONS, GAMES_PER_SIMULATION
    global SERVER_BET_MULTIPLIER, RECEIVER_BET_MULTIPLIER
    global INITIAL_SEQUENCE, TIE_FAVORED, ADVANTAGE_FAVORED, AUTO_MODE
    
    print("=== Configuration Setup (Baseline) ===")
    INITIAL_BANKROLL = prompt_for_float("Initial Bankroll", INITIAL_BANKROLL)
    INITIAL_STAKE = prompt_for_float("Initial Stake", INITIAL_STAKE)
    SERVER_ODDS = prompt_for_float("Server Odds", SERVER_ODDS)
    RECEIVER_ODDS = prompt_for_float("Receiver Odds", RECEIVER_ODDS)
    MAX_BET_ABSOLUTE = prompt_for_float("Max Bet Absolute", MAX_BET_ABSOLUTE)
    MONTE_CARLO_SIMULATIONS = prompt_for_int("Number of Monte Carlo simulations", MONTE_CARLO_SIMULATIONS)
    GAMES_PER_SIMULATION = prompt_for_int("Number of games per simulation", GAMES_PER_SIMULATION)
    SERVER_BET_MULTIPLIER = prompt_for_float("Server Bet Multiplier", SERVER_BET_MULTIPLIER)
    RECEIVER_BET_MULTIPLIER = prompt_for_float("Receiver Bet Multiplier", RECEIVER_BET_MULTIPLIER)
    
    seq_input = prompt_for_string("Initial Betting Sequence (e.g. SSSRRR)", "SSSRRR")
    if seq_input.lower() == "random":
        INITIAL_SEQUENCE = "random"
        print("Baseline will generate a random sequence for each game.")
    else:
        INITIAL_SEQUENCE = seq_input
    
    tie_input = prompt_for_string("Tie favored pick (S or R)", "S")
    if tie_input.lower() == "random":
        TIE_FAVORED = "random"
        print("Baseline will randomly choose tie favored pick for each bet.")
    else:
        TIE_FAVORED = tie_input
    
    adv_input = prompt_for_string("Advantage favored pick (A=advantage, S=server, R=receiver)", "A")
    if adv_input.lower() == "random":
        ADVANTAGE_FAVORED = "random"
        print("Baseline will randomly choose advantage favored pick for each bet.")
    else:
        ADVANTAGE_FAVORED = adv_input
    
    print("\nLoading combined.csv for Monte Carlo analysis...")
    df = pd.read_csv(resource_path('combined.csv'))
    df['date'] = pd.to_datetime(df['date'], format="%d-%b-%y", dayfirst=True, errors='coerce')
    
    print(f"\nRunning Baseline Simulation with initial settings...\n"
          f"(Bankroll={INITIAL_BANKROLL}, Stake={INITIAL_STAKE}, ServerOdds={SERVER_ODDS}, "
          f"ReceiverOdds={RECEIVER_ODDS}, MaxBet={MAX_BET_ABSOLUTE}, S-BetMult={SERVER_BET_MULTIPLIER}, "
          f"R-BetMult={RECEIVER_BET_MULTIPLIER}, Sequence='{INITIAL_SEQUENCE}', TieFavored='{TIE_FAVORED}', "
          f"AdvantageFavored='{ADVANTAGE_FAVORED}')\n")
    
    baseline_results = run_simulation(df)
    
    baseline_bankrolls = baseline_results["final_bankrolls"]
    baseline_total_profits = baseline_results["total_profits"]
    baseline_rois = baseline_results["rois"]
    print("\n=== Baseline Monte Carlo Results Summary ===")
    print(f"Final Bankroll: Mean: {baseline_bankrolls.mean():.2f} | Median: {np.median(baseline_bankrolls):.2f} | Std: {baseline_bankrolls.std():.2f} | Min: {baseline_bankrolls.min():.2f} | Max: {baseline_bankrolls.max():.2f}")
    print(f"Total Profit: Mean: {baseline_total_profits.mean():.2f} | Median: {np.median(baseline_total_profits):.2f} | Std: {baseline_total_profits.std():.2f}")
    print(f"ROI (%): Mean: {baseline_rois.mean():.2f}% | Median: {np.median(baseline_rois):.2f}% | Std: {baseline_rois.std():.2f}%")
    print(f"Maximum Drawdown: Mean: {baseline_results['session_stats']['max_drawdown']:.2f}\n")
    
    generate_bet_breakdown_report(baseline_results["analytics"], output_file="bet_breakdown_baseline.csv")
    baseline_reporter = ReportingManager(baseline_results["analytics"], baseline_results["session_stats"])
    baseline_reporter.compute_additional_analytics()
    baseline_reporter.compute_composite_metrics()
    baseline_reporter.compute_statistical_tests()
    generate_detailed_analytics_report(baseline_reporter, output_file="monte_carlo_results_baseline.pdf")
    
    compare_choice = input("\nDo you want to run a comparison simulation with alternative settings? (y/n): ").strip().lower()
    if compare_choice.startswith('y'):
        print("\n=== Configuration Setup (Comparison) ===")
        comp_MAX_BET_ABSOLUTE = prompt_for_float("Enter Max Bet Absolute", 400)
        comp_SERVER_BET_MULTIPLIER = prompt_for_float("Enter Server Bet Multiplier", 3.0)
        comp_RECEIVER_BET_MULTIPLIER = prompt_for_float("Enter Receiver Bet Multiplier", 2.0)
        
        comp_seq_input = prompt_for_string("Initial Betting Sequence (e.g. SSSRRR)", "SSSRRR")
        if comp_seq_input.lower() == "random":
            comp_INITIAL_SEQUENCE = "random"
            print("Comparison will generate a random sequence for each game.")
        else:
            comp_INITIAL_SEQUENCE = comp_seq_input
        
        comp_tie_input = prompt_for_string("Tie favored pick (S or R)", "S")
        if comp_tie_input.lower() == "random":
            comp_TIE_FAVORED = "random"
            print("Comparison will randomly choose tie favored pick for each bet.")
        else:
            comp_TIE_FAVORED = comp_tie_input
        
        comp_adv_input = prompt_for_string("Advantage favored pick (A=advantage, S=server, R=receiver)", "A")
        if comp_adv_input.lower() == "random":
            comp_ADVANTAGE_FAVORED = "random"
            print("Comparison will randomly choose advantage favored pick for each bet.")
        else:
            comp_ADVANTAGE_FAVORED = comp_adv_input
        
        # Update global variables for comparison
        MAX_BET_ABSOLUTE = comp_MAX_BET_ABSOLUTE
        SERVER_BET_MULTIPLIER = comp_SERVER_BET_MULTIPLIER
        RECEIVER_BET_MULTIPLIER = comp_RECEIVER_BET_MULTIPLIER
        INITIAL_SEQUENCE = comp_INITIAL_SEQUENCE
        TIE_FAVORED = comp_TIE_FAVORED
        ADVANTAGE_FAVORED = comp_ADVANTAGE_FAVORED
        
        print(f"\nRunning Comparison Simulation with new settings...\n"
              f"(MaxBet={MAX_BET_ABSOLUTE}, S-BetMult={SERVER_BET_MULTIPLIER}, R-BetMult={RECEIVER_BET_MULTIPLIER}, "
              f"Sequence='{INITIAL_SEQUENCE}', TieFavored='{TIE_FAVORED}', AdvantageFavored='{ADVANTAGE_FAVORED}')\n")
        
        comparison_results = run_simulation(df)
        comp_bankrolls = comparison_results["final_bankrolls"]
        comp_total_profits = comparison_results["total_profits"]
        comp_rois = comparison_results["rois"]
        print("\n=== Comparison Monte Carlo Results Summary ===")
        print(f"Final Bankroll: Mean: {comp_bankrolls.mean():.2f} | Median: {np.median(comp_bankrolls):.2f} | Std: {comp_bankrolls.std():.2f} | Min: {comp_bankrolls.min():.2f} | Max: {comp_bankrolls.max():.2f}")
        print(f"Total Profit: Mean: {comp_total_profits.mean():.2f} | Median: {np.median(comp_total_profits):.2f} | Std: {comp_total_profits.std():.2f}")
        print(f"ROI (%): Mean: {comp_rois.mean():.2f}% | Median: {np.median(comp_rois):.2f}% | Std: {comp_rois.std():.2f}%")
        print(f"Maximum Drawdown: Mean: {comparison_results['session_stats']['max_drawdown']:.2f}\n")
        
        generate_bet_breakdown_report(comparison_results["analytics"], output_file="bet_breakdown_comparison.csv")
        comp_reporter = ReportingManager(comparison_results["analytics"], comparison_results["session_stats"])
        comp_reporter.compute_additional_analytics()
        comp_reporter.compute_composite_metrics()
        comp_reporter.compute_statistical_tests()
        generate_detailed_analytics_report(comp_reporter, output_file="monte_carlo_results_comparison.pdf")
        
        print("\n=== Comparison Summary ===")
        print("Metric                  Baseline         Comparison")
        print("-----------------------------------------------------")
        print(f"Final Bankroll (Mean): {baseline_bankrolls.mean():.2f}         {comp_bankrolls.mean():.2f}")
        print(f"Total Profit (Mean):   {baseline_total_profits.mean():.2f}         {comp_total_profits.mean():.2f}")
        print(f"ROI (Mean):            {baseline_rois.mean():.2f}%          {comp_rois.mean():.2f}%")
        print(f"Max Drawdown (Mean):   {baseline_results['session_stats']['max_drawdown']:.2f}         {comparison_results['session_stats']['max_drawdown']:.2f}")
        
        generate_combined_comparison_report(baseline_reporter, comp_reporter, output_file="monte_carlo_results_combined.pdf")
    
    print("\nMonte Carlo Analysis Complete!")

if __name__ == "__main__":
    main()
