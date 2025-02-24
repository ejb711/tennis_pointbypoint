import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from datetime import datetime

# Global configuration
INITIAL_BANKROLL = 1500.0
SERVER_ODDS = -250
RECEIVER_ODDS = 135
MAX_BET_ABSOLUTE = 40000.0
AUTO_MODE = False

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

# Utility Functions for Betting System
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
        
    # Apply bet limits
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
    else:  # pick == 'R'
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
    Handles both positive and negative odds formats:
    - For -235 (server): $100 stake wins ~$42.55
    - For +145 (receiver): $100 stake wins $145
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

def check_user_input():
    """
    Handles user interaction during simulation.
    Supports manual progression, automatic completion, and early exit.
    """
    global AUTO_MODE
    if AUTO_MODE:
        return False
        
    user_input = input("\nPress Enter to continue, 'q' to quit, or 'finish_it' to auto-complete: ")
    if user_input.lower() == 'q':
        return True
    elif user_input.lower() == 'finish_it':
        print("\nSwitching to automatic completion mode...")
        AUTO_MODE = True
    return False

def display_bet_info(bet_number, match_info, game_str, current_point, points_played, 
                     stake, pick, bankroll, game_state, strategy_type):
    """
    Displays comprehensive information about the current bet situation.
    Shows game context, betting details, and financial status.
    """
    if AUTO_MODE:
        return
        
    print("\n" + "="*80)
    print(f"Bet #{bet_number}")
    print("="*80)
    
    print(f"\nMatch Information:")
    print(f"Tournament: {match_info['tournament']}")
    print(f"Date: {match_info['date']}")
    print(f"Server: {match_info['server1']} vs {match_info['server2']}")
    
    print(f"\nGame Information:")
    print(f"Full points sequence: {game_str}")
    print(f"Current score: {game_state.get_score_summary()}")
    print(f"Current point: {current_point} (Point {points_played} of this game)")
    print(f"Points played: {' '.join(game_state.points_sequence)}")
    
    print(f"\nBetting Information:")
    print(f"Strategy: {strategy_type}")
    print(f"Stake Amount: {format_currency(stake)}")
    print(f"Betting On: {'Server' if pick == 'S' else 'Receiver'}")
    print(f"Odds: {SERVER_ODDS if pick == 'S' else RECEIVER_ODDS}")
    
    print(f"\nBankroll Status:")
    print(f"Current Bankroll: {format_currency(bankroll)}")
    print(f"Maximum allowed bet: {format_currency(min(bankroll / 2, MAX_BET_ABSOLUTE))}")
    
    if not AUTO_MODE:
        input("\nPress Enter to see the result...")

def display_bet_result(won_bet, profit, bankroll, peak, max_drawdown, move_to_next_game):
    """
    Shows the outcome of a bet and updates key statistics.
    Provides clear feedback on financial impact and progression status.
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
    
    if not AUTO_MODE:
        input("\nPress Enter to continue...")

def simulate_interactive_betting(df, initial_stake=1):
    """
    Simulates betting on tennis points with detailed tracking and verification.
    
    This is the main simulation engine that:
    1. Processes matches chronologically
    2. Implements the S>S>S>R>R>R progression
    3. Tracks all bets and maintains detailed analytics
    4. Verifies financial calculations and statistics
    """
    # Initialize simulation state
    bankroll = INITIAL_BANKROLL
    peak = bankroll
    max_drawdown = 0
    bet_number = 0
    analytics = SessionAnalytics()
    
    # Initialize tracking statistics
    session_stats = {
        'matches_processed': 0,
        'total_games': 0,         # Valid games we process
        'actual_games': 0,        # Total actual tennis games
        'total_bets': 0,
        'wins': 0,
        'total_profit': 0,
        'total_stakes': 0,
        'progression_wins': 0,
        'score_based_wins': 0,
        'tiebreaks_skipped': 0
    }
    
    # Process matches in chronological order
    for _, match in df.sort_values('date').iterrows():
        if bankroll <= 0:
            print(f"\n\033[91mBankroll depleted during match: {match['tny_name']} on {match['date']}\033[0m")
            break
            
        session_stats['matches_processed'] += 1
        match_info = {
            'tournament': match['tny_name'],
            'date': match['date'],
            'server1': match['server1'],
            'server2': match['server2']
        }
        
        # Process each set in the match
        sets = [s.strip() for s in str(match['pbp']).split('.') if s.strip()]
        for set_str in sets:
            # Count all games for verification
            games = [g.strip() for g in set_str.split(';') if g.strip()]
            session_stats['actual_games'] += len(games)
            
            for game_str in games:
                # Skip tiebreaks but count them
                if is_tiebreak(game_str):
                    session_stats['tiebreaks_skipped'] += 1
                    continue
                    
                points = process_game_points(game_str)
                if not points:
                    continue
                
                # Count valid games we'll bet on
                session_stats['total_games'] += 1
                
                # Initialize game state and betting variables
                game_state = GameState()
                current_stake = initial_stake
                previous_pick = None
                points_played = 0
                using_score_strategy = False
                strategy_picks = ['S', 'S', 'S', 'R', 'R', 'R']
                current_sequence = 0
                
                # Process each point in the game
                while points_played < len(points):
                    current_point = points[points_played]
                    game_state.add_point(current_point)
                    
                    # Determine betting strategy
                    if not using_score_strategy:
                        if current_sequence >= len(strategy_picks):
                            using_score_strategy = True
                        else:
                            pick = strategy_picks[current_sequence]
                    
                    if using_score_strategy:
                        if game_state.is_score_tied():
                            pick = 'S'  # Bet on server when tied
                        else:
                            pick = game_state.get_advantage_pick()
                    
                    # Calculate stake using progression rules
                    if previous_pick is not None:
                        current_stake = calculate_next_bet(current_stake, pick, previous_pick, bankroll)
                    
                    stake = min(current_stake, bankroll / 2, MAX_BET_ABSOLUTE)
                    bet_number += 1
                    strategy_type = "Progression" if not using_score_strategy else "Score-Based"
                    
                    # Display current bet information
                    display_bet_info(bet_number, match_info, game_str, current_point,
                                     points_played + 1, stake, pick, bankroll,
                                     game_state, strategy_type)
                    
                    # Evaluate bet outcome
                    won_bet = evaluate_point(current_point, pick)
                    
                    # Update bet statistics first
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
                        
                        # Record the winning bet
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
                        
                        # Update peak and drawdown tracking
                        peak = max(peak, bankroll)
                        max_drawdown = max(max_drawdown, peak - bankroll)
                        
                        # Display result and move to next game
                        display_bet_result(True, profit, bankroll, peak, max_drawdown, True)
                        break  # Move to next game after a win
                    else:
                        profit = -stake
                        bankroll -= stake
                        session_stats['total_profit'] += profit
                        
                        # Record the losing bet
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
                        
                        # Update peak and drawdown tracking
                        peak = max(peak, bankroll)
                        max_drawdown = max(max_drawdown, peak - bankroll)
                        
                        # Display result for loss
                        display_bet_result(False, profit, bankroll, peak, max_drawdown, points_played >= len(points) - 1)
                    
                    previous_pick = pick
                    points_played += 1
                    
                    if check_user_input():
                        return bankroll, peak, max_drawdown, session_stats, analytics

    # Verify final statistics
    analytics_verification = analytics.verify_statistics()
    if analytics_verification['total_bets_match']:
        print("\nStatistics verification passed.")
    else:
        print("\nWarning: Statistics verification failed - inconsistency detected.")
        
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
        
        # 2. Bet Size Distribution with unique bin edges
        plt.subplot(5, 1, 2)
        bins = np.linspace(min(analytics.bet_sizes), max(analytics.bet_sizes), 51)
        plt.hist(analytics.bet_sizes, bins=bins, color='green', alpha=0.7)
        mean_bet = np.mean(analytics.bet_sizes)
        plt.axvline(mean_bet, color='red', linestyle='--', label=f'Mean Bet: ${mean_bet:,.2f}')
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
        
        # Calculate Win Rate and ROI for each strategy
        win_rates = [
            (analytics.strategy_outcomes[s]['wins'] / analytics.strategy_outcomes[s]['total'] * 100)
            if analytics.strategy_outcomes[s]['total'] > 0 else 0
            for s in strategies
        ]
        
        rois = [
            (analytics.strategy_outcomes[s]['profit'] / 
             sum(bet.stake for bet in analytics.bets if bet.strategy_type == s) * 100)
            if sum(bet.stake for bet in analytics.bets if bet.strategy_type == s) > 0 else 0
            for s in strategies
        ]
        
        plt.bar(x - width/2, win_rates, width, label='Win Rate %', color='blue')
        plt.bar(x + width/2, rois, width, label='ROI %', color='green')
        
        plt.title('Strategy Performance Metrics', fontsize=14, pad=20)
        plt.xticks(x, strategies)
        plt.ylabel('Percentage')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # 4. Win Rate by Bet Size Range
        plt.subplot(5, 1, 4)
        bet_data = pd.DataFrame({
            'stake': [bet.stake for bet in analytics.bets],
            'won': [bet.won for bet in analytics.bets]
        })
        # Create deciles, dropping duplicates if needed
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
            f"Total Games Processed: {session_stats['total_games']:,}\n"
            f"Total Bets Placed: {session_stats['total_bets']:,}\n"
            f"Overall Win Rate: {(session_stats['wins']/session_stats['total_bets']*100):.1f}%\n"
            f"Total Amount Staked: ${session_stats['total_stakes']:,.2f}\n"
            f"Net Profit/Loss: ${session_stats['total_profit']:,.2f}\n"
            f"ROI: {(session_stats['total_profit']/session_stats['total_stakes']*100):.1f}%\n\n"
        )
        
        if analytics.max_bet:
            stats_text += (
                f"Maximum Bet Details:\n"
                f"Amount: ${analytics.max_bet.stake:,.2f}\n"
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
        
        print(f"\nDetailed analytics report saved as '{output_file}'")
        
    except Exception as e:
        print(f"\nWarning: Could not generate visual analytics due to: {str(e)}")
        print("Continuing with text-based results...")

def main():
    """
    Main execution function that handles data loading, simulation execution,
    and comprehensive results reporting.
    """
    print("Loading and processing tennis match data...")
    # Updated to handle multiple possible date formats
    df = pd.read_csv('combined.csv')
    df['date'] = pd.to_datetime(df['date'], infer_datetime_format=True, dayfirst=True, errors='coerce')
    
    print("\nBetting System Configuration:")
    print("-----------------------------")
    print(f"Initial Bankroll: ${INITIAL_BANKROLL:,.2f}")
    print(f"Server Odds: {SERVER_ODDS}")
    print(f"Receiver Odds: {RECEIVER_ODDS}")
    print(f"Maximum Single Bet: Lesser of 1/2 bankroll or ${MAX_BET_ABSOLUTE:,.2f}")
    
    print("\nProgression Rules:")
    print("- Start with $1 bet")
    print("- Triple previous bet when betting on Server")
    print("- Double previous bet when betting on Receiver")
    print("- Progress through S>S>S>R>R>R sequence")
    print("- Skip all tiebreak games")
    
    print("\nScore-Based Strategy (after progression):")
    print("- Bet on server when score is tied")
    print("- Bet on player with point advantage")
    
    print("\nInteraction Options:")
    print("- Press Enter to process each bet")
    print("- Type 'q' to quit at any time")
    print("- Type 'finish_it' to complete simulation automatically")
    
    input("\nPress Enter to start betting simulation...")
    
    # Run simulation
    results = simulate_interactive_betting(df)
    final_bankroll, peak, max_drawdown, session_stats, analytics = results
    
    # Generate comprehensive reports
    print("\n=== Final Session Results ===")
    
    print(f"\nActivity Overview:")
    print(f"Matches Processed: {session_stats['matches_processed']:,}")
    print(f"Total Tennis Games: {session_stats['actual_games']:,}")
    print(f"Games Bet On: {session_stats['total_games']:,}")
    print(f"Tiebreaks Skipped: {session_stats['tiebreaks_skipped']:,}")
    print(f"Total Bets Placed: {session_stats['total_bets']:,}")
    
    print(f"\nBetting Results:")
    print(f"Total Wins: {session_stats['wins']:,}")
    print(f"  - Progression Sequence Wins: {session_stats['progression_wins']:,}")
    print(f"  - Score-Based Strategy Wins: {session_stats['score_based_wins']:,}")
    print(f"Overall Win Rate: {(session_stats['wins']/session_stats['total_bets']*100):.1f}%")
    
    print(f"\nFinancial Summary:")
    print(f"Total Amount Staked: {format_currency(session_stats['total_stakes'])}")
    print(f"Net Profit/Loss: {format_currency(session_stats['total_profit'])}")
    print(f"ROI: {(session_stats['total_profit']/session_stats['total_stakes']*100):.1f}%")
    
    print(f"\nBankroll Statistics:")
    print(f"Starting Bankroll: {format_currency(INITIAL_BANKROLL)}")
    print(f"Final Bankroll: {format_currency(final_bankroll)}")
    print(f"Peak Bankroll: {format_currency(peak)}")
    print(f"Maximum Drawdown: {format_currency(max_drawdown)}")
    
    # Generate and verify analytics
    generate_detailed_analytics_report(analytics, session_stats)
    analytics_verification = analytics.verify_statistics()
    
    # Show verification results
    print("\nStatistics Verification:")
    print(f"Total Bets Match: {'✓' if analytics_verification['total_bets_match'] else '✗'}")
    print(f"Bankroll Tracking Consistent: {'✓' if analytics_verification['bankroll_consistent'] else '✗'}")
    
    expected_bankroll = INITIAL_BANKROLL + analytics_verification['total_profit']
    if abs(expected_bankroll - final_bankroll) > 0.01:
        print(f"\nWarning: Bankroll tracking discrepancy detected!")
        print(f"Expected final bankroll: {format_currency(expected_bankroll)}")
        print(f"Actual final bankroll: {format_currency(final_bankroll)}")
    
    print("\nSimulation Complete!")

if __name__ == "__main__":
    main()
