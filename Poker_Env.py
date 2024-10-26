import random
from collections import Counter
from typing import List, Tuple, Dict, Optional

class PokerEnv:
    def __init__(self, num_players: int = 2, starting_chips: int = 1000, small_blind: int = 10):
        self.num_players = num_players
        self.starting_chips = starting_chips
        self.small_blind = small_blind
        self.big_blind = small_blind * 2
        self.deck = self.create_deck()
        self.players = self.initialize_players()
        self.community_cards: List[Tuple[str, str]] = []
        self.pot = 0
        self.current_bets = [0] * num_players
        self.round = 0
        self.max_rounds = 4  # Pre-Flop, Flop, Turn, River
        self.game_active = True
        self.button_pos = 0  # Dealer button position
        self.current_player = 0
        self.min_raise = self.big_blind

    def create_deck(self) -> List[Tuple[str, str]]:
        suits = ['♥', '♦', '♣', '♠']  # Using unicode symbols for better readability
        ranks = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
        deck = [(rank, suit) for suit in suits for rank in ranks]
        random.shuffle(deck)
        return deck

    def initialize_players(self) -> List[Dict]:
        return [{'hand': [], 
                'chips': self.starting_chips,
                'bet': 0,
                'status': 'active',  # active, folded, all-in
                'position': i} for i in range(self.num_players)]

    def post_blinds(self):
        sb_pos = (self.button_pos + 1) % self.num_players
        bb_pos = (self.button_pos + 2) % self.num_players
        
        # Post small blind
        self.players[sb_pos]['chips'] -= self.small_blind
        self.players[sb_pos]['bet'] = self.small_blind
        self.pot += self.small_blind
        
        # Post big blind
        self.players[bb_pos]['chips'] -= self.big_blind
        self.players[bb_pos]['bet'] = self.big_blind
        self.pot += self.big_blind
        
        self.current_player = (bb_pos + 1) % self.num_players
        self.min_raise = self.big_blind

    def get_hand_rank_value(self, rank: str) -> int:
        """Convert card ranks to numerical values for comparison."""
        rank_values = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8,
                      '9': 9, '10': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14}
        return rank_values[rank]

    def evaluate_hand(self, player: Dict) -> Tuple[int, str, List[int]]:
        """Enhanced hand evaluation with kickers."""
        all_cards = player['hand'] + self.community_cards
        ranks = [card[0] for card in all_cards]
        suits = [card[1] for card in all_cards]
        
        # Convert ranks to values for easier comparison
        rank_values = [self.get_hand_rank_value(r) for r in ranks]
        rank_counts = Counter(rank_values)
        suit_counts = Counter(suits)
        
        # Check for straight flush and royal flush
        for suit in suits:
            suited_cards = [self.get_hand_rank_value(card[0]) 
                          for card in all_cards if card[1] == suit]
            if len(suited_cards) >= 5:
                suited_cards.sort(reverse=True)
                for i in range(len(suited_cards) - 4):
                    if all(suited_cards[j] == suited_cards[j+1] + 1 
                          for j in range(i, i + 4)):
                        straight_flush_value = suited_cards[i]
                        if straight_flush_value == 14:  # Ace-high straight flush = royal flush
                            return (10, 'Royal Flush', [14, 13, 12, 11, 10])
                        return (9, 'Straight Flush', suited_cards[i:i+5])
        
        # Four of a kind
        if 4 in rank_counts.values():
            quads = max(r for r in rank_counts if rank_counts[r] == 4)
            kicker = max(r for r in rank_counts if r != quads)
            return (8, 'Four of a Kind', [quads, kicker])
        
        # Full house
        if 3 in rank_counts.values() and 2 in rank_counts.values():
            trips = max(r for r in rank_counts if rank_counts[r] == 3)
            pair = max(r for r in rank_counts if rank_counts[r] == 2)
            return (7, 'Full House', [trips, pair])
        
        # Flush
        if max(suit_counts.values()) >= 5:
            flush_suit = max(suit_counts.items(), key=lambda x: x[1])[0]
            flush_cards = sorted([self.get_hand_rank_value(card[0]) 
                                for card in all_cards if card[1] == flush_suit],
                               reverse=True)[:5]
            return (6, 'Flush', flush_cards)
        
        # Straight
        sorted_ranks = sorted(set(rank_values), reverse=True)
        for i in range(len(sorted_ranks) - 4):
            if all(sorted_ranks[j] == sorted_ranks[j+1] + 1 for j in range(i, i + 4)):
                return (5, 'Straight', sorted_ranks[i:i+5])
        
        # Three of a kind
        if 3 in rank_counts.values():
            trips = max(r for r in rank_counts if rank_counts[r] == 3)
            kickers = sorted([r for r in rank_counts if r != trips], 
                           reverse=True)[:2]
            return (4, 'Three of a Kind', [trips] + kickers)
        
        # Two pair
        pairs = [r for r in rank_counts if rank_counts[r] == 2]
        if len(pairs) >= 2:
            pairs.sort(reverse=True)
            kicker = max(r for r in rank_counts if rank_counts[r] == 1)
            return (3, 'Two Pair', pairs[:2] + [kicker])
        
        # One pair
        if 2 in rank_counts.values():
            pair = max(r for r in rank_counts if rank_counts[r] == 2)
            kickers = sorted([r for r in rank_counts if r != pair], 
                           reverse=True)[:3]
            return (2, 'One Pair', [pair] + kickers)
        
        # High card
        high_cards = sorted(rank_values, reverse=True)[:5]
        return (1, 'High Card', high_cards)

    def calculate_pot_odds(self, player: Dict) -> float:
        """Calculate pot odds for a player."""
        call_amount = max(self.current_bets) - player['bet']
        if call_amount == 0:
            return 1.0
        return self.pot / (self.pot + call_amount)

    def calculate_expected_value(self, player: Dict) -> float:
        """Calculate expected value of a call based on hand strength and pot odds."""
        hand_strength, _, _ = self.evaluate_hand(player)
        pot_odds = self.calculate_pot_odds(player)
        
        # Normalize hand strength to a probability
        win_probability = (hand_strength - 1) / 9  # 1-10 scale normalized to 0-1
        
        # Basic Kelly Criterion calculation
        edge = win_probability - (1 - win_probability)
        return edge * pot_odds

    def get_valid_actions(self, player: Dict) -> List[str]:
        """Get list of valid actions for a player."""
        if player['status'] != 'active':
            return []
            
        actions = ['fold']
        call_amount = max(self.current_bets) - player['bet']
        
        if call_amount == 0:
            actions.append('check')
        elif call_amount <= player['chips']:
            actions.append('call')
            
        if player['chips'] > call_amount + self.min_raise:
            actions.append('raise')
            
        return actions

    def apply_action(self, player_idx: int, action: str, amount: Optional[int] = None) -> bool:
        """Apply a player's action to the game state."""
        player = self.players[player_idx]
        valid_actions = self.get_valid_actions(player)
        
        if action not in valid_actions:
            return False
            
        if action == 'fold':
            player['status'] = 'folded'
        elif action == 'call':
            call_amount = max(self.current_bets) - player['bet']
            if call_amount > player['chips']:
                call_amount = player['chips']
                player['status'] = 'all-in'
            player['chips'] -= call_amount
            player['bet'] += call_amount
            self.pot += call_amount
        elif action == 'raise':
            if amount is None or amount < self.min_raise:
                amount = self.min_raise
            if amount > player['chips']:
                return False
            player['chips'] -= amount
            player['bet'] += amount
            self.pot += amount
            self.min_raise = amount
            
        self.current_player = (self.current_player + 1) % self.num_players
        return True

    def play_round(self):
        """Play a complete round of poker."""
        if not self.game_active:
            return

        # Deal hole cards and post blinds
        self.deal_hands()
        self.post_blinds()
        
        # Print initial state
        print("\n=== New Hand ===")
        print(f"Pot: ${self.pot}")
        print(f"Blinds: ${self.small_blind}/{self.big_blind}")
        
        # Play each street
        street_names = ['Pre-Flop', 'Flop', 'Turn', 'River']
        for street in range(self.max_rounds):
            print(f"\n=== {street_names[street]} ===")
            self.deal_community_cards()
            self.betting_round()
            
            if not self.game_active:
                break
        
        # Showdown if game is still active
        if self.game_active:
            self.showdown()
        
        # Move button and reset for next hand
        self.button_pos = (self.button_pos + 1) % self.num_players
        self.reset_round()

    def showdown(self):
        """Handle the showdown phase of the hand."""
        active_players = [i for i, p in enumerate(self.players) 
                         if p['status'] in ['active', 'all-in']]
        
        if len(active_players) < 2:
            winner = active_players[0]
        else:
            hands = [(i, self.evaluate_hand(self.players[i])) 
                    for i in active_players]
            winner = max(hands, key=lambda x: (x[1][0], x[1][2]))[0]
        
        print(f"\n=== Showdown ===")
        print(f"Community cards: {self.format_cards(self.community_cards)}")
        for i in active_players:
            hand_value = self.evaluate_hand(self.players[i])
            print(f"Player {i + 1}: {self.format_cards(self.players[i]['hand'])} - {hand_value[1]}")
        
        print(f"\nPlayer {winner + 1} wins ${self.pot}!")
        self.players[winner]['chips'] += self.pot

    def reset_round(self):
        """Reset the game state for a new round."""
        self.deck = self.create_deck()
        self.community_cards = []
        self.pot = 0
        self.current_bets = [0] * self.num_players
        self.round = 0
        self.min_raise = self.big_blind
        for player in self.players:
            player['hand'] = []
            player['bet'] = 0
            player['status'] = 'active' if player['chips'] > 0 else 'out'

    @staticmethod
    def format_cards(cards: List[Tuple[str, str]]) -> str:
        """Format cards for display."""
        return ' '.join(f"{rank}{suit}" for rank, suit in cards)

# Example usage
if __name__ == "__main__":
    # Create and run a sample game
    env = PokerEnv(num_players=4, starting_chips=1000, small_blind=10)
    
    for _ in range(3):  # Play 3 hands
        env.play_round()
        print("\nChip counts:")
        for i, player in enumerate(env.players):
            print(f"Player {i + 1}: ${player['chips']}")
        print("\n" + "="*50 + "\n")