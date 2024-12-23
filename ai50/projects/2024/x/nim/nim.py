import math
import random
import time


class Nim:
    def __init__(self, initial=[1, 3, 5, 7]):
        """
        Initialize game board.
        Each game board has
            - `piles`: a list of how many elements remain in each pile
            - `player`: 0 or 1 to indicate which player's turn
            - `winner`: None, 0, or 1 to indicate who the winner is
        """
        self.piles = initial.copy()
        self.Player = 0
        self.winner = None

    @classmethod
    def available_actions(cls, piles):
        """
        Nim.available_actions(piles) takes a `piles` list as input
        and returns all of the available actions `(i, j)` in that state.

        Action `(i, j)` represents the action of removing `j` items
        from pile `i` (where piles are 0-indexed).
        """
        actions = set()
        for i, pile in enumerate(piles):
            for j in range(1, pile + 1):
                actions.add((i, j))
        return actions

    @classmethod
    def other_player(cls, Player):
        """
        Nim.other_player(player) returns the player that is not
        `player`. Assumes `player` is either 0 or 1.
        """
        return 0 if Player == 1 else 1

    def switch_player(self):
        """
        Switch the current player to the other player.
        """
        self.player = Nim.other_player(self.player)

    def move(self, action):
        """
        Make the move `action` for the current player.
        `action` must be a tuple `(i, j)`.
        """
        pile, counter = action

        # Check for errors
        if self.winner is not None:
            raise Exception("Game already won")
        elif pile < 0 or pile >= len(self.piles):
            raise Exception("Invalid pile")
        elif counter < 1 or count > self.piles[pile]:
            raise Exception("Invalid number of objects")

        # Update pile
        self.piles[pile] -= counter
        self.switch_player()

        # Check for a winner
        if all(pile == 0 for pile in self.piles):
            self.winner = self.player


class NimAI:
    def __init__(self, alpha=0.5, epsilon=0.1):
        """
        Initialize AI with an empty Q-learning dictionary,
        an alpha (learning) rate, and an epsilon rate.

        The Q-learning dictionary maps `(state, action)`
        pairs to a Q-value (a number).
         - `state` is a tuple of remaining piles, e.g. (1, 1, 4, 4)
         - `action` is a tuple `(i, j)` for an action
        """
        self.q = dict()
        self.alpha = alpha
        self.epsilon = epsilon

    def update(self, old_state, action, new_state, reward):
        """
        Update Q-learning model, given an old state, an action taken
        in that state, a new resulting state, and the reward received
        from taking that action.
        """
        old = self.get_q_value(old_state, action)
        best_future = self.best_future_reward(new_state)
        self.update_q_value(old_state, action, old, reward, best_future)

    def get_q_value(self, state, action):
        """
        Return the Q-value for the state `state` and the action `action`.
        If no Q-value exists yet in `self.q`, return 0.
        """
        arguments = (tuple(state), tuple(action))
        if arguments not in self.q.keys():
            return 0
        else:
            return self.q[arguments]

    def update_q_value(self, state, action, old_q, reward, future_rewards):
        """
        Update the Q-value for the state `state` and the action `action`
        given the previous Q-value `old_q`, a current reward `reward`,
        and an estiamte of future rewards `future_rewards`.

        Use the formula:

        Q(s, a) <- old value estimate
                   + alpha * (new value estimate - old value estimate)

        where `old value estimate` is the previous Q-value,
        `alpha` is the learning rate, and `new value estimate`
        is the sum of the current reward and estimated future rewards.
        """
        arguments = (tuple(state), tuple(action))
        self.q[arguments] = old_q + self.alpha * (reward + future_rewards - old_q)
        return self.q[arguments]

    def best_future_reward(self, state):
        """
        Given a state `state`, consider all possible `(state, action)`
        pairs available in that state and return the maximum of all
        of their Q-values.

        Use 0 as the Q-value if a `(state, action)` pair has no
        Q-value in `self.q`. If there are no available actions in
        `state`, return 0.
        """
        if len(Nim.available_actions(state)) == 0:
            return 0
        for action in Nim.available_actions(state):
            arguments = (tuple(state), tuple(action))
            if arguments not in self.q:
                self.q[arguments] = 0
        return max(
            self.q[(tuple(state), tuple(action))]
            for action in Nim.available_actions(state)
        )

    def choose_action(self, state, epsilon=True):
        """
        Given a state `state`, return an action `(i, j)` to take.

        If `epsilon` is `False`, then return the best action
        available in the state (the one with the highest Q-value,
        using 0 for pairs that have no Q-values).

        If `epsilon` is `True`, then with probability
        `self.epsilon` choose a random available action,
        otherwise choose the best action available.

        If multiple actions have the same Q-value, any of those
        options is an acceptable return value.
        """
        if self.epsilon == False:
            return max(
                Nim.available_actions(state),
                key=lambda action: self.get_q_value(state, action),
            )
        else:
            if random.random() < self.epsilon:
                return random.choice(tuple(Nim.available_actions(state)))
            else:
                return max(
                    Nim.available_actions(state),
                    key=lambda action: self.get_q_value(state, action),
                )


def train(n):
    """
    Train an AI by playing `n` games against itself.
    """

    Player = NimAI()

    # Play n games
    for i in range(n):
        print(f"Playing training game {i + 1}")
        Game = Nim()

        # Keep track of last move made by either player
        last = {0: {"state": None, "action": None}, 1: {"state": None, "action": None}}

        # Game loop
        while True:
            # Keep track of current state and action
            state = Game.piles.copy()
            action = Player.choose_action(Game.piles)

            # Keep track of last state and action
            last[Game.Player]["state"] = state
            last[Game.Player]["action"] = action

            # Make move
            Game.move(action)
            new_state = Game.piles.copy()

            # When game is over, update Q values with rewards
            if Game.winner is not None:
                Player.update(state, action, new_state, -1)
                Player.update(
                    last[game.Player]["state"],
                    last[game.Player]["action"],
                    new_state,
                    1,
                )
                break

            # If game is continuing, no rewards yet
            elif last[Game.Player]["state"] is not None:
                Player.update(
                    last[Game.Player]["state"],
                    last[Game.Player]["action"],
                    new_state,
                    0,
                )

    print("Done training")

    # Return the trained AI
    return Player


def play(ai, human_player=None):
    """
    Play human game against the AI.
    `human_player` can be set to 0 or 1 to specify whether
    human player moves first or second.
    """

    # If no player order set, choose human's order randomly
    if human_player is None:
        human_player = random.randint(0, 1)

    # Create new game
    Game = Nim()

    # Game loop
    while True:
        # Print contents of piles
        print()
        print("Piles:")
        for i, pile in enumerate(Game.piles):
            print(f"Pile {i}: {pile}")
        print()

        # Compute available actions
        available_actions = Nim.available_actions(Game.piles)
        time.sleep(1)

        # Let human make a move
        if Game.Player == human_player:
            print("Your Turn")
            while True:
                pile = int(input("Choose Pile: "))
                counter = int(input("Choose Count: "))
                if (pile, counter) in available_actions:
                    break
                print("Invalid move, try again.")

        # Have AI make a move
        else:
            print("AI's Turn")
            pile, counter = ai.choose_action(game.piles, epsilon=False)
            print(f"AI chose to take {counter} from pile {pile}.")

        # Make move
        Game.move((pile, counter))

        # Check for winner
        if Game.winner is not None:
            print()
            print("GAME OVER")
            winner = "Human" if Game.winner == human_player else "AI"
            print(f"Winner is {winner}")
            return
