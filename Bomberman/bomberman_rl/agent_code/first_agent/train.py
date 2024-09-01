import numpy as np
import events as e
import pickle
import time
import os
from datetime import datetime

from .callbacks import state_to_features, ACTIONS

LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.99


def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # Set the logger if it doesn't exist
    if not hasattr(self, 'logger'):
        self.logger = logging.getLogger('default_logger')
        print("If not has attr, this will be printed")

    self.transitions = []

    # Keep track of training data for analysis later:
    # 1. Define self here
    self.training_start_time = time.time() # Record start time
    self.training_start_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S") # Record start timestamp
    self.round_counter = 0 # Initialize round counter
    self.total_reward = 0
    self.event_log = [] # Log events

    # Action counter
    self.action_counts = {action: 0 for action in ACTIONS}


def game_events_occurred(self, old_game_state, self_action, new_game_state, events):
    """
    Called once per step to allow intermediate rewards based on game events.

    When this method is called, self.events will contain a list of all game
    events relevant to your agent that occurred during the previous step. Consult
    settings.py to see what events are tracked. You can hand out rewards to your
    agent based on these events and your knowledge of the (new) game state.

    This is *one* of the places where you could update your agent.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param old_game_state: The state that was passed to the last call of `act`.
    :param self_action: The action that you took.
    :param new_game_state: The state the agent is in now.
    :param events: The events that occurred when going from  `old_game_state` to `new_game_state`
    """
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')
    
    self.action_counts[self_action] += 1

    if old_game_state is None or new_game_state is None:
        return

    old_features = state_to_features(old_game_state)
    new_features = state_to_features(new_game_state)

    reward = reward_from_events(events)
    self.total_reward += reward

    self.event_log.extend(events)

    # Q-update
    if old_features not in self.q_table:
        self.q_table[old_features] = np.zeros(len(ACTIONS))
    if new_features not in self.q_table:
        self.q_table[new_features] = np.zeros(len(ACTIONS))

    old_q_value = self.q_table[old_features][ACTIONS.index(self_action)]
    max_future_q = np.max(self.q_table[new_features])
    self.q_table[old_features][ACTIONS.index(self_action)] = old_q_value + LEARNING_RATE * (reward + DISCOUNT_FACTOR * max_future_q - old_q_value)


def end_of_round(self, last_game_state, last_action, events):
    """
    Called at the end of each game or when the agent died to hand out final rewards.
    This replaces game_events_occurred in this round.

    This is similar to game_events_occurred. self.events will contain all events that
    occurred during your agent's final step.

    This is *one* of the places where you could update your agent.
    This is also a good place to store an agent that you updated.

    :param self: The same object that is passed to all of your callbacks.
    """
    self.logger.debug(f"End of round: events occurred: {' '.join(map(str, events))}")
    last_state = state_to_features(last_game_state)
    reward = reward_from_events(events)
    # Track total reward
    self.total_reward += reward

    if last_state not in self.q_table:
        self.q_table[last_state] = np.zeros(len(ACTIONS))

    old_q_value = self.q_table[last_state][ACTIONS.index(last_action)]
    self.q_table[last_state][ACTIONS.index(last_action)] = old_q_value + LEARNING_RATE * (reward - old_q_value)

    with open("q_table.pkl", "wb") as file:
        pickle.dump(self.q_table, file)

    # Q-table size
    self.logger.info(f"Q-table size after round: {len(self.q_table)} entries")

    # Track statistics
    # Round counter
    self.round_counter += 1
    self.event_log.extend(events)


def reward_from_events(events) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        e.COIN_COLLECTED: 50,
        e.INVALID_ACTION: -5,
        e.MOVED_RIGHT: -1,
        e.MOVED_LEFT: -1,
        e.MOVED_UP: -1,
        e.MOVED_DOWN: -1,
        e.WAITED: -1,
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    return reward_sum
