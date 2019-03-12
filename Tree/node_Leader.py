import numpy as np

from BeTr_Zerg import BTZSmartSelector, BTZN
from pysc2.RL_algos.DQN import DQNModel
from pysc2.lib import actions, features, units


class King_NN(BTZSmartSelector):

    # actions: 5
    def decide(self):
        self.current_state = self._process_blackboard()

        if self.previous_action is not None:
            self.learn()

        action = self.model.act(self.current_state)

        self.decision = action
        self.previous_action = action
        self.previous_state = self.current_state

    def learn(self):
        if BTZN().blackboard["obs"].last():
            reward = BTZN().blackboard["obs"].reward
            self.model.learn(self.previous_state, self.previous_action, reward, 'terminal', True)
            self.model.save_model(self.data_file)
        else:
            reward = -0.0001 if BTZN().blackboard["action"] == actions.FUNCTIONS.no_op() else 0
            self.model.learn(self.previous_state, self.previous_action, reward, self.current_state)

    def _process_blackboard(self):
        enemy_locs = (BTZN().blackboard["obs"].observation["feature_minimap"].player_relative == 4).flatten()

        army_locs = BTZN().blackboard["obs"].observation["feature_minimap"].selected.flatten()

        troops = np.zeros(5)
        ind = 0
        for key in sorted(BTZN().blackboard["troops"].iterkeys()):
            troops[ind] = BTZN().blackboard["troops"][key]
            ind += 1

        enemy_units = np.zeros(178)
        ind = 0
        for key in sorted(BTZN().blackboard["enemy_units"].iterkeys()):
            enemy_units[ind] = BTZN().blackboard["enemy_units"][key]
            ind += 1

        last_seen = np.zeros(150)
        ind = 0
        for tuple in BTZN().blackboard["last_seen"]:
            last_seen[ind] = tuple[1]
            last_seen[ind + 1] = tuple[0][0]
            last_seen[ind + 2] = tuple[0][1]
            ind += 3

        free_supply = BTZN().blackboard["free_supply"]



        # total state dimension: 64*64 + 64*64 + 5 + 178 + 150 + 1 = 8525
        state = np.concatenate((enemy_locs, army_locs, troops, enemy_units, last_seen), axis=0)

        return state

    def setup(self):
        self.model = DQNModel(state_size=self.no_of_inputs, action_size=self.no_of_actions, architecture="AlphaZero")
        self.model.load_model(self.data_file)

    def __init__(self, decendant):
        self.no_of_inputs = 8525
        self.no_of_actions = 5
        self.current_state = None
        self.previous_state = None
        self.previous_action = None
        self.children = decendant
        self.name = "King NN"
        self.data_file = "King_model_file"

