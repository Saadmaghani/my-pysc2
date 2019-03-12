from pysc2.lib import actions, features, units
import numpy as np

from Tree.BeTr_Zerg import *
from pysc2.RL_algos.DQN import DQNModel

def can_do(self, action):
    obs = BTZN().blackboard["obs"]
    return action in obs.observation.available_actions


class Warlord_NN(BTZSmartSelector):

    # actions: 32*32*2 +1 = 32*32 (1024) minimap coordinates. *2 - move and attack. +1 - noop
    def decide(self):
        self.current_state = self._process_blackboard()

        if self.previous_action is not None:
            self.learn()

        action = self.model.act(self.current_state)

        # first 1024 actions is attack at that coordinate.
        if action < 2048:  # 0-2047 so attack/move
            coord_ind = action % 1024
            BTZN().blackboard["warlord_cmd_coords"] = ((coord_ind // 32)*2 + 1, (coord_ind % 32)*2 + 1)
            if action < 1024:  # 0-1023 so attack
                self.decision = 0
            else:  # 1024-2047 so move
                self.decision = 1
        else:  # 2048 so noop
            self.decision = 2
        self.previous_action = action
        self.previous_state = self.current_state

    def learn(self):
        if BTZN().blackboard["obs"].last():
            reward = BTZN().blackboard["obs"].reward
            self.model.learn(self.previous_state, self.previous_action, reward, 'terminal', True)
            self.model.save_model(self.data_file)
        else:
            reward = -0.0001 if self.previous_action == 2048 else 0
            self.model.learn(self.previous_state, self.previous_action, reward, self.current_state)

    def _process_blackboard(self):
        enemy_locs = (BTZN().blackboard["obs"].observation["feature_minimap"].player_relative == 4).flatten()

        army_locs = BTZN().blackboard["obs"].observation["feature_minimap"].selected.flatten()

        my_troops = BTZN().blackboard["obs"].observation.multi_select

        counts = {}
        for troop in my_troops:
            str_type = str(troop.unit_type)
            if str_type in BTZN().blackboard["troops"]:
                if str_type in counts:
                    counts[str_type] += 1
                else:
                    counts[str_type] = 0


        BTZN().blackboard["troops"].update(counts)

        troops = np.zeros(5)
        ind = 0
        for key in sorted(BTZN().blackboard["troops"].keys()):
            troops[ind] = BTZN().blackboard["troops"][key]
            ind += 1

        enemy_units = np.zeros(178)
        ind = 0
        for key in sorted(BTZN().blackboard["enemy_units"].keys()):
            enemy_units[ind] = BTZN().blackboard["enemy_units"][key]
            ind += 1

        """
        #last_seen = np.zeros(150)
        #ind = 0
        #for tuple in BTZN().blackboard["last_seen"]:
            last_seen[ind] = tuple[1]
            last_seen[ind + 1] = tuple[0][0]
            last_seen[ind + 2] = tuple[0][1]
            ind += 3
        """


        # total state dimension: 64*64 + 64*64 + 5 + 178 + 150 -150 = 8375
        state = np.concatenate((enemy_locs, army_locs, troops, enemy_units), axis=0)

        return state

    def setup(self):
        self.model = DQNModel(state_size=self.no_of_inputs, action_size=self.no_of_actions, architecture="AlphaZero")
        self.model.load_model(self.data_file)

    def __init__(self, decendant):
        self.no_of_inputs = 8375
        self.no_of_actions = 32*32*2 + 1
        self.current_state = None
        self.previous_state = None
        self.previous_action = None
        self.children = decendant
        self.name = "Warlord NN"
        self.data_file = "Warlord_model_file"
        self.setup()

class leaf_army_attack(BTZLeaf):

    def execute(self):
        if can_do(self, actions.FUNCTIONS.Attack_minimap.id):
            BTZN().blackboard["action"] = actions.FUNCTIONS.Attack_minimap("now", (
            BTZN().blackboard["warlord_cmd_coords"][0] ,
            BTZN().blackboard["warlord_cmd_coords"][1]))

    def __init__(self):
        self.name = self.name + " Attack"


class leaf_army_move(BTZLeaf):

    def execute(self):
        if can_do(self, actions.FUNCTIONS.Attack_minimap.id):
            BTZN().blackboard["action"] = actions.FUNCTIONS.Move_minimap ("now", (
            BTZN().blackboard["warlord_cmd_coords"][0],
            BTZN().blackboard["warlord_cmd_coords"][1]))

    def __init__(self):
        self.name = self.name + " Move"


class leaf_select_army(BTZLeaf):

    def execute(self):
        if can_do(self, actions.FUNCTIONS.select_army.id):
            BTZN().blackboard["action"] = actions.FUNCTIONS.select_army("select")

    def __init__(self):
        self.name = self.name + " Select Army"


class leaf_move_cam_to_army():
    def execute(self):
        x,y = BTZN().blackboard["obs"].observation.feature_minimap.selected.nonzero()
        if len(x) == 0:
            BTZN().blackboard["action"] = actions.FUNCTIONS.no_op()
        else:
            BTZN().blackboard["action"] = actions.FUNCTIONS.move_camera((x.mean(), y.mean()))

    def __init__(self):
        self.name = self.name + " Cam Army"



class leaf_action_noop(BTZLeaf):

    def execute(self):
        BTZN().blackboard["action"] = actions.FUNCTIONS.no_op()

    def __init__(self):
        self.name = self.name + " Action NO-OP"

