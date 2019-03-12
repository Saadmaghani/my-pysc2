from pysc2.agents import base_agent
from pysc2.env import sc2_env
from pysc2.lib import actions, features, units
from absl import app

from Tree.offense_subtree import *
from Tree.BeTr_Zerg import BTZSequence, BTZRoot


class ExpandAgent(base_agent.BaseAgent):
    root = BTZRoot([])

    def __init__(self):
        super(ExpandAgent, self).__init__()

        self.gas_harvesters = 0

        self.attack_coordinates = None

    def unit_type_is_selected(self, obs, unit_type):

        if (len(obs.observation.single_select) > 0 and obs.observation.single_select[0].unit_type == unit_type):
            return True

        if (len(obs.observation.multi_select) > 0 and obs.observation.multi_select[0].unit_type == unit_type):
            return True

        return False

    def get_units_by_type(self, obs, unit_type):
        return [unit for unit in obs.observation.feature_units
                if unit.unit_type == unit_type]

    def can_do(self, obs, action):
        return action in obs.observation.available_actions

    def step(self, obs):

        super(ExpandAgent, self).step(obs)

        if obs.first():
            player_y, player_x = (obs.observation.feature_minimap.player_relative ==
                                  features.PlayerRelative.SELF).nonzero()
            xmean = player_x.mean()
            ymean = player_y.mean()

            self.build_tree()
            self.root.setup(obs)

            self.gas_harvesters = 0

            if xmean <= 31 and ymean <= 31:
                self.root.write("attack_coords", (49, 49))
            else:
                self.root.write("attack_coords", (12, 16))

            self.root.write("hatcheries", {"main": (xmean, ymean)})

        self.root.execute(obs)
        #   print(self.root.act())
        return self.root.act()

    def build_tree(self):

        attack = leaf_army_attack()
        move = leaf_army_move()
        get_army = leaf_select_army()
        cam_army = leaf_move_cam_to_army()

        nop = leaf_action_noop()

        warlord_nn = Warlord_NN([attack, move, nop])
        offence_tree = BTZSequence([get_army, cam_army, warlord_nn])

        observe = decorator_step_obs([offence_tree])

        self.root = BTZRoot([observe])
