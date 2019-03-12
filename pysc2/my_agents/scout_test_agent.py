from pysc2.agents import base_agent
from pysc2.env import sc2_env
from pysc2.lib import actions, features, units
from absl import app

from Tree.scout_subtree import *
from Tree.nodes_BeTr_Zerg import decorator_step_obs

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

            self.root.write("hatcheries", {"main" : (xmean,ymean)})

        self.root.execute(obs)
        #   print(self.root.act())
        return self.root.act()

    def build_tree(self):
        noop = leaf_action_noop()

        base_cam = leaf_move_cam_to_base()
        select_scout_unit = leaf_select_unit_for_scouting()
        set_scout_cg = leaf_set_scouting_control_group()

        recall_scout = leaf_recall_scout_control_group()
        scout_cam = leaf_move_cam_to_scout()
        scouting_coords = BTZSequence([leaf_send_scout(coords=(x,y)) for (x,y) in [(10,10), (20,20), (30,30), (40,40), (50,50), (60,60)]])

        get_set_send_scouts = BTZSequence([select_scout_unit, set_scout_cg, scouting_coords])

        any_units_for_scouts = selector_any_scouts([get_set_send_scouts, noop])

        scouting_sequence = BTZSequence([base_cam, any_units_for_scouts])
        get_info = BTZSequence([recall_scout, scout_cam])

        any_scouts = selector_any_units_in_scouting_cg([scouting_sequence, get_info])

        get_enemy_status = decorator_get_enemy_information([any_scouts])
        observe = decorator_step_obs([get_enemy_status])

        self.root = BTZRoot([observe])
