from Tree.BeTr_Zerg import *
from Tree.nodes_BeTr_Zerg import can_do, get_units_by_type, leaf_action_noop
from pysc2.lib import units

import random


class selector_any_units_in_scouting_cg(BTZSelector):
    def decide(self):
        if BTZN().blackboard["obs"].observation.control_groups[9][1] > 0:
            self.decision = 1  # control group is set and has units
        else:
            self.decision = 0  # control group either not set or doesnt have any units

    def __init__(self, decendant):
        self.children = decendant
        self.name = self.name + " Scouting Control Groups?"


class leaf_move_cam_to_base(BTZLeaf):
    def execute(self):
        # print("Aspect = ", end = "")
        x, y = BTZN().blackboard["hatcheries"]["main"]
        BTZN().blackboard["action"] = actions.FUNCTIONS.move_camera((x, y))

    def __init__(self):
        self.name = self.name + " Cam Aspect"


class selector_any_scouts(BTZSelector):
    def decide(self):
        unit_types = BTZN().blackboard["obs"].observation.feature_screen.unit_type
        x, y = (unit_types == self.scout.value).nonzero()

        if len(x) > 0:
            self.decision = 0  # overlords so select em
        else:
            self.decision = 1  # no overlords so noop

    def __init__(self, decendant, scout=units.Zerg.Overlord):
        self.scout = scout
        self.children = decendant
        self.name = self.name + " Are there any overlords for scouting?"


class leaf_select_unit_for_scouting(BTZLeaf):
    def execute(self):
        overlord = random.choice(get_units_by_type((BTZN().blackboard["obs"]), self.scout))
        BTZN().blackboard["action"] = actions.FUNCTIONS.select_point("select", (overlord.x, overlord.y))

    def __init__(self, scout=units.Zerg.Overlord):
        self.scout = scout
        self.name = self.name + " Select scout (overlord)"


class leaf_set_scouting_control_group(BTZLeaf):
    def execute(self):
        BTZN().blackboard["action"] = actions.FUNCTIONS.select_control_group("set", 9)

    def __init__(self):
        self.name = self.name + " Set Scouting control group"


class leaf_send_scout(BTZLeaf):
    def execute(self):
        x, y = self.coords
        BTZN().blackboard["action"] = actions.FUNCTIONS.Move_minimap("queued", (x, y))

    def __init__(self, coords):
        self.coords = coords
        self.name = self.name + " Set Scouting control group"


class leaf_recall_scout_control_group(BTZLeaf):
    def execute(self):
        if can_do(self, actions.FUNCTIONS.select_control_group.id):
            BTZN().blackboard["action"] = actions.FUNCTIONS.select_control_group("recall", 9)

    def __init__(self):
        self.name = self.name + " Recall Scouting Control Group"


class leaf_move_cam_to_scout(BTZLeaf):
    def execute(self):
        x, y = BTZN().blackboard["obs"].observation.feature_minimap.selected.nonzero()

        if len(x) == 0:
            BTZN().blackboard["action"] = actions.FUNCTIONS.no_op()
        else:
            BTZN().blackboard["action"] = actions.FUNCTIONS.move_camera((x, y))

    def __init__(self):
        self.name = self.name + " Cam Scout"


class decorator_get_enemy_information(BTZDecorator):
    def execute(self):
        enemy_units = [unit for unit in BTZN().blackboard["obs"].observation.feature_units if unit.alliance == 4]
        if len(enemy_units) > 0:
            counts = {}
            for unit in enemy_units:
                if unit.unit_type not in counts:
                    counts[unit.unit_type] = 1
                else:
                    counts[unit.unit_type] += 1

            BTZN().blackboard["enemy_units"].update(counts)
        BTZDecorator.execute(self)

    def __init__(self, decendant):
        self.children = decendant
        self.name = self.name + " extra_hatchery"
