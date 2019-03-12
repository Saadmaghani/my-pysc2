# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""A random agent for starcraft."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy, random, time

from pysc2.agents import base_agent
from pysc2.lib import actions, units


class RandomAgent(base_agent.BaseAgent):
  """A random agent for starcraft."""

  def unit_type_is_selected(self, obs, unit_type):
    if (len(obs.observation.single_select) > 0 and
            obs.observation.single_select[0].unit_type == unit_type):
      return True

    if (len(obs.observation.multi_select) > 0 and
            obs.observation.multi_select[0].unit_type == unit_type):
      return True



  def step(self, obs):
    super(RandomAgent, self).step(obs)



    drones = [unit for unit in obs.observation.feature_units if unit.unit_type == 104]

    enemy = (obs.observation.feature_screen.player_relative == 4).nonzero()

    x = {val.value: 0 for val in units.Protoss}

    if obs.first():
      self.cp = 1

    if self.cp == 1 and len(drones) > 0:
      drone = random.choice(drones)
      self.cp = 2
      return actions.FUNCTIONS.select_point("select_all_type", (drone.x, drone.y))
    elif self.cp==2:
      self.cp = 3
      x, y = obs.observation.feature_minimap.selected.nonzero()

      return actions.FUNCTIONS.select_control_group("set", 1)
    if self.cp == 3:
      self.cp = 4
      return actions.FUNCTIONS.select_point("select", (42, 42))
    elif self.cp == 4:
      self.cp = 5
      return actions.FUNCTIONS.move_camera((36,36))
    elif self.cp == 5:
      self.cp = 6
      return actions.FUNCTIONS.select_control_group("recall", 1)
    elif self.cp == 6:
      self.cp = 7
      x,y = obs.observation.feature_minimap.selected.nonzero()
      return actions.FUNCTIONS.move_camera((x,y))
    #return actions.FUNCTIONS.select_control_group("recall", 1)
    return actions.FUNCTIONS.no_op()
