import numpy as np

from Agent.BaseAgent import BaseAgent
from utilize.form_action import *


class RandomAgent(BaseAgent):
    def __init__(self, settings, seed=None):
        BaseAgent.__init__(self, settings)
        self.seed = seed
        self.settings = settings

    def act(self, obs, reward=0.0, done=False):

        adjust_gen_p_action_space = obs.action_space["adjust_gen_p"]
        adjust_gen_v_action_space = obs.action_space["adjust_gen_v"]
        adjld_p_action_space = obs.action_space["adjust_adjld_p"]
        stoenergy_p_action_space = obs.action_space["adjust_stoenergy_p"]

        if self.seed is not None:
            # To make sure sample same value
            adjust_gen_p_action_space.np_random.seed(self.seed)
            adjust_gen_v_action_space.np_random.seed(self.seed)
            adjld_p_action_space.np_random.seed(self.seed)
            stoenergy_p_action_space.np_random.seed(self.seed)

        adjust_gen_p = adjust_gen_p_action_space.sample()
        adjust_gen_v = adjust_gen_v_action_space.sample()
        # adjust_gen_v = np.zeros(self.settings.gen_num)
        adjust_adjld_p = adjld_p_action_space.sample()
        adjust_stoenergy_p = stoenergy_p_action_space.sample()
        return form_action(
            adjust_gen_p, adjust_gen_v, adjust_adjld_p, adjust_stoenergy_p
        )
