import pandas as pd

class ForecastReader(object):
    def __init__(self, settings):
        def_max_renewable_gen_p = pd.read_csv(settings.max_renewable_gen_p_filepath)
        self.max_renewable_gen_p_all = def_max_renewable_gen_p.values.tolist()
        def_ld_p = pd.read_csv(settings.ld_p_filepath)
        self.ld_p_all = def_ld_p.values.tolist()
        self.settings = settings

    def read_step_renewable_gen_p_max(self, t):
        cur_step_renewable_gen_p_max = self.max_renewable_gen_p_all[t]
        if t == self.settings.sample_num - 1:
            next_step_renewable_gen_p_max = self.max_renewable_gen_p_all[t]
        else:
            next_step_renewable_gen_p_max = self.max_renewable_gen_p_all[t+1]
        return cur_step_renewable_gen_p_max, next_step_renewable_gen_p_max

    def read_step_ld_p(self, t):
        cur_step_ld_p = self.ld_p_all[t]
        if t == self.settings.sample_num - 1:
            next_step_ld_p = self.ld_p_all[t]
        else:
            next_step_ld_p = self.ld_p_all[t+1]
        return cur_step_ld_p, next_step_ld_p
