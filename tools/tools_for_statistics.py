# -*- coding: utf-8 -*-
# @Time : 2022/5/13 23:13
# @Author : Mengtian Zhang
# @Version : v-dev-0.0
# @Function 

"""Summary.

Description.----------------------------------------------------------------
----------------------------------------------------------------------------
----------------------------------------------------------------------------

Example:

"""


class ValueStat:
    """A simple module for value stats."""

    def __init__(self):
        self.val = 0
        self.count = 0

    def update(self, val: float):
        self.val += val
        self.count += 1

    def get_avg(self):
        if self.count == 0:
            return self.val
        else:
            return self.val / self.count
