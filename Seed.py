#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Utility class used to set the seed of the seed of experiments.

def getSeed(seed = 376726348, time = 0):
    s = list(str(seed))
    s[1] = str((int(s[1]) + time) % 10) 
    s[5] = str((int(s[5]) + time) % 10)
    s[7] = str((int(s[7]) + time) % 10)
    return int(''.join(s))