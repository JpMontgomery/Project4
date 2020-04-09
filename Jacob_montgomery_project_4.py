#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed April 8 11:55:11 2020

@author: jacobmontgomery

"""


# %% Givens

g = 9.8
gps_std = 3
gps_vel = .2
acc_bias0 = [.25, .077, -.12]
acc_mark = 0.0005*g
acc_T = 300
acc_noise = .12 * g
gyro_bias0 = [2.4 * 10**-4, -1.3 * 10**-4, 5.6 * 10**-4]
gyro_mark = .3
gyro_T = 300
gyro_noise = .95

# %% Reading Data
