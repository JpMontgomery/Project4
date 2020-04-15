#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed April 8 11:55:11 2020

@author: jacobmontgomery

"""
# %% Libs
import pandas as pd
from pandas import DataFrame, read_csv
import csv
import math as m
import numpy as np
from numpy import linalg as la
from matplotlib import pyplot as plt

# %% Functions

def C_n_b(phi, theta, psi):
    DCM = np.array([[m.cos(theta)*m.cos(psi), m.sin(phi)*m.sin(theta)*m.cos(psi)-m.cos(phi)*m.sin(psi),
                     m.cos(phi)*m.sin(theta)*m.cos(psi)+m.sin(phi)*m.sin(psi)],
                    [m.cos(theta)*m.sin(psi), m.sin(phi)*m.sin(theta)*m.sin(psi)+m.cos(phi)*m.cos(psi),
                     m.cos(phi)*m.sin(theta)*m.sin(psi)-m.sin(phi)*m.cos(psi)],
                    [-m.sin(theta), m.sin(phi)*m.cos(theta), m.cos(phi)*m.cos(theta)]])
    return DCM

def M(phi,theta,psi):
    zeros = np.zeros((3, 3))

    top = np.hstack((zeros, zeros, zeros, zeros))

    second = np.hstack((C_n_b(phi, theta, psi), zeros, zeros, zeros))

    third = np.hstack((zeros, -C_n_b(phi, theta, psi), zeros, zeros))

    fourth = np.hstack((zeros, zeros, np.identity(3), zeros))

    bottom = np.hstack((zeros, zeros, zeros, np.identity(3)))

    M = np.vstack((top, second, third, fourth, bottom))

    return M

def A(theta, phi):
    A = (1/m.cos(theta))* np.array([[1, m.sin(phi)*m.sin(theta), m.cos(phi)*m.sin(theta)],
                  [0, m.cos(phi)*m.cos(theta), -m.sin(phi)*m.cos(theta)],
                  [0, m.sin(phi), m.cos(phi)]])
    return A

def ERR(lat):
    ERR = 7.292115*10**(-5)*np.array([[m.cos(lat)], [0], [-m.sin(lat)]])
    return ERR

def Trans_rate(vel_x, vel_y, lat, h):

    a = 6378137
    f = 1/298.257223563
    e = m.sqrt(f*(2-f))
    Rn = (a*(1-e**2))/(m.sqrt(1-e**2*sin(lat)**2))
    Ra =
    Tran_rate = np.array([[(vel_y/(Re+h))], [(-vel_x)/(Rn+h)], [-(vel_y*m.tan(lat)/(Re+h))]])

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
times = []

with open("time.txt", "r") as filestream:
    for line in filestream:
        currentline = line.split(",")
        times.append(float(currentline[0]))

vel_data = pd.read_csv("gps_vel_ned.txt", names=['vel_N', 'vel_E', 'vel_D'])

pos_data = pd.read_csv("gps_pos_lla.txt", names=['Lat', 'Long', 'Alt'])

vel_data = pd.read_csv("gps_vel_ned.txt", names=['vel_N', 'vel_E', 'vel_D'])

imu_data = pd.read_csv("imu.txt", names=['times', 'gyro_x', 'gyro_y', 'gyro_z', 'acc_x', 'acc_y', 'acc_z'])






# %% INS Formulation

cols = ['lat', 'long', 'h', 'phi', 'theta', 'psi', 'vel_N', 'vel_E', 'vel_D']

INS_data = pd.DataFrame(index=imu_data.times, columns=cols)

INS_data = INS_data.fillna(0)

nums = [0, 1, 2]

for num in nums:
    INS_data.iloc[0, num] = pos_data.iloc[0, num]
    INS_data.iloc[0, num+6] = pos_data.iloc[0, num]

for time in INS_data.index:
    E_rot = ERR(INS_data.loc[(time, 'lat')])


# %% Forming noise covariances matrices

PSDa = (2 * acc_mark/acc_T)

PSDg = (2 * gyro_mark/gyro_T)

S = np.diag((acc_mark, acc_mark, acc_mark, m.radians(gyro_mark), m.radians(gyro_mark), m.radians(gyro_mark),
             PSDa, PSDa, PSDa, PSDg, PSDg, PSDg))




