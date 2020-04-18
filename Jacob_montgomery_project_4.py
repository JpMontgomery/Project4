#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed April 8 11:55:11 2020

@author: jacobmontgomery

TODO:
"""
# %% Libs
import pandas as pd
from pandas import DataFrame, read_csv
import csv
import math as m
import numpy as np
from numpy import linalg as la
from scipy import linalg as las
from matplotlib import pyplot as plt


# %% Functions

def C_n_b(phi, theta, psi):
    DCM = np.array([[m.cos(theta) * m.cos(psi), m.sin(phi) * m.sin(theta) * m.cos(psi) - m.cos(phi) * m.sin(psi),
                     m.cos(phi) * m.sin(theta) * m.cos(psi) + m.sin(phi) * m.sin(psi)],
                    [m.cos(theta) * m.sin(psi), m.sin(phi) * m.sin(theta) * m.sin(psi) + m.cos(phi) * m.cos(psi),
                     m.cos(phi) * m.sin(theta) * m.sin(psi) - m.sin(phi) * m.cos(psi)],
                    [-m.sin(theta), m.sin(phi) * m.cos(theta), m.cos(phi) * m.cos(theta)]])
    return DCM


def M(phi, theta, psi):
    zeros = np.zeros((3, 3))

    top = np.hstack((zeros, zeros, zeros, zeros))

    second = np.hstack((C_n_b(phi, theta, psi), zeros, zeros, zeros))

    third = np.hstack((zeros, -C_n_b(phi, theta, psi), zeros, zeros))

    fourth = np.hstack((zeros, zeros, np.identity(3), zeros))

    bottom = np.hstack((zeros, zeros, zeros, np.identity(3)))

    M = np.vstack((top, second, third, fourth, bottom))

    return M


def A(theta, phi):
    A = (1 / m.cos(theta)) * np.array([[1, m.sin(phi) * m.sin(theta), m.cos(phi) * m.sin(theta)],
                                       [0, m.cos(phi) * m.cos(theta), -m.sin(phi) * m.cos(theta)],
                                       [0, m.sin(phi), m.cos(phi)]])
    return A


def ERR(lat):
    ERR = 7.292115 * 10 ** (-5) * np.array([[m.cos(lat)], [0], [-m.sin(lat)]])
    return ERR


def Trans_rate(vel_x, vel_y, lat, h):
    a = 6378137
    f = 1 / 298.257223563
    e = m.sqrt(f * (2 - f))
    Rn = (a * (1 - e ** 2)) / ((1 - e ** 2 * m.sin(lat) ** 2) ** 1.5)
    Re = a // (m.sqrt(1 - e ** 2 * m.sin(lat) ** 2))
    Tran_rate = np.array([[(vel_y / (Re + h))], [(-vel_x) / (Rn + h)], [-(vel_y * m.tan(lat) / (Re + h))]])
    return Tran_rate


def local_grav(lat, h):
    a = 6378137
    f = 1 / 298.257223563
    g0 = (9.7803253359 / m.sqrt(1 - f * (2 - f) * m.sin(lat) ** 2)) * (1 + 0.0019311853 * m.sin(lat) ** 2)
    OMEGA = 7.292115 * 10 ** (-5)
    mu = 3.986004418 * 10 ** 14
    ch = 1 - 2 * (1 + f + (a ** 3 * (1 - f) * (OMEGA ** 2)) / mu) * (h / a) + 3 * (h / a) ** 2
    gn = np.array([[0], [0], [ch * g0]])
    return gn


def skew_op(vec):
    x = float(vec[0])
    y = float(vec[1])
    z = float(vec[2])
    skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return skew


def pos_dot_mat(lat, h):
    a = 6378137
    f = 1 / 298.257223563
    e = m.sqrt(f * (2 - f))
    Rn = (a * (1 - e ** 2)) / ((1 - e ** 2 * m.sin(lat) ** 2) ** 1.5)
    Re = a // (m.sqrt(1 - e ** 2 * m.sin(lat) ** 2))
    pos_dot = np.diag((1 / (Rn + h), 1 / ((Re + h) * m.cos(lat)), -1))
    return pos_dot

def big_A(omega_n_e_n, gn, omega_n_i_e, fb, omega_n_i_n, euler):
    a = 6378137
    top = np.hstack((-1*skew_op(omega_n_e_n), np.identity(3), np.zeros((3, 3)), np.zeros((3, 3)), np.zeros((3, 3))))
    sec = np.hstack((np.diag((-1, -1, 2))*la.norm(gn)/a, -1*skew_op(2 * omega_n_i_e + omega_n_e_n),
                     skew_op(C_n_b(euler[0], euler[1], euler[2])@fb), C_n_b(euler[0], euler[1], euler[2]),
                     np.zeros((3, 3))))
    third = np.hstack((np.zeros((3, 3)), np.zeros((3, 3)), -1 * skew_op(omega_n_i_n), np.zeros((3, 3)),
                       -1*C_n_b(euler[0], euler[1], euler[2])))
    fourth = np.hstack((np.zeros((3, 3)), np.zeros((3, 3)), np.zeros((3, 3)), -1/300*np.identity(3), np.zeros((3, 3))))
    bottom = np.hstack((np.zeros((3, 3)), np.zeros((3, 3)), np.zeros((3, 3)), np.zeros((3, 3)), -1/300*np.identity(3)))
    big_a = np.vstack((top, sec, third, fourth, bottom))
    return big_a

def LLA_ECEF(pos):
    a = 6378137
    f = 1 / 298.257223563
    e = m.sqrt(f * (2 - f))
    b_2 = - a**2 * (e**2 - 1)
    N = a/m.sqrt(1 - e**2 * (m.sin((pos[0]))))
    X = (N+pos[2])*m.cos((pos[0])) * m.cos((pos[1]))
    Y = (N + pos[2]) * m.cos((pos[0])) * m.sin((pos[1]))
    Z = ((b_2/(a**2)*N + pos[2])) * m.sin((pos[0]))
    return np.array([[X[0]], [Y[0]], [Z[0]]])

# %% Givens

g = 9.8
gps_std = 3
gps_vel = .2
acc_bias01 = [.25, .077, -.12]
acc_bias0 = acc_bias01
acc_mark = 0.0005 * g
acc_T = 300
acc_noise = .12 * g
gyro_bias01 = [2.4 * 10 ** -4, -1.3 * 10 ** -4, 5.6 * 10 ** -4]
gyro_bias0 = gyro_bias01
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

imu_data.set_index('times', inplace=True)

pos_data.index = imu_data.index

vel_data.index = imu_data.index


# %% INS Formulation

cols = ['lat', 'long', 'h', 'phi', 'theta', 'psi', 'vel_N', 'vel_E', 'vel_D', 'a_b_x', 'a_b_y', 'a_b_z',
        'g_b_x', 'g_b_y', 'g_b_z']

INS_data = pd.DataFrame(index=imu_data.index, columns=cols)

INS_data = INS_data.fillna(0)

nums = [0, 1, 2]

for num in nums:
    INS_data.iloc[0, num] = pos_data.iloc[0, num]
    INS_data.iloc[0, num + 6] = vel_data.iloc[0, num]
    INS_data.iloc[0, num + 9] = acc_bias01[num]
    INS_data.iloc[0, num + 12] = gyro_bias01[num]

old_time = INS_data.index[0]

PSDa = (2 * acc_mark**2 / acc_T)

PSDg = (2 * m.radians(gyro_mark)**2 / gyro_T)

S = np.diag((acc_noise**2, acc_noise**2, acc_noise**2, m.radians(gyro_noise)**2, m.radians(gyro_noise)**2,
             m.radians(gyro_noise)**2, PSDa, PSDa, PSDa, PSDg, PSDg, PSDg))

P_old = 10 * np.diag((gps_std**2, gps_std**2, gps_std**2, gps_vel**2, gps_vel**2, gps_vel**2, m.radians(10)**2,
                      m.radians(10)**2, m.radians(10)**2, (10 *acc_mark)**2, (10 *acc_mark)**2, (10 *acc_mark)**2,
                      (10 * m.radians(gyro_mark))**2, (10 * m.radians(gyro_mark))**2, (10 * m.radians(gyro_mark))**2))

for time in INS_data.index:
    if time != INS_data.index[0]:
        dt = time - old_time
        if time != INS_data.index[1]:
            INS_data.loc[(old_time, 'lat')] = float(pos_new[0])
            INS_data.loc[(old_time, 'long')] = float(pos_new[1])
            INS_data.loc[(old_time, 'h')] = float(pos_new[2])
            INS_data.loc[(old_time, 'phi')] = float(euler_new[0])
            INS_data.loc[(old_time, 'theta')] = float(euler_new[1])
            INS_data.loc[(old_time, 'psi')] = float(euler_new[2])
            INS_data.loc[(old_time, 'vel_N')] = float(velo_new[0])
            INS_data.loc[(old_time, 'vel_E')] = float(velo_new[1])
            INS_data.loc[(old_time, 'vel_D')] = float(velo_new[2])
            INS_data.loc[(old_time, 'a_b_x')] = acc_bias0[0]
            INS_data.loc[(old_time, 'a_b_y')] = acc_bias0[1]
            INS_data.loc[(old_time, 'a_b_z')] = acc_bias0[2]
            INS_data.loc[(old_time, 'g_b_x')] = gyro_bias0[0]
            INS_data.loc[(old_time, 'g_b_y')] = gyro_bias0[1]
            INS_data.loc[(old_time, 'g_b_z')] = gyro_bias0[2]
        lat = INS_data.loc[(old_time, 'lat')]
        long = INS_data.loc[(old_time, 'long')]
        vel_x = INS_data.loc[(old_time, 'vel_N')]
        vel_y = INS_data.loc[(old_time, 'vel_E')]
        vel_z = INS_data.loc[(old_time, 'vel_D')]
        h = INS_data.loc[(old_time, 'h')]
        phi = INS_data.loc[(old_time, 'phi')]
        theta = INS_data.loc[(old_time, 'theta')]
        psi = INS_data.loc[(old_time, 'psi')]
        omega_b_i_b = np.array([[imu_data.loc[(old_time, 'gyro_x')]], [imu_data.loc[(old_time, 'gyro_y')]],
                                [imu_data.loc[(old_time, 'gyro_z')]]])
        omega_n_i_e = ERR(lat)
        omega_n_e_n = Trans_rate(vel_x, vel_y, lat, h)
        C_nb = C_n_b(phi, theta, psi)
        omega_b_i_n = C_nb @ (omega_n_i_e + omega_n_e_n)
        omega_b_n_b = omega_b_i_b - omega_b_i_n + np.array([[gyro_bias0[0]], [gyro_bias0[1]], [gyro_bias0[2]]])
        omega_n_i_n = omega_n_i_e + omega_n_e_n
        euler_old = np.array([[phi], [theta], [psi]])
        euler_new = euler_old + dt * A(theta, phi) @ omega_b_n_b
        fb = np.array([[imu_data.loc[(old_time, 'acc_x')]], [imu_data.loc[(old_time, 'acc_y')]],
                       [imu_data.loc[(old_time, 'acc_z')]]]) + \
             np.array([[acc_bias0[0]], [acc_bias0[1]], [acc_bias0[2]]])
        fn = C_nb @ fb
        gn = local_grav(lat, h)
        velo_old = np.array([[vel_x], [vel_y], [vel_z]])
        v_n_dot = fn + gn - skew_op(2 * omega_n_i_e - omega_n_e_n) @ velo_old
        velo_new = velo_old + dt * v_n_dot
        pos_old = np.array([[lat], [long], [h]])
        pos_dot = pos_dot_mat(lat, h) @ velo_new
        pos_new = pos_old + dt * pos_dot
        lat = pos_new[0]
        long = pos_new[1]
        alt = pos_new[2]
        AA = big_A(omega_n_e_n, local_grav(lat, h), omega_n_i_e, fb, omega_n_i_n, euler_old)
        Ms = M(phi, theta, psi)
        Q = (np.identity(15) + .02 * AA) @ (.02 * Ms @ S @ Ms.T)
        F = las.expm(AA * .02)
        top = np.hstack((-AA, Ms @ S @ Ms.T))
        bottom = np.hstack((np.zeros((15, 15)), AA.T))
        BIG = np.vstack((top, bottom))
        BIG = las.expm(np.vstack((top, bottom)) * .02)
        BR = BIG[15:30, 15:30]
        TR = BIG[0:15, 15:30]
        Q_new = BR.T @ TR
        P_new = F@P_old@F.T+Q
        P_new = .5 * (P_new + P_new.T)
        if time % 1 == 0:
            vel_gps = np.array([[vel_data.loc[(time, 'vel_N')]],
                        [vel_data.loc[(time, 'vel_E')]],
                        [vel_data.loc[(time, 'vel_D')]]])
            pos_gps = np.array([[pos_data.loc[(time, 'Lat')]], [pos_data.loc[(time, 'Long')]],
                                [pos_data.loc[(time, 'Alt')]]])
            pos_gps = LLA_ECEF(pos_gps)
            pos_ins = LLA_ECEF(pos_new)
            diff_vel = velo_new - vel_gps
            diff_pos = pos_ins - pos_gps
            H = np.asarray(np.bmat([[np.identity(6), np.zeros((6, 9))]]))
            R = np.diag((9, 9, 9, .2**2, .2**2, .2**2))
            K_gain = P_new@H.T@la.inv((H@P_new@H.T+R))
            v = np.array([[3], [3], [3], [.2], [.2], [.2]])
            del_x = np.vstack((diff_pos, diff_vel))
            del_y = del_x + v
            P_new = (np.identity(15) - K_gain@H)@P_new
            P_new = .5 * (P_new + P_new.T)
            del_x_new = K_gain@del_y
            a = 6378137
            f = 1 / 298.257223563
            e = m.sqrt(f * (2 - f))
            Rn = (a * (1 - e ** 2)) / ((1 - e ** 2 * m.sin(lat) ** 2) ** 1.5)
            Re = a // (m.sqrt(1 - e ** 2 * m.sin(lat) ** 2))
            new_lat = lat - del_x_new[0]/(Rn + alt)
            new_long = long - del_x_new[1] / ((Re + alt)*m.cos(lat))
            h_new = alt - del_x_new[2]
            velo_new = velo_new - del_x_new[3:6]
            velo_new[2] = vel_data.loc[(time, 'vel_D')]
            pos_new = np.array([[new_lat[0]], [new_long[0]], [pos_data.loc[(time, 'Alt')]]])
            # pos_new = np.array([[new_lat[0]], [new_long[0]], [h_new]])
            acc_bias0 = [-del_x_new[9][0] + acc_bias0[0], -del_x_new[10][0] + acc_bias0[1], -del_x_new[11][0]
                         + acc_bias0[2]]
            gyro_bias0 = [-del_x_new[12][0] + gyro_bias0[0], -del_x_new[13][0] + gyro_bias0[1], -del_x_new[14][0]
                         + gyro_bias0[2]]
            euler_error = np.array([[del_x_new[6][0]], [del_x_new[7][0]], [del_x_new[8][0]]])
            new_c_n_b = (np.identity(3) + skew_op(euler_error))@C_n_b(euler_old[0][0], euler_old[1][0], euler_old[2][0])
            phi_up = m.atan2(new_c_n_b[2][1], new_c_n_b[2][2])
            theta_up = -m.asin(new_c_n_b[2][0])
            psi_update = m.atan2(new_c_n_b[1][0], new_c_n_b[0][0])
            euler_new = np.array([[phi_up], [theta_up], [psi_update]])
        if time == INS_data.index[-1]:
            INS_data.loc[(time, 'lat')] = float(pos_new[0])
            INS_data.loc[(time, 'long')] = float(pos_new[1])
            INS_data.loc[(time, 'h')] = float(pos_new[2])
            INS_data.loc[(time, 'phi')] = float(euler_new[0])
            INS_data.loc[(time, 'theta')] = float(euler_new[1])
            INS_data.loc[(time, 'psi')] = float(euler_new[2])
            INS_data.loc[(time, 'vel_N')] = float(velo_new[0])
            INS_data.loc[(time, 'vel_E')] = float(velo_new[1])
            INS_data.loc[(time, 'vel_D')] = float(velo_new[2])
            INS_data.loc[(time, 'a_b_x')] = acc_bias0[0]
            INS_data.loc[(time, 'a_b_y')] = acc_bias0[1]
            INS_data.loc[(time, 'a_b_z')] = acc_bias0[2]
            INS_data.loc[(time, 'g_b_x')] = gyro_bias0[0]
            INS_data.loc[(time, 'g_b_y')] = gyro_bias0[1]
            INS_data.loc[(time, 'g_b_z')] = gyro_bias0[2]
        old_time = time
        P_old = P_new
        # if time == 472.0:
        #     break
