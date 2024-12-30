#!/usr/bin/env python3

from funcy import partial, compose, rcompose
from operator import add
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import k # Boltzmann constant
import sys
import math


temp = 300.15 # kelvin
bandwidth = 10000000 # Hz
power_tx = 33-30 # dB
gain_tx = 14 # dB
gain_rx = 14 # dB
height_tx = 1.5+50 # meter
height_rx = 1.5 # meter
std_dev = 6 # standard deviation for log-normal distribution (in dB)

rng = np.random.default_rng()

d = list(range(1,1000+1, 1)) # list of distances (in meter)


def to_dB(x):
    return 10*math.log10(x)

def from_dB(db):
    return pow(10, db/10)

def path_loss(h_t, h_r, dist):
    # returns 10*log(g(dist)), i.e., gain in dB
    # using two-ray-ground model for path loss
    return to_dB(pow(h_t*h_r, 2)/pow(dist, 4))

def shadowing(_std_dev):
    # returns the gain caused by shawdowing, in dB
    # log-normal distribution model for shadowing
    return rng.normal(loc=0, scale=_std_dev)

def noise(temp, bandwidth):
    # returns thermal noise power in dB
    return to_dB(k*temp*bandwidth)

def SNR_dB(noise_power_dB, signal_power_dB):
    return signal_power_dB-noise_power_dB


def q1():
    power_rx = list(map(compose(partial(add, power_tx+gain_tx+gain_rx),
                                partial(path_loss, height_tx, height_rx)),
                        d))
    SINR = list(map(partial(SNR_dB, noise(temp, bandwidth)), power_rx))
    #print(noise(temp, bandwidth))

    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(10,10))
    ax1.plot(d, power_rx, color='red', linewidth=2)
    ax1.set_xlim(left=0)
    ax1.grid()
    ax1.set_xlabel('Distance (m)')
    ax1.set_ylabel('Received Power (dB)')
    ax1.set_title('Received Power of the Mobile Device')

    ax2.plot(d, SINR, color='red', linewidth=2)
    ax2.set_xlim(left=0)
    ax2.grid()
    ax2.set_xlabel('Distance (m)')
    ax2.set_ylabel('SINR (dB)')
    ax2.set_title('SINR of the Mobile Device')

    fig.tight_layout()
    fig.savefig('1.png')


def q2():
    power_rx = list(map(compose(lambda x: x+shadowing(std_dev),
                                partial(add, power_tx+gain_tx+gain_rx),
                                partial(path_loss, height_tx, height_rx)),
                        d))
    SINR = list(map(partial(SNR_dB, noise(temp, bandwidth)), power_rx))
    #print(noise(temp, bandwidth))

    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(10,10))
    ax1.plot(d, power_rx, color='red', linewidth=2)
    ax1.set_xlim(left=0)
    ax1.grid()
    ax1.set_xlabel('Distance (m)')
    ax1.set_ylabel('Received Power (dB)')
    ax1.set_title('Received Power of the Mobile Device')

    ax2.plot(d, SINR, color='red', linewidth=2)
    ax2.set_xlim(left=0)
    ax2.grid()
    ax2.set_xlabel('Distance (m)')
    ax2.set_ylabel('SINR (dB)')
    ax2.set_title('SINR of the Mobile Device')

    fig.tight_layout()
    fig.savefig('2.png')










def print_help():
    print("Usage: python3 main.py <problem_number>")
    sys.exit(1)


if __name__=='__main__':
    if len(sys.argv) != 2:
        print_help()

    try:
        arg = int(sys.argv[1])
    except ValueError:
        print("Error: Problem number must be an integer.")
        print_help()

    match arg:
        case 1:
            q1()
        case 2:
            q2()
        case _:
            print("Error: Unknown problem number.")
            print_help()
