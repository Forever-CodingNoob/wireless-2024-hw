#!/usr/bin/env python3

from funcy import partial, rpartial, compose, rcompose, wraps, curry, autocurry
from functools import reduce
from operator import add, sub, truediv, mul
import numpy as np
import numpy.typing as npt
import numpy.linalg as la
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import colormaps
from scipy.constants import k # Boltzmann constant
import sys
from math import sqrt, log10
import random

to_dB = np.vectorize(compose(partial(mul, 10), log10))
from_dB = np.vectorize(compose(partial(pow, 10), rpartial(truediv, 10)))

W_to_dBW = to_dB
dBW_to_W = from_dB
dBW_to_dBm = rpartial(add, 30)
dBm_to_dBW = rpartial(sub, 30)
dBm_to_W = compose(dBW_to_W, dBm_to_dBW)
W_to_dBm = compose(dBW_to_dBm, W_to_dBW)

def negate(f):
    @wraps(f)
    def g(*args,**kwargs):
        return not f(*args,**kwargs)
    return g

def sum_functions(funcs):
    return lambda *args, **kwargs: reduce(add, (func(*args, **kwargs) for func in funcs))

@autocurry
def map_(f, it):
    '''another map that returns a list'''
    l=list(map(f, it))
    return np.array(l) if isinstance(it, np.ndarray) else l

@autocurry
def filter_(f, it):
    '''another filter that returns a list'''
    l=list(filter(f, it))
    return np.array(l) if isinstance(it, np.ndarray) else l




''' constants and definitions '''
temp = 300.15 # kelvin
bandwidth = 1e7 # Hz
num_cell = 19
power_bs = dBm_to_W(33) # W
power_ms = dBm_to_W(23) # W
gain_tx = from_dB(14) # ratio
gain_rx = from_dB(14) # ratio
isd = 500. # meter
l = isd/sqrt(3) # length of a side of every hexagonal cell (in meter)
height_bs = 1.5+50 # meter
height_ms = 1.5 # meter
noise = k*temp*bandwidth # thermal noise power (in W)

rng = np.random.default_rng()


vectors_basis = list(np.array([
                    (-l,0),
                    (l/2,l/2*sqrt(3)),
                    (l/2,-l/2*sqrt(3))
                ]))

vectors_to_bs = list(np.array([
                    (l*3/2, l/2*sqrt(3)),
                    (0, l*sqrt(3)),
                    (-l*3/2, l/2*sqrt(3)),
                    (-l*3/2, -l/2*sqrt(3)),
                    (0, -l*sqrt(3)),
                    (l*3/2, -l/2*sqrt(3))
                ]))

bss: np.ndarray = np.unique([i+j for i in vectors_to_bs for j in vectors_to_bs], axis=0)
order = np.argsort(la.norm(bss, axis=1))
bss = bss[order]

#print(bss)
assert len(bss)==num_cell


@autocurry
def distance(v_a, v_b):
    return la.norm(v_a - v_b)

@autocurry
def path_loss(h_t, h_r, dist):
    '''
    returns g(dist), i.e., gain in W
    using two-ray-ground model for path loss
    '''
    return pow(h_t*h_r, 2)/pow(dist, 4)

@autocurry
def received_power(power_tx, h_t, h_r, dist):
    return power_tx*gain_tx*gain_rx*path_loss(h_t, h_r, dist)


def generate_ms(cell_center: npt.ArrayLike) -> np.ndarray:
    vs = np.array(random.sample(vectors_basis, k=2)).T
    while 114514:
        x = rng.uniform(0, 1, size=2)
        if not np.all(x == 0):
            break
    return cell_center + vs@x

def pick_cell() -> int:
    return random.randint(0, len(bss)-1)

def get_bs_pos(cell_id: int) -> np.ndarray:
    return bss[cell_id]

def generate_ms_in_any_cell() -> np.ndarray:
    cell_id = pick_cell()
    return cell_id, generate_ms_in_certian_cell(get_bs_pos(cell_id))

@autocurry
def SINR(signal_power, interference, noise):
    return signal_power/(noise+interference)


def hex_vertices(cell_center: npt.ArrayLike, radius) -> np.ndarray:
    return map_(lambda angle: (cell_center[0]+radius*np.cos(angle), cell_center[1]+radius*np.sin(angle)),
                np.linspace(0, 2*np.pi, 7))











def q1():
    global fig, gs

    central_bs_pos = np.array((0,0))
    num_ms = 50

    mss = np.array(map_(generate_ms, [central_bs_pos] * num_ms))
    dist = map_(distance(central_bs_pos), mss)

    received_power_downlink = received_power(power_bs, height_bs, height_ms) # an unary function of distance

    power_rx = received_power_downlink(dist) # in W

    other_bss = filter_(negate(curry(np.array_equiv)(central_bs_pos)), bss)
    assert len(other_bss)==num_cell-1

    interference_per_ms = lambda ms_pos: sum(map_(compose(
                            received_power_downlink,
                            distance(ms_pos)
                        ), other_bss))

    SINR_ = np.vectorize(lambda signal, ms_pos: SINR(signal, interference_per_ms(ms_pos), noise), signature='(),(m)->()')(power_rx, mss)

    ax1 = fig.add_subplot(gs[:, 0])  # `:` means spanning across both rows
    ax1.scatter(*mss.T, color='blue', label='Mobile Device', alpha=0.7, s=2)
    ax1.scatter(*central_bs_pos, color='red', label='Central BS', marker='x', s=100)
    hex_v = hex_vertices(central_bs_pos, l)
    ax1.plot(*hex_v.T, color='#e83d31', label='Central Cell Edge', linewidth=1)

    ax1.set_xlim(left=-l*1.1, right=l*1.1)
    ax1.set_ylim(bottom=-l*1.1, top=l*1.1)
    ax1.set_xlabel('X Position (meter)')
    ax1.set_ylabel('Y Position (meter)')
    ax1.set_title('Location of the Central BS and Mobile Devices in the Central Cell')
    ax1.legend()
    ax1.grid()
    ax1.set_aspect('equal')


    ax2 = fig.add_subplot(gs[0, 1])  # `:` means spanning across both rows
    ax2.scatter(dist, W_to_dBW(power_rx), color='blue', s=2)

    ax2.set_xlabel('Distance to the central BS (meter)')
    ax2.set_ylabel('Received Power (dB)')
    ax2.set_title('Received Power of the Mobile Devices')
    ax2.grid()


    ax3 = fig.add_subplot(gs[1, 1])  # `:` means spanning across both rows
    ax3.scatter(dist, to_dB(SINR_), color='limegreen', s=2)

    ax3.set_xlabel('Distance to the central BS (meter)')
    ax3.set_ylabel('SINR (dB)')
    ax3.set_title('SINR of the Mobile Devices')
    ax3.grid()

    fig.suptitle('Problem 1: Downlink in the Central Cell', fontsize=16, color='black')
    fig.savefig('1.png')



def q2():
    global fig, gs

    central_bs_pos = np.array((0,0))
    num_ms = 50

    mss = np.array(map_(generate_ms, [central_bs_pos] * num_ms))
    dist = map_(distance(central_bs_pos), mss)

    received_power_uplink = received_power(power_ms, height_ms, height_bs) # an unary function of distance

    power_rx = received_power_uplink(dist) # in W

    other_mss = lambda ms_pos: filter_(negate(curry(np.array_equiv)(ms_pos)), mss)
    assert len(other_mss(mss[0]))==num_ms-1

    interference_per_ms = lambda ms_pos: sum(map_(compose(
                            received_power_uplink,
                            distance(central_bs_pos)
                        ), other_mss(ms_pos)))

    SINR_ = np.vectorize(lambda signal, ms_pos: SINR(signal, interference_per_ms(ms_pos), noise), signature='(),(m)->()')(power_rx, mss)

    ax1 = fig.add_subplot(gs[:, 0])  # `:` means spanning across both rows
    ax1.scatter(mss[:, 0], mss[:, 1], color='blue', label='Mobile Device', alpha=0.7, s=2)
    ax1.scatter(*central_bs_pos, color='red', label='Central BS', marker='x', s=100)
    hex_v = hex_vertices(central_bs_pos, l)
    ax1.plot(*hex_v.T, color='#e83d31', label='Central Cell Edge', linewidth=1)

    ax1.set_xlim(left=-l*1.1, right=l*1.1)
    ax1.set_ylim(bottom=-l*1.1, top=l*1.1)
    ax1.set_xlabel('X Position (meter)')
    ax1.set_ylabel('Y Position (meter)')
    ax1.set_title('Location of the Central BS and Mobile Devices in the Central Cell')
    ax1.legend()
    ax1.grid()
    ax1.set_aspect('equal')


    ax2 = fig.add_subplot(gs[0, 1])  # `:` means spanning across both rows
    ax2.scatter(dist, W_to_dBW(power_rx), color='blue', s=2)

    ax2.set_xlabel('Distance to the corresponding MS (meter)')
    ax2.set_ylabel('Received Power (dB)')
    ax2.set_title('Received Power of the Central BS')
    ax2.grid()


    ax3 = fig.add_subplot(gs[1, 1])  # `:` means spanning across both rows
    ax3.scatter(dist, to_dB(SINR_), color='limegreen', s=2)

    ax3.set_xlabel('Distance to the corresponding MS (meter)')
    ax3.set_ylabel('SINR (dB)')
    ax3.set_title('SINR of the Central BS')
    ax3.grid()

    fig.suptitle('Problem 2: Uplink in the Central Cell', fontsize=16, color='black')
    fig.savefig('2.png')



def q3():
    global fig, gs
    cmap_name = 'tab20'
    cmap = colormaps.get_cmap(cmap_name)

    num_ms_per_cell = 50
    cell_ids = np.repeat(np.arange(len(bss)), num_ms_per_cell)
    colors = map_(cmap, cell_ids)

    mss = map_(compose(generate_ms, get_bs_pos), cell_ids)
    dist = np.vectorize(lambda cell_id, ms_pos: distance(get_bs_pos(cell_id), ms_pos), signature='(),(m)->()')(cell_ids, mss)

    received_power_uplink = received_power(power_ms, height_ms, height_bs) # an unary function of distance

    power_rx = received_power_uplink(dist) # in W

    other_mss = lambda ms_pos: filter_(negate(curry(np.array_equiv)(ms_pos)), mss)
    assert len(other_mss(mss[0]))==num_ms_per_cell*num_cell-1

    interference_per_ms = lambda cell_id, ms_pos: sum(map_(compose(
                            received_power_uplink,
                            distance(get_bs_pos(cell_id))
                        ), other_mss(ms_pos)))

    SINR_ = np.vectorize(lambda signal, ms_pos, cell_id: SINR(signal, interference_per_ms(cell_id, ms_pos), noise), signature='(),(m),()->()')(power_rx, mss, cell_ids)

    #print(get_bs_pos(cell_ids[np.argmin(SINR_)]))

    ax1 = fig.add_subplot(gs[:, 0])  # `:` means spanning across both rows
    ax1.scatter(mss[:, 0], mss[:, 1], c=colors, label='Mobile Device', alpha=0.7, s=2)
    ax1.scatter(bss[:, 0], bss[:, 1], color='red', label='BS', marker='x', s=50)

    for i in range(len(bss)):
        hex_v = hex_vertices(bss[i], l)
        ax1.plot(*hex_v.T, color=cmap(i), linewidth=1)

    max_range = l*sqrt(3)/2*5
    ax1.set_xlim(left=-max_range*1.05, right=max_range*1.05)
    ax1.set_ylim(bottom=-max_range*1.05, top=max_range*1.05)
    ax1.set_xlabel('X Position (meter)')
    ax1.set_ylabel('Y Position (meter)')
    ax1.set_title('Location of BS and Mobile Devices in Each Cell')
    ax1.legend()
    ax1.grid()
    ax1.set_aspect('equal')


    ax2 = fig.add_subplot(gs[0, 1])  # `:` means spanning across both rows
    ax2.scatter(dist, W_to_dBW(power_rx), c=colors, s=2)

    ax2.set_xlabel('Distance to the corresponding MS (meter)')
    ax2.set_ylabel('Received Power (dB)')
    ax2.set_title('Received Power of Each BS')
    ax2.grid()


    ax3 = fig.add_subplot(gs[1, 1])  # `:` means spanning across both rows
    ax3.scatter(dist, to_dB(SINR_), c=colors, s=2)

    ax3.set_xlabel('Distance to the corresponding MS (meter)')
    ax3.set_ylabel('SINR (dB)')
    ax3.set_title('SINR of Each BS')
    ax3.grid()

    fig.suptitle('Bonus: Uplink in All Cells', fontsize=16, color='black')
    fig.savefig('3.png')







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

    ''' plotting  '''
    fig = plt.figure(figsize=(16, 8))
    gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1], hspace=0.4)


    match arg:
        case 1:
            q1()
        case 2:
            q2()
        case 3:
            q3()
        case _:
            print("Error: Unknown problem number.")
            print_help()
