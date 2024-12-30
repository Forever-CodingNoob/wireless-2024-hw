#!/usr/bin/env python3

from typing import Any, Optional
from funcy import partial, rpartial, compose, rcompose, wraps, curry, autocurry
from functools import reduce
from operator import add, sub, truediv, mul
import numpy as np
import numpy.typing as npt
import numpy.linalg as la
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import colormaps
from matplotlib.ticker import ScalarFormatter
from scipy.constants import k # Boltzmann constant
import sys
from math import sqrt, log10, log2
import random
from enum import Enum
import time
import csv

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
def map_(f, it) -> np.ndarray|list:
    '''another map that returns a list'''
    l=list(map(f, it))
    return np.array(l) if isinstance(it, np.ndarray) else l

@autocurry
def filter_(f, it) -> np.ndarray|list:
    '''another filter that returns a list'''
    l=list(filter(f, it))
    return np.array(l) if isinstance(it, np.ndarray) else l




''' constants and definitions '''
temp = 300.15 # kelvin
bandwidth_per_cell = 1e7 # Hz
power_bs = dBm_to_W(33) # W
power_ms = dBm_to_W(0) # W
gain_tx = from_dB(14) # ratio
gain_rx = from_dB(14) # ratio
isd = 500. # meter
l = isd/sqrt(3) # length of a side of every hexagonal cell (in meter)
height_bs = 1.5+50 # meter
height_ms = 1.5 # meter
noise = lambda bandwidth: k*temp*bandwidth # thermal noise power (in W)


num_mobile_devices = 50
bs_buf_size = 6e6 # bit
CBR_per_ms = [5e5, 1e6, 2e6]
#CBR_per_ms = list(range(int(1e5),int(5e6),int(1e5)))# bit/s

sim_time = 1000 # s
time_unit = 1 # s

rng = np.random.default_rng()
cmap_name = 'tab20'
cmap = colormaps.get_cmap(cmap_name)

cell_basis = list(np.array([
                    (-l,0),
                    (l/2,l/2*sqrt(3)),
                    (l/2,-l/2*sqrt(3))
                ]))

basis = np.array([
    (l*3/2, l/2*sqrt(3)),
    (0, l*sqrt(3)),
])

#sum_cache = dict()





''' data structures and functions '''
def distance(v_a: npt.ArrayLike, v_b: npt.ArrayLike) -> float:
    return la.norm(v_a - v_b)

def SINR(signal_power: float, interference: float, noise: float) -> float:
    return signal_power/(noise+interference)

def path_loss(h_t: float, h_r: float, dist: float) -> float:
    '''
    returns g(dist), i.e., gain in W
    using two-ray-ground model for path loss
    '''
    return pow(h_t*h_r, 2)/pow(dist, 4)



class LinkType(Enum):
    DOWNLINK = 'DL'
    UPLINK = 'UL'

class HexPos:
    def __init__(self, i: int, j: int):
        self.i = i
        self.j = j

    def __repr__(self) -> str:
        return f"HexPos({self.i}, {self.j})"

    @property
    def first(self) -> int:
        return self.i

    @property
    def second(self) -> int:
        return self.j

    def __eq__(self, other: "HexPos") -> bool:
        if isinstance(other, HexPos):
            return self.i == other.i and self.j == other.j
        else:
            return False

    def __hash__(self) -> int:
        return hash((self.i, self.j))

    def __add__(self, other: "HexPos") -> "HexPos":
        if isinstance(other, HexPos):
            return HexPos(self.i + other.i, self.j + other.j)
        else:
            raise TypeError("Can only add HexPos with another HexPos")

    def to_tuple(self) -> tuple[int, int]:
        return (self.i, self.j)

    def to_standard_coordinate(self) -> np.ndarray:
        return (self.i, self.j) @ basis

    @classmethod
    def from_standard_coordinate(cls, coord: np.ndarray) -> "HexPos":
        if coord.shape != (2,):
            raise ValueError("Coordinate must be a 2-element array or tuple")

        i, j = coord @ la.inv(basis)
        return cls(int(round(i)), int(round(j)))


class Direction:
    null            = HexPos(0, 0)
    top_right       = HexPos(1, 0)
    top             = HexPos(0, 1)
    top_left        = HexPos(-1, 1)
    bottom_left     = HexPos(-1, 0)
    bottom          = HexPos(0, -1)
    bottom_right    = HexPos(1, -1)
    vectors_to_neighbor = [null, top_right, top, top_left, bottom_left, bottom, bottom_right]

class Cell:
    colormap = cmap

    def __init__(self, pos: HexPos):
        self.pos: HexPos = pos
        self.id: int = (pos.first + pos.second * 8 ) % 19 + 1
        self.color: np.ndarray = self.__class__.colormap(self.id - 1)
        self._real_pos: Optional[np.ndarray] = None

    def __repr__(self) -> str:
        return f"Cell_{self.id}@{self.pos.to_tuple()}"

    def __eq__(self, other: "Cell") -> bool:
        if isinstance(other, Cell):
            return self.pos == other.pos
        else:
            return False

    def __hash__(self) -> int:
        return hash(self.pos)

    @property
    def real_pos(self) -> np.ndarray:
        if self._real_pos is None:
            self._real_pos = self.pos.to_standard_coordinate()
        return self._real_pos

    def get_neighbors(self, dist: int) -> set["Cell"]:
        '''
        Get neighboring cells within `dist` steps from the current cell (including itself)
        '''
        if dist < 0:
            raise ValueError
        elif dist==0:
            return {self}
        else:
            return {elem for vector in Direction.vectors_to_neighbor
                    for elem in Cell(self.pos+vector).get_neighbors(dist-1)}

class MobileDevice:
    def __init__(self, id_: int, bandwidth: int, pos: Optional[np.ndarray] = None, cells: Optional[set[Cell]] = None):
        self.id: int = id_
        self.bandwidth: int = bandwidth
        self.channel_noise: float = noise(bandwidth)
        self.pos: np.ndarray = pos if pos is not None else self.generate_pos(cells)
        self.v: np.ndarray = np.array((0,0))
        self.remaining_time: float|int = 0
        self.connected_bs: Optional[Cell] = None

    def __repr__(self) -> str:
        return f"MS({self.id}){self.pos}"

    def __eq__(self, other: "MobileDevice") -> bool:
        if isinstance(other, MobileDevice):
            return self.id == other.id
        else:
            return False

    def __hash__(self) -> int:
        return hash(self.id)

    def sinr(self, linktype: LinkType, cells: set[Cell], mss: list['MobileDevice']) -> float:
        return calculate_SINR(
            DL_or_UL= linktype,
            bs=self.connected_bs,
            ms=self,
            cells= cells,
            mss= mss
        )

    def shannon_capacity(self, linktype: LinkType, cells: set[Cell], mss: list['MobileDevice']) -> float: # bit/s
        return self.bandwidth * log2(1+self.sinr(linktype, cells, mss))

    @staticmethod
    def generate_pos(cells: set[Cell]) -> np.ndarray:
        vs = np.array(random.sample(cell_basis, k=2))
        while 114514:
            x = rng.uniform(0, 1, size=2)
            if not np.all(x == 0):
                break
        return random.sample(list(cells), 1)[0].real_pos + x @ vs

    def get_nearest_cell_approx(self) -> Cell:
        return Cell(HexPos.from_standard_coordinate(self.pos))

    def get_nearest_cell(self) -> Cell:
        approx_nearest_cell = self.get_nearest_cell_approx()
        neighboring_cells = approx_nearest_cell.get_neighbors(1)
        nearest_cell: Cell = min(
            neighboring_cells,
            key = lambda neighbor: la.norm(neighbor.real_pos - self.pos)
        )
        return nearest_cell


    def move(self, time_unit: float|int, min_speed: int, max_speed: int, min_t: int, max_t: int, core_cells: set[Cell], mapped_cells: set[Cell]) -> tuple[set[Cell], set[Cell]]:
        '''
        may have a side effect
        '''
        if self.remaining_time <= 0:
            angle = rng.uniform(0, 2*np.pi)
            length = rng.uniform(min_speed, max_speed)
            self.v = np.array([length*np.cos(angle), length*np.sin(angle)])
            self.remaining_time = time_unit * rng.uniform(int(np.ceil(min_t/time_unit)), max_t//time_unit)
            assert self.remaining_time>0
        self.pos += self.v * time_unit
        self.remaining_time -= time_unit

        # check if te MS has entered a new cell that is not in core_cells but in mapped_cells
        current_cell: Cell = self.get_nearest_cell()
        if current_cell not in core_cells:
            core_cells.add(current_cell)
            mapped_cells |= {neighbor for neighbor in current_cell.get_neighbors(2) if neighbor not in core_cells}

        return core_cells, mapped_cells


    def connect_to_best_bs(self, DL_or_UL: LinkType, cells: set[Cell], mss: list['MobileDevice'], received_power_cache: dict[Cell, dict['MobileDevice', float]]) -> Optional[tuple[int, int]]:
        '''
        Find the base station (cell) with the highest SINR w.r.t. a mobile device
        and check if handoff is needed.
        If there is no connected BS, then connect to the best BS.

        Return the handoff (if happened) in the form of (source cell ID, destination cell ID).
        '''
        if not cells:
            return None

        best_cell: Cell = max(
            cells,
            key=lambda cell: calculate_SINR(DL_or_UL, cell, self, cells, mss, received_power_cache)
        )

        if self.connected_bs is None:
            # unconnected
            self.connected_bs = best_cell
            return None
        elif best_cell == self.connected_bs:
            # no handoff required
            return None
        else:
            # handoff required
            old_bs_id = self.connected_bs.id
            self.connected_bs = best_cell
            return (old_bs_id, best_cell.id)




def _received_power(power_tx: float, h_t: float, h_r: float, dist: float) -> float:
    return power_tx*gain_tx*gain_rx*path_loss(h_t, h_r, dist)

def received_power(DL_or_UL: LinkType, bs: Cell, ms: MobileDevice, received_power_cache: Optional[dict[Cell, dict[MobileDevice, float]]] = None) -> float:
    assert isinstance(bs, Cell) and isinstance(ms, MobileDevice), (bs,ms)
    if DL_or_UL not in [LinkType.DOWNLINK, LinkType.UPLINK]:
        raise ValueError("DL_or_UL should be either 'DL' or 'UL'")

    if received_power_cache is not None:
        if bs not in received_power_cache:
            received_power_cache[bs]=dict()
        if ms in received_power_cache[bs]:
            return received_power_cache[bs][ms]

    if DL_or_UL == LinkType.DOWNLINK:
        dist = distance(bs.real_pos, ms.pos)
        rpow = _received_power(power_bs, height_bs, height_ms, dist)
    else:
        dist = distance(ms.pos, bs.real_pos)
        rpow = _received_power(power_ms, height_ms, height_bs, dist)

    if received_power_cache is not None:
        received_power_cache[bs][ms] = rpow

    return rpow

def calculate_SINR(DL_or_UL: LinkType, bs: Cell, ms: MobileDevice, cells: set[Cell], mss: list[MobileDevice], received_power_cache: Optional[dict[Cell, dict[MobileDevice, float]]] = None, sum_cache: Optional[dict[Cell|MobileDevice, float]] = None) -> float:
    if DL_or_UL not in [LinkType.DOWNLINK, LinkType.UPLINK]:
        raise ValueError("DL_or_UL should be either 'DL' or 'UL'")

    if DL_or_UL == LinkType.DOWNLINK:
        signal = received_power(DL_or_UL, bs, ms, received_power_cache)
        if sum_cache is not None and ms in sum_cache:
            all_received_signals = sum_cache[ms]
        else:
            all_received_signals = sum(received_power(DL_or_UL, detected_bs, ms, received_power_cache) for detected_bs in cells)
            if sum_cache is not None:
                sum_cache[ms] = all_received_signals
        interference = all_received_signals - signal

    else:
        signal = received_power(DL_or_UL, bs, ms, received_power_cache)
        if sum_cache is not None and bs in sum_cache:
            all_received_signals = sum_cache[bs]
        else:
            all_received_signals = sum(received_power(DL_or_UL, bs, detected_ms, received_power_cache) for detected_ms in mss)
            if sum_cache is not None:
                sum_cache[bs] = all_received_signals
        interference = all_received_signals - signal

    return SINR(signal, interference, ms.channel_noise)







''' Simulation '''

def hex_vertices(cell_center: npt.ArrayLike, radius: int|float) -> np.ndarray:
    return map_(lambda angle: (cell_center[0]+radius*np.cos(angle), cell_center[1]+radius*np.sin(angle)),
                np.linspace(0, 2*np.pi, 7))


def _simulate(sim_time: int|float, time_unit: int|float, DL_or_UL: LinkType, cells: set[Cell], mss: list[MobileDevice], CBR_per_ms: float, bs_buf_size: float, throughput: Optional[dict[MobileDevice, float]]=None) -> float:
    '''
    Returns the bits loss probability.
    '''
    print(f"{sim_time=}, {CBR_per_ms=}, {bs_buf_size=}")

    if throughput is None:
        throughput: dict[MobileDevice, float] = {ms:ms.shannon_capacity(LinkType.DOWNLINK, cells, mss) for ms in mss} # bit/s

    bs_buf: dict[MobileDevice, float] = {ms:0 for ms in mss}

    bits_loss: float = 0
    total_bits: float = CBR_per_ms * len(mss) * ((sim_time//time_unit) * time_unit)

    t: int|float = 0
    while t < sim_time:

        buf_space_left = max(0, bs_buf_size - sum(bs_buf.values()))
        #assert buf_space_left >= 0, buf_space_left

        for ms in mss:
            # send bits in the buffer first
            can_send = throughput[ms]*time_unit
            sent = min(bs_buf[ms], can_send)
            can_send -= sent
            bs_buf[ms] -= sent
            buf_space_left += sent

            # then send bits that arrive at the current timestamp
            arrived_bits = CBR_per_ms*time_unit
            sent = min(arrived_bits, can_send)
            to_store_in_buf = min(arrived_bits-sent, buf_space_left)
            bs_buf[ms] += to_store_in_buf
            buf_space_left -= to_store_in_buf
            bits_loss += arrived_bits - sent - to_store_in_buf
            #print(f"space left: {buf_space_left}")

        #print(bs_buf.values())
        t += time_unit
    print(f"# lost bits = {bits_loss}, # total bits arrived = {total_bits}")
    print(f"bits loss probability = {bits_loss/total_bits}")
    print("===================================================")

    return bits_loss/total_bits


def simulate(task_name: str = "", sim_time: int|float = 1000, time_unit: int|float = 1, DL_or_UL: LinkType = LinkType.DOWNLINK, num_mobile_devices: int = 50, bs_buf_size: float = 6e6, cbrs: list[int|float] = [6e6]):

    bandwidth_per_ms = bandwidth_per_cell/num_mobile_devices # Hz

    central_cell: Cell = Cell(HexPos(0,0))
    cells: set[Cell] = central_cell.get_neighbors(2)
    mss: list[MobileDevice] = [MobileDevice(id_=i, bandwidth=bandwidth_per_ms, cells = {central_cell}) for i in range(num_mobile_devices)]
    assert len(cells) == 19

    for ms in mss:
        ms.connected_bs = central_cell

    throughput: dict[MobileDevice, float] = {ms:ms.shannon_capacity(LinkType.DOWNLINK, cells, mss) for ms in mss} # bit/s

    #plt.rcParams['text.usetex'] = True
    fig = plt.figure(figsize=(16, 8))
    gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1], hspace=0.4)


    ax1 = fig.add_subplot(gs[:, 0])  # `:` means spanning across both rows
    ax1.scatter([ms.pos[0] for ms in mss],[ms.pos[1] for ms in mss], color='red', label='Mobile Device', alpha=0.7, s=4)
    ax1.scatter(*central_cell.real_pos, color='blue', label='Central BS', marker='x', s=100)

    hex_v = hex_vertices(central_cell.real_pos, l)
    ax1.plot(*hex_v.T, color=central_cell.color, linewidth=1)

    ax1.set_xlim(left=-l*1.1, right=l*1.1)
    ax1.set_ylim(bottom=-l*1.1, top=l*1.1)
    ax1.set_xlabel('X Position (meter)')
    ax1.set_ylabel('Y Position (meter)')
    ax1.set_title('Location of the BS and Mobile Devices in the Central Cell')
    ax1.legend()
    ax1.grid()
    ax1.set_aspect('equal')

    ax2 = fig.add_subplot(gs[0, 1])  # `:` means spanning across both rows
    ax2.scatter([distance(central_cell.real_pos, ms.pos) for ms in mss],
                [throughput[ms] for ms in mss],
                color='blue', s=4)
    ax2.set_xlabel('Distance to the Central BS (meter)')
    ax2.set_ylabel('Shannon Capacity (bit/s)')
    ax2.set_title('Shannon Capacities of the Mobile Devices')
    ax2.grid()
    formatter = ScalarFormatter(useMathText=False, useOffset=False)
    formatter.set_scientific(False)
    ax2.yaxis.set_major_formatter(formatter)


    bits_loss_prob: list[float] = [
        _simulate(
            sim_time= sim_time,
            time_unit= time_unit,
            DL_or_UL= DL_or_UL,
            cells= cells,
            mss= mss,
            CBR_per_ms= cbr,
            bs_buf_size= bs_buf_size,
            throughput= throughput
        )
        for cbr in cbrs
    ]

    ax3 = fig.add_subplot(gs[1, 1])  # `:` means spanning across both rows
    bars = ax3.bar([str(int(cbr)) for cbr in cbrs], bits_loss_prob, color='red', width=0.3, linewidth=0)
    for bar in bars:
        yval = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width() / 2, yval, f'{yval:.2f}', ha='center', va='bottom', fontsize=10)
    ax3.set_xlabel('CBR per Mobile Device (bit/s)')
    ax3.set_ylabel('Bits Loss Probability')
    ax3.set_title('Bits Loss Probabilities in Different Traffic Loads')
    ax3.grid()


    plt.savefig((task_name if task_name else "figures") + ".png", format="png")
    plt.show()



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
            simulate(
                task_name="1",
                sim_time=sim_time,
                time_unit=time_unit,
                DL_or_UL=LinkType.DOWNLINK,
                num_mobile_devices=num_mobile_devices,
                bs_buf_size=bs_buf_size,
                cbrs = CBR_per_ms
            )
        case 2:
            pass
        case _:
            print("Error: Unknown problem number.")
            print_help()
