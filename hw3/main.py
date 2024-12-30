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
from scipy.constants import k # Boltzmann constant
import sys
from math import sqrt, log10
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
bandwidth = 1e7 # Hz
power_bs = dBm_to_W(33) # W
power_ms = dBm_to_W(23) # W
gain_tx = from_dB(14) # ratio
gain_rx = from_dB(14) # ratio
isd = 500. # meter
l = isd/sqrt(3) # length of a side of every hexagonal cell (in meter)
height_bs = 1.5+50 # meter
height_ms = 1.5 # meter
noise = k*temp*bandwidth # thermal noise power (in W)

min_speed = 1 # m/s
max_speed = 15 # m/s
min_t = 1 # s
max_t = 6 # s

sim_time = 900 # s
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

sum_cache = dict()





''' data structures and functions '''
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
    def __init__(self, id_: int, pos: Optional[np.ndarray] = None, cells: Optional[set[Cell]] = None):
        self.id: int = id_
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

def _received_power(power_tx: float, h_t: float, h_r: float, dist: float) -> float:
    return power_tx*gain_tx*gain_rx*path_loss(h_t, h_r, dist)

def received_power(DL_or_UL: LinkType, bs: Cell, ms: MobileDevice, received_power_cache: Optional[dict[Cell, dict[MobileDevice, float]]] = None) -> float:
    assert isinstance(bs, Cell) and isinstance(ms, MobileDevice)
    if DL_or_UL not in [LinkType.DOWNLINK, LinkType.UPLINK]:
        raise ValueError("DL_or_UL should be either 'DL' or 'UL'")

    if received_power_cache is not None:
        return received_power_cache[bs][ms]

    if DL_or_UL == LinkType.DOWNLINK:
        dist = distance(bs.real_pos, ms.pos)
        return _received_power(power_bs, height_bs, height_ms, dist)
    else:
        dist = distance(ms.pos, bs.real_pos)
        return _received_power(power_ms, height_ms, height_bs, dist)

def calculate_SINR(DL_or_UL: LinkType, bs: Cell, ms: MobileDevice, cells: set[Cell], mss: list[MobileDevice], received_power_cache: dict[Cell, dict[MobileDevice, float]]) -> float:
    global sum_cache
    if DL_or_UL not in [LinkType.DOWNLINK, LinkType.UPLINK]:
        raise ValueError("DL_or_UL should be either 'DL' or 'UL'")

    if DL_or_UL == LinkType.DOWNLINK:
        signal = received_power(DL_or_UL, bs, ms, received_power_cache)
        if ms not in sum_cache:
            sum_cache[ms] = sum(received_power(DL_or_UL, detected_bs, ms, received_power_cache) for detected_bs in cells)
        interference = sum_cache[ms] - signal
    else:
        signal = received_power(DL_or_UL, bs, ms, received_power_cache)
        if bs not in sum_cache:
            sum_cache[bs] = sum(received_power(DL_or_UL, bs, detected_ms, received_power_cache) for detected_ms in mss)
        interference = sum_cache[bs] - signal
    return SINR(signal, interference, noise)


def connect_to_best_bs(DL_or_UL: LinkType, ms: MobileDevice, cells: set[Cell], mss: list[MobileDevice], received_power_cache: dict[Cell, dict[MobileDevice, float]]) -> Optional[tuple[int, int]]:
    '''
    Find the base station (cell) with the highest SINR w.r.t. a mobile device
    and check if handoff is needed.
    If there is no connected BS, then connect to the best BS.

    Return the handoff (if happened) in the form of (source cell ID, destination cell ID).
    '''
    if not cells:
        return None

    #start_time = time.time()
    best_cell: Cell = max(
        cells,
        key=lambda cell: calculate_SINR(DL_or_UL, cell, ms, cells, mss, received_power_cache)
    )
    #print(f"Time to perform best_cell: {(time.time()-start_time):.6f} seconds")

    if ms.connected_bs is None:
        # unconnected
        ms.connected_bs = best_cell
        return None
    elif best_cell == ms.connected_bs:
        # no handoff required
        return None
    else:
        # handoff required
        old_bs_id = ms.connected_bs.id
        ms.connected_bs = best_cell
        return (old_bs_id, best_cell.id)

def update_all_ms_connection(DL_or_UL: LinkType, cells: set[Cell], mss: list[MobileDevice], received_power_cache: dict[Cell, dict[MobileDevice, float]]) -> list[tuple[int, int]]:
    '''
    Connect every mobile device to the BS with the highest SINR.
    Return all handoffs whose elements are in the form of (source cell ID, destination cell ID).
    '''
    return filter_(lambda handoff: handoff is not None,
                   [connect_to_best_bs(DL_or_UL, ms, cells, mss, received_power_cache) for ms in mss])

def update_all_ms_position(time_unit: float|int, min_speed: int, max_speed: int, min_t: int, max_t: int, core_cells: set[Cell], mapped_cells: set[Cell], mss: list[MobileDevice]) -> tuple[set[Cell], set[Cell]]:
    for ms in mss:
        core_cells, mapped_cells = ms.move(
            time_unit = time_unit,
            min_speed = min_speed,
            max_speed = max_speed,
            min_t = min_t,
            max_t = max_t,
            core_cells = core_cells,
            mapped_cells = mapped_cells
        )
    return core_cells, mapped_cells


''' Simulation '''

def hex_vertices(cell_center: npt.ArrayLike, radius: int|float) -> np.ndarray:
    return map_(lambda angle: (cell_center[0]+radius*np.cos(angle), cell_center[1]+radius*np.sin(angle)),
                np.linspace(0, 2*np.pi, 7))

def save_handoffs_to_csv(filename: str, handoffs: list[tuple[int|float, int, int]]):
    '''
    Save all handoff events to a CSV file
    '''
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Time', 'Source Cell ID', 'Destination Cell ID'])
        for handoff in handoffs:
            writer.writerow([str(handoff[0])+'s', handoff[1], handoff[2]])

    print(f"Handoffs saved to {filename}")

def simulate(task_name: str = "", sim_time = 900, time_unit: int|float = 1, DL_or_UL: LinkType = LinkType.UPLINK, num_mobile_devices: int = 100):
    print(f"{sim_time=}, {DL_or_UL=}")

    core_cells: set[Cell] = Cell(HexPos(0,0)).get_neighbors(2)
    if num_mobile_devices == 1:
        mss: list[MobileDevice] = [MobileDevice(id_ = 0, pos = np.array([250, 0], dtype='d'))]
    else:
        mss: list[MobileDevice] = [MobileDevice(id_=i, cells = core_cells) for i in range(num_mobile_devices)]

    mapped_cells: set[Cell] = {neighbor for cell in core_cells for neighbor in cell.get_neighbors(2)} | \
                            {neighbor for ms in mss for neighbor in ms.get_nearest_cell().get_neighbors(2)}
    assert len(core_cells) == 19

    fig, ax = plt.subplots(figsize=(8, 8))

    ax.scatter([ms.pos[0] for ms in mss],[ms.pos[1] for ms in mss], color='red', label='Mobile Device', alpha=0.7, s=2)
    ax.scatter([cell.real_pos[0] for cell in mapped_cells], [cell.real_pos[1] for cell in mapped_cells], c=[cell.color for cell in mapped_cells], label='BS', marker='x', s=25)

    for cell in mapped_cells:
        hex_v = hex_vertices(cell.real_pos, l)
        ax.plot(*hex_v.T, color=cell.color, linewidth=1)
        ax.text(cell.real_pos[0]+35, cell.real_pos[1]+35, str(cell.id), color=cell.color, fontsize=8, ha='center', va='center')

    ax.set_xlabel('X Position (meter)')
    ax.set_ylabel('Y Position (meter)')
    ax.set_title('Location of BS and Mobile Devices in Each Cell')
    ax.legend()
    ax.set_aspect('equal')
    plt.savefig((task_name + '_' if task_name else "") + "initial_map.png", format="png")

    print("Close the map to start the simulation ...")
    plt.show()

    t: int|float = 0
    handoff_count: int = 0
    handoffs: list[tuple[int|float, int, int]] = list()

    while t < sim_time:
        print(f"====================== t = {t} =======================")

        sum_cache = dict()

        start_time = time.time()
        received_power_cache: dict[Cell, dict[MobileDevice, float]] = {
            cell: {
                ms: received_power(DL_or_UL, cell, ms) for ms in mss
            }
            for cell in mapped_cells
        }
        print(f"Time to calculate received power list: {(time.time()-start_time):.6f} s")

        start_time = time.time()
        handoff_events = update_all_ms_connection(DL_or_UL, mapped_cells, mss, received_power_cache)
        print(f"Time to perform update_all_ms_connection: {(time.time()-start_time):.6f} s")

        handoff_count += len(handoff_events)
        handoffs.extend([(t, *handoff) for handoff in handoff_events])

        t += time_unit

        start_time = time.time()
        core_cells, mapped_cells = update_all_ms_position(
            time_unit = time_unit,
            min_speed = min_speed,
            max_speed = max_speed,
            min_t = min_t,
            max_t = max_t,
            core_cells = core_cells,
            mapped_cells = mapped_cells,
            mss = mss
        )
        print(f"Time to move all mobile devices: {(time.time()-start_time):.6f} s")

        for n in mss[0].get_nearest_cell().get_neighbors(2):
            assert n in mapped_cells

        print(f"{handoff_count=}")
        print(f"{handoff_events=}")

    print("======================================================")
    print(f"total number of handoffs: {handoff_count}")
    save_handoffs_to_csv(filename = (task_name + '_' if task_name else "") + "handoffs.csv",
                         handoffs = handoffs)



    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter([ms.pos[0] for ms in mss],[ms.pos[1] for ms in mss], color='red', label='Mobile Device', alpha=0.7, s=2)
    ax.scatter([cell.real_pos[0] for cell in mapped_cells], [cell.real_pos[1] for cell in mapped_cells], c=[cell.color for cell in mapped_cells], label='BS', marker='x', s=25)

    for cell in mapped_cells:
        hex_v = hex_vertices(cell.real_pos, l)
        ax.plot(*hex_v.T, color=cell.color, linewidth=1)
        ax.text(cell.real_pos[0]+35, cell.real_pos[1]+35, str(cell.id), color=cell.color, fontsize=8, ha='center', va='center')

    ax.set_xlabel('X Position (meter)')
    ax.set_ylabel('Y Position (meter)')
    ax.set_title('Location of BS and Mobile Devices in Each Cell')
    ax.legend()
    ax.set_aspect('equal')
    plt.savefig((task_name + '_' if task_name else "") + "final_map.png", format="png")
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
                task_name="q1",
                sim_time=sim_time,
                time_unit=time_unit,
                DL_or_UL=LinkType.DOWNLINK,
                num_mobile_devices=1
            )
        case 2:
            simulate(
                task_name="bonus",
                sim_time=sim_time,
                time_unit=time_unit,
                DL_or_UL=LinkType.UPLINK,
                num_mobile_devices=100
            )
        case _:
            print("Error: Unknown problem number.")
            print_help()
