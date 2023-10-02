from dataclasses import dataclass, field
from senfuslib import AtIndex, NamedArray, MetaData
import numpy as np


@dataclass
class StateCV(NamedArray):
    x: AtIndex[0]
    y: AtIndex[1]
    u: AtIndex[2]
    v: AtIndex[3]

    pos: AtIndex[0:2] = field(init=False)
    vel: AtIndex[2:4] = field(init=False)

    prev_mode: MetaData[int] = field(default=None)
    
    def add(self, other): # Added to make addition possible
        return StateCV(
            self.x + other.x,
            self.y + other.y,
            self.u + other.u,
            self.v + other.v
        )


@dataclass
class MeasPos(NamedArray):
    x: AtIndex[0]
    y: AtIndex[1]

    time: MetaData[float] = field(default=None)
