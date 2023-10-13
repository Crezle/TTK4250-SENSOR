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


@dataclass
class MeasPos(NamedArray):
    x: AtIndex[0]
    y: AtIndex[1]

    isclutter: MetaData[bool] = field(default=False)
