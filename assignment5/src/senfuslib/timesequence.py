from bisect import bisect_left, insort_right, bisect_right
from dataclasses import InitVar, dataclass, field
from typing import (Callable, ClassVar, Generic, Iterable, Optional, TypeVar,
                    Union, Any)
from itertools import islice
import numpy as np
from collections.abc import Mapping
T = TypeVar("T")


class NoDefault:
    pass


@dataclass(repr=False)
class TimeSequence(Mapping, Generic[T]):
    """A class for storing a sequence of objects in time order"""

    times: list[float] = field(default_factory=list, init=False)
    t_min: Optional[float] = field(default=None, init=False)
    t_max: Optional[float] = field(default=None, init=False)

    init_iter: InitVar[Iterable[tuple[float, T]]] = None

    __value_dict: ClassVar[dict[float, T]]
    __array_dict_cache: ClassVar[dict[str, np.ndarray]]

    def __post_init__(self, iter: Optional[Iterable[tuple[float, T]]]):
        """Create a new TimeSeries from an iterable of (ts, value) pairs"""
        self.__value_dict = dict()
        self.__array_dict_cache = dict()
        for ts, value in iter or []:
            self.insert(ts, value)

    @property
    def values(self) -> list[T]:
        """Get the values of the time series, in time order"""
        return [self.__value_dict[ts] for ts in self.times]

    def items(self) -> Iterable[tuple[float, T]]:
        return ((t, self.__value_dict[t]) for t in self.times)

    def insert(self, ts: float, value: T):
        """Insert a new value into the time series"""
        ts = float(ts)
        if ts in self.__value_dict:
            raise ValueError(f"Timestamp {ts} already exists")
        self.__value_dict[ts] = value

        lo, hi = (0, len(self.times))
        if self.t_max is None or ts > self.t_max:
            self.t_max = ts
            lo = hi
        if self.t_min is None or ts < self.t_min:
            self.t_min = ts
            hi = lo
        insort_right(self.times, ts, lo=lo, hi=hi)

        self.__array_dict_cache = dict()

    def pop(self, idx: int) -> tuple[float, T]:
        t = self.times.pop(idx)
        val = self.__value_dict.pop(t)
        if t == self.t_min:
            self.t_min = self.times[0] if self.times else None
        if t == self.t_max:
            self.t_max = self.times[-1] if self.times else None

        self.__array_dict_cache = dict()
        return t, val

    def pop_t(self, ts: float) -> tuple[float, T]:
        idx = bisect_left(self.times, ts)
        return self.pop(idx)

    def get_item(self, idx: int) -> tuple[float, T]:
        return self.times[idx], self.__value_dict[self.times[idx]]

    def get_t(self, t: float, default=NoDefault) -> T:
        if default is NoDefault:
            return self.__value_dict[t]
        return self.__value_dict.get(t, default)

    def map(self, f: Callable[[T], T]) -> 'TimeSequence':
        return TimeSequence((t, f(v)) for t, v in self.items())

    def field_as_array(self, field: str) -> np.ndarray:
        if field in self.__array_dict_cache:
            return self.__array_dict_cache[field]

        def get_field(v, field):
            for f in field.split('.'):
                v = getattr(v, f)
            return v

        arr = np.stack([get_field(v, field) for v in self.values])
        self.__array_dict_cache[field] = arr
        return arr

    def values_as_array(self) -> np.ndarray:
        return self.field_as_array(None)

    def slice_idx(self, start=0, stop=None, step=None
                  ) -> 'TimeSequence':
        stop = stop or len(self.times)
        stop = stop if stop >= 0 else len(self.times) + stop
        return TimeSequence(islice(self.items(), start, stop, step))

    def slice_time(self, start=None, stop=None, min_step=0,
                   lopen=False, ropen=True) -> Iterable[tuple[float, T]]:
        def gen():
            _start = start or self.t_min
            _stop = stop or self.t_max
            start_idx = (bisect_right if lopen else bisect_left)(
                self.times, _start)
            stop_idx = (bisect_left if ropen else bisect_right)(
                self.times, _stop)
            prev = float('-inf')
            for ts, value in self.slice_idx(start_idx, stop_idx).items():
                if ts - prev >= min_step:
                    yield ts, value
                    prev = ts
        return TimeSequence(gen())

    def get_min_max(self, key: Callable[[T], Any], return_time=False):
        vals = [key(v) for v in self.values]
        argmin = np.argmin(vals)
        argmax = np.argmax(vals)
        if return_time:
            return self.times[argmin], self.times[argmax]
        else:
            return vals[argmin], vals[argmax]

    def __iter__(self) -> Iterable[float]:
        return iter(self.times)

    def __getitem__(self, idx: Union[slice, int, float]) -> tuple[float, T]:
        if isinstance(idx, int):
            return self.get_item(idx)
        elif isinstance(idx, float):
            return self.get_t(idx)
        elif isinstance(idx, slice):
            if any(isinstance(i, float) for i in (idx.start, idx.stop, idx.step)):
                return self.slice_time(idx.start, idx.stop, idx.step)
            else:
                return self.slice_idx(idx.start, idx.stop, idx.step)
        else:
            raise TypeError(f"Invalid index type {type(idx)}")

    def __contains__(self, t: float) -> bool:
        return t in self.__value_dict

    def __len__(self) -> int:
        return len(self.times)
