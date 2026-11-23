"""Model outputs per tile window, kept in memory so a preview that changes only
strength, or a preview or export over tiles already rendered, skips the model.

A model's output depends only on its input window, so the key names what fills
that window: the source, every earlier step (model and params), this step's
model, and the window's span in this step's input coordinates. The value is
`model.run(window)` before strength is blended in, exactly as the model
returned it, so a cached tile gives bit-identical results.

Previews read and write. Full exports only read: an export of a 24 MP photo
would push every preview tile out of the cache.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from collections.abc import Hashable

import numpy as np


class TileCache:
    def __init__(self, max_bytes: int):
        self.max_bytes = max_bytes
        self._items: OrderedDict[Hashable, np.ndarray] = OrderedDict()
        self._bytes = 0
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def get(self, key: Hashable) -> np.ndarray | None:
        with self._lock:
            value = self._items.get(key)
            if value is None:
                self.misses += 1
                return None
            self._items.move_to_end(key)
            self.hits += 1
            return value

    def put(self, key: Hashable, value: np.ndarray) -> None:
        if value.base is not None:  # a view: don't keep its parent array alive
            value = value.copy()
        size = value.nbytes
        if size > self.max_bytes:
            return
        value.setflags(write=False)  # callers must not change a cached tile
        with self._lock:
            old = self._items.pop(key, None)
            if old is not None:
                self._bytes -= old.nbytes
            self._items[key] = value
            self._bytes += size
            while self._bytes > self.max_bytes:
                _, dropped = self._items.popitem(last=False)
                self._bytes -= dropped.nbytes

    @property
    def bytes(self) -> int:
        return self._bytes

    def __len__(self) -> int:
        return len(self._items)
