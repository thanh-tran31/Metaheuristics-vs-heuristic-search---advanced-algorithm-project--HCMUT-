"""1-based Binary Min-Heap data structures and optimizations for 8-connected A*.

Heap array representation follows standard 1-based indexing (slot 0 is unused/sentinel,
root is at index 1, active elements occupy indices 1 .. size):
    - parent_index(i)      = i // 2
    - left_child_index(i)  = 2 * i
    - right_child_index(i) = 2 * i + 1
"""

from __future__ import annotations

from typing import Generic, TypeVar

T = TypeVar("T")

PRIORITY_SCALE: int = 10_000_000
TIE_BITS: int = 40
MAX_TIE_INT: int = (1 << TIE_BITS) - 1

HEAP_MODES: tuple[str, ...] = ("fast", "decrease_key", "lazy_tuple")


def parent_index(i: int) -> int:
    """Return 1-based parent index `i // 2` of child node `i` (`i >= 2`)."""
    return i // 2


def left_child_index(i: int) -> int:
    """Return 1-based left child index `2 * i` of node `i` (`i >= 1`)."""
    return 2 * i


def right_child_index(i: int) -> int:
    """Return 1-based right child index `2 * i + 1` of node `i` (`i >= 1`)."""
    return 2 * i + 1


def compute_packing_params(num_cells: int) -> tuple[int, int, int]:
    """Compute bit-packing shifts and masks sized safely for `num_cells`.

    Layout of packed non-negative integer key (from MSB to LSB):
        [ f_int (quantized f * 1e7) | tie_int (40 bits) | node_idx (idx_bits) ]
    """
    idx_bits = max(20, num_cells.bit_length())
    idx_mask = (1 << idx_bits) - 1
    f_shift = idx_bits + TIE_BITS
    return idx_bits, idx_mask, f_shift


def pack_priority_key(
    f_int: int,
    g_val: float,
    push_order: int,
    node_idx: int,
    tie_mode: str,
    idx_bits: int,
    f_shift: int,
) -> int:
    """Pack `(f, tie_breaker, node_idx)` into one non-negative integer (`>= 0`)."""
    if tie_mode == "high_g":
        g_scaled = int(g_val * PRIORITY_SCALE + 0.5)
        tie_int = MAX_TIE_INT - g_scaled if g_scaled < MAX_TIE_INT else 0
    elif tie_mode == "low_g":
        g_scaled = int(g_val * PRIORITY_SCALE + 0.5)
        tie_int = g_scaled if g_scaled < MAX_TIE_INT else MAX_TIE_INT
    else:
        tie_int = push_order if push_order < MAX_TIE_INT else MAX_TIE_INT
    return (f_int << f_shift) | (tie_int << idx_bits) | node_idx


class BinaryMinHeap(Generic[T]):
    """Classic 1-based Binary Min-Heap (`parent = i // 2`, `children = 2*i, 2*i + 1`).

    Index 0 is a dummy slot (`None`), and active heap items live at `data[1 .. size]`.
    Uses standard 3-way swaps during sift-up / sift-down, serving as the textbook
    baseline heap structure before low-level optimizations.
    """

    __slots__ = ("data", "heap_ops")

    def __init__(self) -> None:
        self.data: list[T] = [None]  # type: ignore[list-item]  # 1-based: index 0 unused
        self.heap_ops: int = 0

    def __len__(self) -> int:
        return len(self.data) - 1

    def __bool__(self) -> bool:
        return len(self.data) > 1

    def peek(self) -> T:
        """Return minimum element at root index 1 in O(1)."""
        if len(self.data) <= 1:
            raise IndexError("peek from an empty BinaryMinHeap")
        return self.data[1]

    def push(self, item: T) -> None:
        """Insert `item` at the end and sift up via `parent_idx = i // 2` in O(log N)."""
        data = self.data
        data.append(item)
        self.heap_ops += 1

        i = len(data) - 1
        while i > 1:
            p_idx = i // 2
            if data[i] < data[p_idx]:  # type: ignore[operator]
                data[i], data[p_idx] = data[p_idx], data[i]
                i = p_idx
            else:
                break

    def pop(self) -> T:
        """Remove and return root `data[1]`, sifting down via `2*i` and `2*i + 1`."""
        data = self.data
        size = len(data) - 1
        if size <= 0:
            raise IndexError("pop from an empty BinaryMinHeap")

        self.heap_ops += 1
        min_item = data[1]
        last_item = data.pop()
        size -= 1
        if size > 0:
            data[1] = last_item
            i = 1
            while True:
                left_idx = 2 * i
                if left_idx > size:
                    break
                right_idx = 2 * i + 1
                smallest_idx = left_idx
                if right_idx <= size and data[right_idx] < data[left_idx]:  # type: ignore[operator]
                    smallest_idx = right_idx

                if data[smallest_idx] < data[i]:  # type: ignore[operator]
                    data[i], data[smallest_idx] = data[smallest_idx], data[i]
                    i = smallest_idx
                else:
                    break

        return min_item


class FastIntMinHeap:
    """Optimized 1-based Integer Min-Heap with Sentinel `data[0] = -1` and Hole-Sifting.

    Optimizations over textbook `BinaryMinHeap` while keeping 1-based tree layout:
      1. Sentinel at index 0 (`data[0] = -1`): since all packed priority keys are `>= 0`,
         `key < data[i // 2]` automatically terminates at `i = 1` (`i // 2 == 0`) without
         checking `i > 1` on every sift-up iteration (Sedgewick sentinel technique).
      2. Floyd Hole-Sifting: shifts parent/child values by 1 assignment per tree level
         instead of 3-way tuple swaps, writing the target key once at its final slot.
    """

    __slots__ = ("data", "heap_ops")

    def __init__(self) -> None:
        self.data: list[int] = [-1]  # index 0 sentinel < all non-negative keys
        self.heap_ops: int = 0

    def __len__(self) -> int:
        return len(self.data) - 1

    def __bool__(self) -> bool:
        return len(self.data) > 1

    def push(self, key: int) -> None:
        """Insert non-negative integer `key` using sentinel sift-up (`i // 2`)."""
        data = self.data
        i = len(data)
        data.append(key)
        self.heap_ops += 1

        while key < data[p_idx := i // 2]:
            data[i] = data[p_idx]
            i = p_idx
        data[i] = key

    def pop(self) -> int:
        """Extract minimum root `data[1]` using hole-sifting (`2*i`, `2*i + 1`)."""
        data = self.data
        size = len(data) - 1
        if size <= 0:
            raise IndexError("pop from an empty FastIntMinHeap")

        self.heap_ops += 1
        min_key = data[1]
        last_key = data.pop()
        size -= 1
        if size > 0:
            i = 1
            half = size // 2
            while i <= half:
                child_idx = 2 * i
                right_idx = child_idx + 1
                child_key = data[child_idx]
                if right_idx <= size and data[right_idx] < child_key:
                    child_idx = right_idx
                    child_key = data[right_idx]

                if child_key < last_key:
                    data[i] = child_key
                    i = child_idx
                else:
                    break
            data[i] = last_key

        return min_key


class IndexedMinHeap:
    """1-based Indexed Binary Min-Heap supporting O(log |OPEN|) in-place `decrease_key`.

    Uses 1-based heap slots `1 .. size` (`keys[0] = -1` sentinel, `nodes[0] = -1`):
      - `parent_index = i // 2`
      - `left_child_index = 2 * i`
      - `right_child_index = 2 * i + 1`
      - `pos[node_idx]` stores the exact 1-based heap index (`1 .. size`), or `0` when
        `node_idx` is NOT in OPEN (eliminating `-1` sentinel initialization).
    """

    __slots__ = ("keys", "nodes", "pos", "heap_ops", "decrease_key_count")

    def __init__(self, num_cells: int) -> None:
        self.keys: list[int] = [-1]   # 1-based: index 0 is -1 sentinel
        self.nodes: list[int] = [-1]  # 1-based: root at index 1
        self.pos: list[int] = [0] * num_cells  # 0 = absent, 1..size = 1-based heap slot
        self.heap_ops: int = 0
        self.decrease_key_count: int = 0

    def __len__(self) -> int:
        return len(self.keys) - 1

    def __bool__(self) -> bool:
        return len(self.keys) > 1

    def contains(self, node_idx: int) -> bool:
        """Return True in O(1) if `node_idx` is currently in OPEN (`pos[node] >= 1`)."""
        return self.pos[node_idx] != 0

    def push_or_decrease(self, node_idx: int, key: int) -> bool:
        """Insert `node_idx` or decrease its key in-place in O(log |OPEN|)."""
        existing_idx = self.pos[node_idx]
        if existing_idx == 0:
            idx = len(self.keys)
            self.keys.append(key)
            self.nodes.append(node_idx)
            self._sift_up(idx, node_idx, key)
            self.heap_ops += 1
            return True

        if key < self.keys[existing_idx]:
            self._sift_up(existing_idx, node_idx, key)
            self.heap_ops += 1
            self.decrease_key_count += 1
            return True

        return False

    def pop_min(self) -> tuple[int, int]:
        """Remove and return `(node_idx, key)` at root index 1 in O(log |OPEN|)."""
        keys = self.keys
        nodes = self.nodes
        pos = self.pos
        size = len(keys) - 1
        if size <= 0:
            raise IndexError("pop_min from an empty IndexedMinHeap")

        min_node = nodes[1]
        min_key = keys[1]
        pos[min_node] = 0
        self.heap_ops += 1

        last_key = keys.pop()
        last_node = nodes.pop()
        size -= 1
        if size > 0:
            self._sift_down(1, last_node, last_key, size)

        return min_node, min_key

    def _sift_up(self, i: int, item_node: int, item_key: int) -> None:
        """Move `(item_node, item_key)` upward from 1-based slot `i` via `i // 2`."""
        keys = self.keys
        nodes = self.nodes
        pos = self.pos

        # Sentinel keys[0] = -1 guarantees termination when i == 1 (p_idx == 0)
        while item_key < keys[p_idx := i // 2]:
            p_node = nodes[p_idx]
            keys[i] = keys[p_idx]
            nodes[i] = p_node
            pos[p_node] = i
            i = p_idx

        keys[i] = item_key
        nodes[i] = item_node
        pos[item_node] = i

    def _sift_down(self, i: int, item_node: int, item_key: int, size: int) -> None:
        """Move `(item_node, item_key)` downward from 1-based slot `i` via `2*i, 2*i+1`."""
        keys = self.keys
        nodes = self.nodes
        pos = self.pos
        half = size // 2

        while i <= half:
            child_idx = 2 * i
            right_idx = child_idx + 1
            child_key = keys[child_idx]
            if right_idx <= size and keys[right_idx] < child_key:
                child_idx = right_idx
                child_key = keys[right_idx]

            if child_key < item_key:
                child_node = nodes[child_idx]
                keys[i] = child_key
                nodes[i] = child_node
                pos[child_node] = i
                i = child_idx
            else:
                break

        keys[i] = item_key
        nodes[i] = item_node
        pos[item_node] = i
