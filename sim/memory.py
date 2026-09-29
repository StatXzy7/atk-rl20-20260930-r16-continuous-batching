"""固定大小 KV 页池。

内存占用时机（在 engine 中严格执行）：
  * prefill token：在该 chunk 开始执行*之前*，按"该 chunk 写完之后"
    的 token 总数在页边界向上取整申请页。即跨页的 chunk 一次性拿到
    跨越的所有目标页（页边界正确申请），绝不写入未申请的页。
  * decode token：同样在执行前申请；对会越过页边界的那次 decode，
    先拿到新页再写入。
  * 完成、拒绝、取消、抢占（重算）时，释放请求的全部私有页。

本模型中所有页均为请求私有（无 prefix sharing），因此不存在外部碎片；
碎片化只体现为每个请求最后一页未写满的槽位（内部碎片）。
"""

from __future__ import annotations

from typing import Iterable, List, Set


class OutOfPagesError(RuntimeError):
    """页池没有足够空闲页。"""


class PagePool:
    def __init__(self, total_pages: int, page_size: int) -> None:
        self.total_pages = total_pages
        self.page_size = page_size
        self._free: Set[int] = set(range(total_pages))
        self._in_use: Set[int] = set()
        self.peak_used: int = 0

    # ---- 基本分配 / 释放 ----
    def allocate(self, n: int) -> List[int]:
        if n < 0:
            raise ValueError("申请页数不能为负")
        if n > len(self._free):
            raise OutOfPagesError(
                f"需要 {n} 页，仅剩 {len(self._free)} 页（共 {self.total_pages} 页）"
            )
        pages = sorted(self._free)[:n]
        for p in pages:
            self._free.remove(p)
            self._in_use.add(p)
        self.peak_used = max(self.peak_used, len(self._in_use))
        return pages

    def free_many(self, pages: Iterable[int]) -> None:
        for p in pages:
            if p not in self._in_use:
                raise RuntimeError(f"页 {p} 并未处于占用状态，不能释放")
            self._in_use.discard(p)
            self._free.add(p)

    # ---- 查询 ----
    @property
    def free_pages(self) -> int:
        return len(self._free)

    @property
    def used_pages(self) -> int:
        return len(self._in_use)

    def internal_fragmentation(self, used_tokens: int) -> int:
        """已分配页中的未写满槽位数（内部碎片）。

        used_tokens: 当前所有被持有页对应请求占用的 token 总数。
        """
        return len(self._in_use) * self.page_size - used_tokens
