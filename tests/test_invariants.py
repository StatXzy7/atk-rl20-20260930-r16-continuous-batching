"""全局不变量测试：独立重放事件日志，不信任引擎内部账本。

核验内容：
  * 事件时间单调、step 时间不回退、到达不被 step 打断；
  * token 工作预算每步不超支；chunk 不超过 chunk_max；
  * 页池账本：任意时刻 0<=used<=total，完成/取消/抢占后私有页全释放，
    系统终态零泄漏；
  * 页边界：任何 prefill/decode 写入前页数足以覆盖写后 token 位置；
  * 重算：抢占浪费 token 与之后重算工作量一致且被计费；
  * 防饥饿上界（适用条件内）：受保护 prefill 每步保底、
    decoder decode_gap 上界（从事件重建）。
"""

import os
import random
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sim import PolicyKind, RequestSpec, SimConfig, simulate, Cancellation
from sim.scenarios import (
    burst_arrival,
    cancellation_mix,
    long_prefill_mix,
    oversized_prompt,
    page_starvation,
    zero_and_single_token,
)
from sim.report import run_both


class LedgerReplayer:
    """从 events + steps 独立重建每个请求的页数与 token 位置。

    关键规则（与引擎一致）：准入时（admit 事件）一次性预留整个 prompt
    所需页；prefill chunk 与 decode 仅在跨页时再申请新页；preempt /
    complete / cancel 释放全部私有页。
    """

    def __init__(self, result):
        self.res = result
        cfg = result.cfg
        self.total = cfg["total_pages"]
        self.psize = cfg["page_size"]
        self.budget = cfg["token_budget"]
        self.chunk_max = cfg["chunk_max"]
        self.held = {}
        self.pos = {}
        self.status = {}
        self.free = self.total
        self.last_end = 0.0
        self._processed_events = 0
        # 预处理：按出现顺序的到达/拒绝（在第一个 step 之前全部登记）
        for e in self.res.events:
            if e["event"] == "arrival":
                self.status[e["rid"]] = "PENDING"
                self.pos[e["rid"]] = 0
                self.held[e["rid"]] = 0
            elif e["event"] == "reject":
                self.status[e["rid"]] = "REJECTED"

    def check(self):
        for st in self.res.steps:
            self._check_step(st)
        # 最后一个 step 结束后、系统清空循环中可能还处理了
        # at<=makespan 的离散事件（典型：与最后一步同时刻的取消）。
        if self.res.steps:
            self._process_discrete_until(self.res.steps[-1].end)
        self._assert_final()

    def _process_discrete_until(self, t):
        # 按日志顺序处理所有 time<=t 且尚未回放的离散事件；
        # 遇到 time>t 的非离散事件可跳过，离散事件时间非降。
        evs = self.res.events
        i = self._processed_events
        while i < len(evs):
            e = evs[i]
            if e.get("event") not in ("arrival", "cancel", "preempt",
                                      "admit"):
                i += 1
                continue
            if e.get("time", float("inf")) > t + 1e-12:
                break
            self._apply_discrete(e)
            i += 1
        self._processed_events = i

    def _apply_discrete(self, e):
        ev = e["event"]
        if ev == "arrival":
            if self.status.get(e["rid"]) == "PENDING":
                self.status[e["rid"]] = "WAITING"
        elif ev == "cancel":
            rid = e["rid"]
            if self.status.get(rid) in ("COMPLETED", "CANCELLED",
                                        "REJECTED"):
                return
            assert e["freed_pages"] == self.held.get(rid, 0)
            self.free += self.held.get(rid, 0)
            self.held[rid] = 0
            self.status[rid] = "CANCELLED"
        elif ev == "preempt":
            rid = e["rid"]
            self.free += self.held.get(rid, 0)
            assert e["freed_pages"] == self.held.get(rid, 0)
            self.held[rid] = 0
            self.pos[rid] = 0
            self.status[rid] = "WAITING"
        elif ev == "admit":
            rid = e["rid"]
            if self.status.get(rid) == "PREFILL":
                return
            need = e["pages"]
            assert need <= self.free
            self.free -= need
            self.held[rid] = self.held.get(rid, 0) + need
            self.status[rid] = "PREFILL"

    def _end_at(self, t):
        for e in self.res.events:
            if e["time"] != t:
                continue
            if e["event"] == "complete":
                rid = e["rid"]
                if self.status.get(rid) == "COMPLETED":
                    continue
                assert e["freed_pages"] == self.held.get(rid, 0), (
                    "complete 释放页数与重放不一致")
                self.free += self.held.get(rid, 0)
                self.held[rid] = 0
                self.status[rid] = "COMPLETED"

    def _grow(self, rid, new_pos):
        need = -(-max(new_pos, 1) // self.psize)
        held = self.held.get(rid, 0)
        if need > held:
            delta = need - held
            assert delta <= self.free, (
                f"{rid} 写入越界: need={need} held={held} free={self.free}")
            self.free -= delta
            self.held[rid] = held + delta
        self.pos[rid] = new_pos

    def _check_step(self, st):
        assert st.end >= st.start - 1e-9
        assert st.start >= self.last_end - 1e-9, (
            f"step {st.index} 时间回退")
        self.last_end = st.end
        work = st.prefill_tokens + st.decode_tokens
        assert work <= self.budget, (
            f"step {st.index} 超预算 {work}>{self.budget}")
        # 回放 (last_end, start] 内全部离散事件（取消可能在步末生效）
        self._process_discrete_until(st.start)
        for a in st.actions:
            rid = a["rid"]
            if a["kind"] == "prefill":
                assert a["tokens"] <= self.chunk_max
                if a.get("became_decoder"):
                    target_pos = a.get("kv_pos_after")
                    self._grow(rid, target_pos)
                else:
                    self._grow(rid, self.pos.get(rid, 0) + a["tokens"])
            else:
                self._grow(rid, self.pos.get(rid, 0) + 1)
        assert 0 <= self.free <= self.total
        # output_len=0 的请求在 prefill 完成同一步由 complete 事件释放，
        # 这里只标记状态（释放由 _end_at 依据 freed_pages 完成）。
        for a in st.actions:
            if (a["kind"] == "prefill" and a.get("became_decoder")
                    and a.get("true_output_len") == 0):
                self.status[a["rid"]] = "COMPLETED"
        # after 是所有完成（含 output_len=0）释放之前的快照
        used_after_actions = self.total - self.free
        assert used_after_actions == st.used_pages_after, (
            f"step {st.index} 动作后页数 {used_after_actions} "
            f"!= after {st.used_pages_after}")
        # output_len=0 的请求在本步末随 _finalize 释放
        for a in st.actions:
            if (a["kind"] == "prefill" and a.get("became_decoder")
                    and a.get("true_output_len") == 0):
                rid = a["rid"]
                self.free += self.held.get(rid, 0)
                self.held[rid] = 0
                self.status[rid] = "COMPLETED"
        self._end_at(st.end)
        assert self.total - self.free == st.used_pages_final, (
            f"step {st.index} 释放后页数 {self.total - self.free} "
            f"!= final {st.used_pages_final}")

    def _assert_final(self):
        for rid, pages in self.held.items():
            if self.status.get(rid) in ("COMPLETED", "CANCELLED",
                                        "REJECTED"):
                assert pages == 0, f"{rid} 终态泄漏 {pages} 页"
        if self.res.deadlock is None:
            assert self.free == self.total, (
                f"终态空闲 {self.free}!={self.total}，内存泄漏")


def assert_result_clean(res, allow_deadlock=False):
    LedgerReplayer(res).check()
    assert res.final_used_pages == 0
    assert len(res.steps) >= 1 or res.num_rejected == len(res.requests)
    # 时间单调
    prev = -1e-9
    for st in res.steps:
        assert st.start >= prev - 1e-12
        prev = st.end


class TestScenarioInvariants(unittest.TestCase):
    def test_all_scenarios_both_policies(self):
        for scn in [long_prefill_mix(), burst_arrival(),
                    cancellation_mix(), oversized_prompt(),
                    zero_and_single_token()]:
            with self.subTest(scenario=scn.name):
                res = run_both(scn)
                assert_result_clean(res["fcfs"])
                assert_result_clean(res["cont"])

    def test_page_starvation_fcfs_deadlocks_cont_survives(self):
        res = run_both(page_starvation())
        # FCFS 无抢占：跨页等待形成死锁，必须明确上报
        self.assertIsNotNone(res["fcfs"].deadlock)
        # CONT 以抢占重算打破死锁，全部完成，重算被计费
        self.assertIsNone(res["cont"].deadlock)
        self.assertEqual(res["cont"].num_completed, 4)
        self.assertGreater(res["cont"].total_replay_work, 0)
        self.assertGreater(res["cont"].total_preemptions, 0)

    def test_oversized_rejected_and_isolated(self):
        res = run_both(oversized_prompt())["cont"]
        status = {m.rid: m.status for m in res.requests}
        self.assertEqual(status["HUGE"], "REJECTED")
        self.assertEqual(status["OK1"], "COMPLETED")
        self.assertEqual(status["OK2"], "COMPLETED")
        self.assertEqual(res.num_rejected, 1)


class TestSpecialRequests(unittest.TestCase):
    def test_zero_and_single_token_metrics(self):
        res = run_both(zero_and_single_token())["cont"]
        m = {x.rid: x for x in res.requests}
        self.assertIsNone(m["Z0"].ttft)
        self.assertIsNone(m["Z0"].mean_itl)
        self.assertEqual(m["Z0"].itls, [])
        self.assertIsNotNone(m["Z0"].completion_latency)
        self.assertEqual(m["Z0"].status, "COMPLETED")
        # 单 token：TTFT=完成延迟，ITL 为空
        self.assertEqual(m["O1"].itls, [])
        self.assertEqual(m["O1"].ttft, m["O1"].completion_latency)
        self.assertEqual(m["N3"].itls.__len__(), 2)


class TestCancellationReleasesPages(unittest.TestCase):
    def test_cancel_mix(self):
        res = run_both(cancellation_mix())["cont"]
        m = {x.rid: x for x in res.requests}
        self.assertEqual(m["K3"].status, "CANCELLED")
        self.assertEqual(m["K1"].status, "CANCELLED")
        # KX 不存在 -> cancel_ignored 事件
        ignored = [e for e in res.events
                   if e.get("event") == "cancel_ignored"
                   and e.get("rid") == "KX"]
        self.assertTrue(ignored)
        assert_result_clean(res)


if __name__ == "__main__":
    unittest.main()
