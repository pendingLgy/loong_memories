# Copyright [2026] [longgyx]
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""
 雪花 ID 生成器：64 位整数，高位时间戳保证趋势递增，便于做主键与排序。

 结构：1 位符号（恒为 0）+ 41 位毫秒时间戳 + 5 位数据中心 + 5 位工作节点 + 12 位序列。
 同一毫秒内最多产生 4096 个 ID，超出则等待下一毫秒。
"""


from __future__ import annotations

import threading
import time

# 起始纪元（2024-01-01 00:00:00 UTC，毫秒）。
EPOCH_MS = 1704067200000

# 各字段位宽与掩码。
_WORKER_BITS = 5
_DATACENTER_BITS = 5
_SEQUENCE_BITS = 12
_MAX_WORKER = (1 << _WORKER_BITS) - 1
_MAX_DATACENTER = (1 << _DATACENTER_BITS) - 1
_MAX_SEQUENCE = (1 << _SEQUENCE_BITS) - 1
_WORKER_SHIFT = _SEQUENCE_BITS
_DATACENTER_SHIFT = _SEQUENCE_BITS + _WORKER_BITS
_TIMESTAMP_SHIFT = _SEQUENCE_BITS + _WORKER_BITS + _DATACENTER_BITS


class SnowflakeGenerator:
    """线程安全的雪花 ID 生成器；同一实例在毫秒内递增，实例间靠节点位区分。"""

    def __init__(self, datacenter_id: int = 0, worker_id: int = 0) -> None:
        if not 0 <= datacenter_id <= _MAX_DATACENTER:
            raise ValueError("datacenter_id 超出范围")
        if not 0 <= worker_id <= _MAX_WORKER:
            raise ValueError("worker_id 超出范围")
        self._datacenter_id = datacenter_id
        self._worker_id = worker_id
        self._sequence = 0
        self._last_ms = -1
        self._lock = threading.Lock()

    def _timestamp(self) -> int:
        return int(time.time() * 1000) - EPOCH_MS

    def next_id(self) -> int:
        """生成下一个雪花 ID。"""
        with self._lock:
            now = self._timestamp()
            if now < self._last_ms:
                # 时钟回拨：以最后一次时间继续，避免产生更小的 ID。
                now = self._last_ms
            if now == self._last_ms:
                self._sequence = (self._sequence + 1) & _MAX_SEQUENCE
                if self._sequence == 0:
                    while self._timestamp() <= self._last_ms:
                        time.sleep(0.001)
                    now = self._timestamp()
            else:
                self._sequence = 0
            self._last_ms = now
            return (
                    (now << _TIMESTAMP_SHIFT)
                    | (self._datacenter_id << _DATACENTER_SHIFT)
                    | (self._worker_id << _WORKER_SHIFT)
                    | self._sequence
            )

    def __call__(self) -> int:
        return self.next_id()


# 默认生成器：进程内单例，节点位可用环境变量区分。
_DEFAULT_GENERATOR = SnowflakeGenerator()


def next_id() -> int:
    """用默认生成器取一个雪花 ID。"""
    return _DEFAULT_GENERATOR.next_id()
