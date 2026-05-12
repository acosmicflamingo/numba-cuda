# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

import numpy as np
from numba.cuda.cudadrv import driver
from numba import cuda


class TestHostAlloc:
    def test_host_alloc_driver(self):
        n = 32
        mem = cuda.current_context().memhostalloc(n, mapped=True)

        dtype = np.dtype(np.uint8)
        ary = np.ndarray(shape=n // dtype.itemsize, dtype=dtype, buffer=mem)

        magic = 0xAB
        driver.device_memset(mem, magic, n)

        assert np.all(ary == magic)

        ary.fill(n)

        recv = np.empty_like(ary)

        driver.device_to_host(recv, mem, ary.size)

        assert np.all(ary == recv)
        assert np.all(recv == n)

    def test_host_alloc_pinned(self):
        ary = cuda.pinned_array(10, dtype=np.uint32)
        ary.fill(123)
        assert all(ary == 123)
        devary = cuda.to_device(ary)
        driver.device_memset(devary, 0, driver.device_memory_size(devary))
        assert all(ary == 123)
        devary.copy_to_host(ary)
        assert all(ary == 0)

    def test_host_alloc_mapped(self):
        ary = cuda.mapped_array(10, dtype=np.uint32)
        ary.fill(123)
        assert all(ary == 123)
        driver.device_memset(ary, 0, driver.device_memory_size(ary))
        assert all(ary == 0)
        assert sum(ary != 0) == 0

    def test_host_operators(self):
        for ary in [
            cuda.mapped_array(10, dtype=np.uint32),
            cuda.pinned_array(10, dtype=np.uint32),
        ]:
            ary[:] = range(10)
            assert sum(ary + 1) == 55
            assert sum((ary + 1) * 2 - 1) == 100
            assert sum(ary < 5) == 5
            assert sum(ary <= 5) == 6
            assert sum(ary > 6) == 3
            assert sum(ary >= 6) == 4
            assert sum(ary**2) == 285
            assert sum(ary // 2) == 20
            assert sum(ary / 2.0) == 22.5
            assert sum(ary % 2)
