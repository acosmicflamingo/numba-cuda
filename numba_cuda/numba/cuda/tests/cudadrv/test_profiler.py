# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

import pytest

from numba import cuda
from numba.cuda.testing import skip_on_cudasim


@skip_on_cudasim("CUDA Profiler unsupported in the simulator")
class TestProfiler:
    def test_profiling(self):
        with cuda.profiling():
            a = cuda.device_array(10)
            del a

        with cuda.profiling():
            a = cuda.device_array(100)
            del a
