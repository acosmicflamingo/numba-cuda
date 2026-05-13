# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

from numba.cuda.cudadrv import nvrtc
from numba.cuda.testing import skip_on_cudasim


@skip_on_cudasim("NVVM Driver unsupported in the simulator")
class TestArchOption:
    def test_get_arch_option(self):
        # Test returning the nearest lowest arch.
        assert nvrtc.get_arch_option(7, 5) == "compute_75"
        assert nvrtc.get_arch_option(7, 7) == "compute_75"
        assert nvrtc.get_arch_option(8, 5) == "compute_80"
        assert nvrtc.get_arch_option(9, 1) == "compute_90"
        # Test known arch.
        supported_ccs = nvrtc.get_supported_ccs()
        for cc in supported_ccs:
            assert nvrtc.get_arch_option(*cc) == "compute_%d%d" % cc
        assert (
            nvrtc.get_arch_option(1000, 0) == "compute_%d%d" % supported_ccs[-1]
        )
