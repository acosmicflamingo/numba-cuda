# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

from numba import cuda
from numba.cuda.testing import skip_on_cudasim, skip_unless_cc_53


class TestIsFP16Supported:
    def test_is_fp16_supported(self):
        assert cuda.is_float16_supported()

    @skip_on_cudasim("fp16 not available in sim")
    @skip_unless_cc_53
    def test_device_supports_float16(self):
        assert cuda.get_current_device().supports_float16
