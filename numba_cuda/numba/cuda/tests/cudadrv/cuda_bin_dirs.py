# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

import os

TEST_BIN_DIR = os.getenv("NUMBA_CUDA_TEST_BIN_DIR")


def _generate_bin_path(filename):
    return os.path.join(TEST_BIN_DIR, filename) if TEST_BIN_DIR else None


test_device_functions_a = _generate_bin_path("test_device_functions.a")
test_device_functions_cubin = _generate_bin_path("test_device_functions.cubin")
test_device_functions_cu = _generate_bin_path("test_device_functions.cu")
test_device_functions_fatbin = _generate_bin_path(
    "test_device_functions.fatbin"
)
test_device_functions_fatbin_multi = _generate_bin_path(
    "test_device_functions_multi.fatbin"
)
test_device_functions_o = _generate_bin_path("test_device_functions.o")
test_device_functions_ptx = _generate_bin_path("test_device_functions.ptx")
test_device_functions_ltoir = _generate_bin_path("test_device_functions.ltoir")
