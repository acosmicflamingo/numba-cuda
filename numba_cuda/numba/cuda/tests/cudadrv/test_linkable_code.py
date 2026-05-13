# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

import os

import pytest

from numba import cuda
from numba.cuda.cudadrv.linkable_code import LinkableCode
from numba.cuda.testing import skip_on_cudasim

TEST_BIN_DIR = os.getenv("NUMBA_CUDA_TEST_BIN_DIR")


def bin_path(filename):
    return os.path.join(TEST_BIN_DIR, filename) if TEST_BIN_DIR else None


test_device_functions_a = bin_path("test_device_functions.a")
test_device_functions_cubin = bin_path("test_device_functions.cubin")
test_device_functions_cu = bin_path("test_device_functions.cu")
test_device_functions_fatbin = bin_path("test_device_functions.fatbin")
test_device_functions_fatbin_multi = bin_path(
    "test_device_functions_multi.fatbin"
)
test_device_functions_o = bin_path("test_device_functions.o")
test_device_functions_ptx = bin_path("test_device_functions.ptx")
test_device_functions_ltoir = bin_path("test_device_functions.ltoir")


class TestLinkableCode:
    @skip_on_cudasim(reason="Simulator does not support linkable code")
    @pytest.mark.skipif(
        not TEST_BIN_DIR, reason="necessary binaries not generated."
    )
    @pytest.mark.parametrize(
        "path,kind",
        [
            (test_device_functions_a, cuda.Archive),
            (test_device_functions_cubin, cuda.Cubin),
            (test_device_functions_cu, cuda.CUSource),
            (test_device_functions_fatbin, cuda.Fatbin),
            (test_device_functions_o, cuda.Object),
            (test_device_functions_ptx, cuda.PTXSource),
            (test_device_functions_ltoir, cuda.LTOIR),
        ],
    )
    def test_linkable_code_from_path_or_obj(self, path, kind):
        obj = LinkableCode.from_path_or_obj(path)
        assert isinstance(obj, kind)

        # test identity of from_path_or_obj
        obj2 = LinkableCode.from_path_or_obj(obj)
        assert obj2 is obj
