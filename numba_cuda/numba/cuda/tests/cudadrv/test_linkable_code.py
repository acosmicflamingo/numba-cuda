# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

import pytest

from numba import cuda
from numba.cuda.cudadrv.linkable_code import LinkableCode
from numba.cuda.testing import skip_on_cudasim

from .cuda_bin_dirs import (
    TEST_BIN_DIR,
    test_device_functions_a,
    test_device_functions_cubin,
    test_device_functions_cu,
    test_device_functions_fatbin,
    test_device_functions_o,
    test_device_functions_ptx,
    test_device_functions_ltoir,
)


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
