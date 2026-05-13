# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

import os
import io
import contextlib

import pytest

from numba import cuda
from numba.cuda import config, get_current_device
from numba.cuda.cudadrv.driver import _Linker, _have_nvjitlink
from numba.cuda.testing import skip_on_cudasim
from numba.cuda.tests.support import cuda_test_setup

from .cuda_bin_dirs import (
    TEST_BIN_DIR,
    test_device_functions_a,
    test_device_functions_cubin,
    test_device_functions_cu,
    test_device_functions_fatbin,
    test_device_functions_fatbin_multi,
    test_device_functions_fatbin_multi,
    test_device_functions_o,
    test_device_functions_ptx,
    test_device_functions_ltoir,
)

if TEST_BIN_DIR:
    require_cuobjdump = (
        test_device_functions_fatbin_multi,
        test_device_functions_fatbin,
        test_device_functions_o,
    )


@pytest.mark.skipif(
    not TEST_BIN_DIR or not _have_nvjitlink(),
    reason="nvJitLink not installed or new enough (>12.3)",
)
@skip_on_cudasim("Linking unsupported in the simulator")
class TestLinker:
    files = [
        test_device_functions_a,
        test_device_functions_cubin,
        test_device_functions_cu,
        test_device_functions_fatbin,
        test_device_functions_o,
        test_device_functions_ptx,
    ]

    @pytest.mark.parametrize("file", files)
    def test_nvjitlink_add_file_guess_ext_linkable_code(self, file):
        linker = _Linker(cc=get_current_device().compute_capability)
        linker.add_file_guess_ext(file)

    def test_nvjitlink_test_add_file_guess_ext_invalid_input(self):
        with open(test_device_functions_cubin, "rb") as f:
            content = f.read()

        linker = _Linker(cc=get_current_device().compute_capability)
        with pytest.raises(
            TypeError, match="Expected path to file or a LinkableCode"
        ):
            # Feeding raw data as bytes to add_file_guess_ext should raise,
            # because there's no way to know what kind of file to treat it as
            linker.add_file_guess_ext(content)

    @pytest.mark.parametrize("lto", [True, False])
    @pytest.mark.parametrize("file", files)
    def test_nvjitlink_jit_with_linkable_code(self, lto, file):
        sig = "uint32(uint32, uint32)"
        add_from_numba = cuda.declare_device("add_from_numba", sig)

        @cuda.jit(link=[file], lto=lto)
        def kernel(result):
            result[0] = add_from_numba(1, 2)

        result = cuda.device_array(1)
        kernel[1, 1](result)
        assert result[0] == 3

    def test_nvjitlink_jit_with_invalid_linkable_code(self):
        with open(test_device_functions_cubin, "rb") as f:
            content = f.read()

        with pytest.raises(
            TypeError, match="Expected path to file or a LinkableCode"
        ):

            @cuda.jit("void()", link=[content])
            def kernel():
                pass


@pytest.mark.skipif(
    not TEST_BIN_DIR or not _have_nvjitlink(),
    reason="nvJitLink not installed or new enough (>12.3)",
)
@skip_on_cudasim("Linking unsupported in the simulator")
class TestLinkerDumpAssembly:
    @pytest.fixture(autouse=True)
    def configure(self, cuda_test_setup):
        self._prev_dump_assembly = config.DUMP_ASSEMBLY
        config.DUMP_ASSEMBLY = True

        yield

        config.DUMP_ASSEMBLY = self._prev_dump_assembly

    @pytest.mark.parametrize(
        "file",
        [
            test_device_functions_cu,
            test_device_functions_ltoir,
            test_device_functions_fatbin_multi,
        ],
    )
    def test_nvjitlink_jit_with_linkable_code_lto_dump_assembly(self, file):
        if (
            file in require_cuobjdump
            and os.getenv("NUMBA_CUDA_TEST_WHEEL_ONLY") is not None
        ):
            pytest.skip(reason="wheel-only environments do not have cuobjdump")

        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            sig = "uint32(uint32, uint32)"
            add_from_numba = cuda.declare_device("add_from_numba", sig)

            @cuda.jit(link=[file], lto=True)
            def kernel(result):
                result[0] = add_from_numba(1, 2)

            result = cuda.device_array(1)
            kernel[1, 1](result)
            assert result[0] == 3

        assert "ASSEMBLY (AFTER LTO)" in f.getvalue()

    @pytest.mark.parametrize(
        "file",
        [
            test_device_functions_a,
            test_device_functions_cubin,
            test_device_functions_fatbin,
            test_device_functions_o,
            test_device_functions_ptx,
        ],
    )
    def test_nvjitlink_jit_with_linkable_code_lto_dump_assembly_warn(
        self, file
    ):
        if (
            file in require_cuobjdump
            and os.getenv("NUMBA_CUDA_TEST_WHEEL_ONLY") is not None
        ):
            pytest.skip(reason="wheel-only environments do not have cuobjdump")

        sig = "uint32(uint32, uint32)"
        add_from_numba = cuda.declare_device("add_from_numba", sig)

        @cuda.jit(link=[file], lto=True)
        def kernel(result):
            result[0] = add_from_numba(1, 2)

        result = cuda.device_array(1)
        func = kernel[1, 1]
        with pytest.warns(
            UserWarning,
            match="it is not optimizable at link time, and `ignore_nonlto == True`",
        ):
            func(result)
        assert result[0] == 3
