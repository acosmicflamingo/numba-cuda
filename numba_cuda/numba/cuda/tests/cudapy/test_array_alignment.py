# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

import re
import itertools

import numpy as np
import pytest

from numba import cuda
from numba.cuda import HAS_NUMBA

if HAS_NUMBA:
    from numba.core.errors import TypingError
else:
    from numba.cuda.core.errors import TypingError
from numba.cuda.testing import (
    skip_on_cudasim,
    skip_unless_cudasim,
)


# Set to true if you want to see dots printed for each subtest.
NOISY = False


# In order to verify the alignment of the local and shared memory arrays, we
# inspect the LLVM IR of the generated kernel using the following regexes.

# Shared memory example:
# @"_cudapy_smem_38" = addrspace(3) global [1 x i8] undef, align 16
SMEM_PATTERN = re.compile(
    r'^@"_cudapy_smem_\d+".*?align (\d+)',
    re.MULTILINE,
)

# Local memory example:
# %"_cudapy_lmem" = alloca [1 x i8], align 64
LMEM_PATTERN = re.compile(
    r'^\s*%"_cudapy_lmem".*?align (\d+)',
    re.MULTILINE,
)


DTYPES = [np.uint8, np.uint32, np.uint64]

# Add in some record dtypes with and without alignment.
for align in (True, False):
    DTYPES += [
        np.dtype(
            [
                ("a", np.uint8),
                ("b", np.int32),
                ("c", np.float64),
            ],
            align=align,
        ),
        np.dtype(
            [
                ("a", np.uint32),
                ("b", np.uint8),
            ],
            align=align,
        ),
        np.dtype(
            [
                ("a", np.uint8),
                ("b", np.int32),
                ("c", np.float64),
                ("d", np.complex64),
                ("e", (np.uint8, 5)),
            ],
            align=align,
        ),
    ]

# N.B. We name the test class TestArrayAddressAlignment to avoid name conflict
#      with the test_alignment.TestArrayAlignment class.


@skip_on_cudasim("Array alignment not supported on cudasim")
class TestArrayAddressAlignment:
    """
    Test cuda.local.array and cuda.shared.array support for an alignment
    keyword argument.
    """

    def _do_test(self, array_types, shape, dtype, alignment):
        which, array_type = array_types

        @cuda.jit
        def f(loc, shrd, which):
            i = cuda.grid(1)
            if which == 0:
                local_array = cuda.local.array(
                    shape=shape,
                    dtype=dtype,
                    alignment=alignment,
                )
                if i == 0:
                    loc[0] = local_array.ctypes.data
            else:
                shared_array = cuda.shared.array(
                    shape=shape,
                    dtype=dtype,
                    alignment=alignment,
                )
                if i == 0:
                    shrd[0] = shared_array.ctypes.data

        loc = np.zeros(1, dtype=np.uint64)
        shrd = np.zeros(1, dtype=np.uint64)
        f[1, 1](loc, shrd, which)

        kernel = f.overloads[f.signatures[0]]
        llvm_ir = kernel.inspect_llvm()

        if alignment is None:
            if which == 0:
                # Local memory shouldn't have any alignment information
                # when no alignment is specified.
                match = LMEM_PATTERN.findall(llvm_ir)
                assert len(match) == 0
            else:
                # Shared memory should at least have a power-of-two
                # alignment when no alignment is specified.
                match = SMEM_PATTERN.findall(llvm_ir)
                assert len(match) == 1

                alignment = int(match[0])
                # Verify alignment is a power of two.
                assert alignment & (alignment - 1) == 0
        else:
            # Verify alignment is in the LLVM IR.
            if which == 0:
                match = LMEM_PATTERN.findall(llvm_ir)
                assert len(match) == 1
                actual_alignment = int(match[0])
                assert alignment == actual_alignment
            else:
                match = SMEM_PATTERN.findall(llvm_ir)
                assert len(match) == 1
                actual_alignment = int(match[0])
                assert alignment == actual_alignment

            # Also verify that the address of the array is aligned.
            # If this fails, there problem is likely with NVVM.
            address = loc[0] if which == 0 else shrd[0]
            alignment_mod = int(address % alignment)
            assert alignment_mod == 0

        if NOISY:
            print(".", end="", flush=True)

    @pytest.mark.parametrize("array_types", [(0, "local"), (1, "shared")])
    @pytest.mark.parametrize("shape", [1, 8, 50])
    @pytest.mark.parametrize("dtype", DTYPES)
    @pytest.mark.parametrize("alignment", [None, 16, 64, 256])
    def test_array_alignment_1d(self, array_types, shape, dtype, alignment):
        self._do_test(array_types, shape, dtype, alignment)

    @pytest.mark.parametrize("array_types", [(0, "local"), (1, "shared")])
    @pytest.mark.parametrize("shape", [(2, 3)])
    @pytest.mark.parametrize("dtype", DTYPES)
    @pytest.mark.parametrize("alignment", [None, 16, 64, 256])
    def test_array_alignment_2d(self, array_types, shape, dtype, alignment):
        self._do_test(array_types, shape, dtype, alignment)

    @pytest.mark.parametrize("array_types", [(0, "local"), (1, "shared")])
    @pytest.mark.parametrize("shape", [(2, 3, 4), (1, 4, 5)])
    @pytest.mark.parametrize("dtype", DTYPES)
    @pytest.mark.parametrize("alignment", [None, 16, 64, 256])
    def test_array_alignment_3d(self, array_types, shape, dtype, alignment):
        self._do_test(array_types, shape, dtype, alignment)

    @pytest.mark.parametrize("array_types", [(0, "local"), (1, "shared")])
    @pytest.mark.parametrize("shape", [1, 50])
    @pytest.mark.parametrize("dtype", [np.uint8, np.uint64])
    @pytest.mark.parametrize(
        "alignment",
        # Invalid alignment values and types, respectively
        (-1, 0, 3, 17, 33) + ("1.0", "1", "foo", 1.0, 1.5, 3.2),
    )
    def test_invalid_aligments(self, array_types, shape, dtype, alignment):
        # Use regex pattern to match error message, handling potential ANSI
        # color codes which appear on CI.
        expected_invalid_type_error_regex = (
            r"RequireLiteralValue:.*alignment must be a constant integer"
        )
        which, array_type = array_types
        if which == 0:

            @cuda.jit
            def f(dest_array):
                i = cuda.grid(1)
                local_array = cuda.local.array(
                    shape=shape,
                    dtype=dtype,
                    alignment=alignment,
                )
                if i == 0:
                    dest_array[0] = local_array.ctypes.data
        else:

            @cuda.jit
            def f(dest_array):
                i = cuda.grid(1)
                shared_array = cuda.shared.array(
                    shape=shape,
                    dtype=dtype,
                    alignment=alignment,
                )
                if i == 0:
                    dest_array[0] = shared_array.ctypes.data

        array = np.zeros(1, dtype=np.uint64)

        # The type of error we expect differs between an invalid value
        # that is still an int, and an invalid type.
        if isinstance(alignment, int):
            with pytest.raises(ValueError, match=r"Alignment must be.*"):
                f[1, 1](array)
        else:
            with pytest.raises(
                TypingError, match=expected_invalid_type_error_regex
            ):
                f[1, 1](array)

        if NOISY:
            print(".", end="", flush=True)


@skip_unless_cudasim("Only check for alignment unsupported in the simulator")
class TestCudasimUnsupportedAlignment:
    def test_local_unsupported(self):
        @cuda.jit
        def f():
            cuda.local.array(1, dtype=np.uint8, alignment=16)

        with pytest.raises(RuntimeError, match="not supported in cudasim"):
            f[1, 1]()

    def test_shared_unsupported(self):
        @cuda.jit
        def f():
            cuda.shared.array(1, dtype=np.uint8, alignment=16)

        with pytest.raises(RuntimeError, match="not supported in cudasim"):
            f[1, 1]()
