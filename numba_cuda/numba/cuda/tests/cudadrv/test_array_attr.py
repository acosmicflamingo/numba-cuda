# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

import numpy as np
import pytest

from numba import cuda
from numba.cuda.testing import skip_on_cudasim
from numba.cuda.tests.support import assertPreciseEqual


class TestArrayAttr:
    def test_contigous_2d(self):
        ary = np.arange(10)
        cary = ary.reshape(2, 5)
        fary = np.asfortranarray(cary)

        dcary = cuda.to_device(cary)
        dfary = cuda.to_device(fary)
        assert dcary.is_c_contiguous()
        assert not dfary.is_c_contiguous()
        assert not dcary.is_f_contiguous()
        assert dfary.is_f_contiguous()

    def test_contigous_3d(self):
        ary = np.arange(20)
        cary = ary.reshape(2, 5, 2)
        fary = np.asfortranarray(cary)

        dcary = cuda.to_device(cary)
        dfary = cuda.to_device(fary)
        assert dcary.is_c_contiguous()
        assert not dfary.is_c_contiguous()
        assert not dcary.is_f_contiguous()
        assert dfary.is_f_contiguous()

    def test_contigous_4d(self):
        ary = np.arange(60)
        cary = ary.reshape(2, 5, 2, 3)
        fary = np.asfortranarray(cary)

        dcary = cuda.to_device(cary)
        dfary = cuda.to_device(fary)
        assert dcary.is_c_contiguous()
        assert not dfary.is_c_contiguous()
        assert not dcary.is_f_contiguous()
        assert dfary.is_f_contiguous()

    def test_ravel_1d(self):
        ary = np.arange(60)
        dary = cuda.to_device(ary)
        for order in "CFA":
            expect = ary.ravel(order=order)
            dflat = dary.ravel(order=order)
            flat = dflat.copy_to_host()
            assert dary is not dflat  # ravel returns new array
            assert flat.ndim == 1
            assert expect == pytest.approx(flat)

    @skip_on_cudasim("CUDA Array Interface is not supported in the simulator")
    def test_ravel_stride_1d(self):
        ary = np.arange(60)
        dary = cuda.to_device(ary)
        # No-copy stride device array
        darystride = dary[::2]
        dary_data = dary.__cuda_array_interface__["data"][0]
        ddarystride_data = darystride.__cuda_array_interface__["data"][0]
        assert dary_data == ddarystride_data
        # Fail on ravel on non-contiguous array
        with pytest.raises(NotImplementedError):
            darystride.ravel()

    def test_ravel_c(self):
        ary = np.arange(60)
        reshaped = ary.reshape(2, 5, 2, 3)

        expect = reshaped.ravel(order="C")
        dary = cuda.to_device(reshaped)
        dflat = dary.ravel()
        flat = dflat.copy_to_host()
        assert dary is not dflat
        assert flat.ndim == 1
        assert expect == pytest.approx(flat)

        # explicit order kwarg
        for order in "CA":
            expect = reshaped.ravel(order=order)
            dary = cuda.to_device(reshaped)
            dflat = dary.ravel(order=order)
            flat = dflat.copy_to_host()
            assert dary is not dflat
            assert flat.ndim == 1
            assertPreciseEqual(expect, flat)

    @skip_on_cudasim("CUDA Array Interface is not supported in the simulator")
    def test_ravel_stride_c(self):
        ary = np.arange(60)
        reshaped = ary.reshape(2, 5, 2, 3)

        dary = cuda.to_device(reshaped)
        darystride = dary[::2, ::2, ::2, ::2]
        dary_data = dary.__cuda_array_interface__["data"][0]
        ddarystride_data = darystride.__cuda_array_interface__["data"][0]
        assert dary_data == ddarystride_data
        with pytest.raises(NotImplementedError):
            darystride.ravel()

    def test_ravel_f(self):
        ary = np.arange(60)
        reshaped = np.asfortranarray(ary.reshape(2, 5, 2, 3))
        for order in "FA":
            expect = reshaped.ravel(order=order)
            dary = cuda.to_device(reshaped)
            dflat = dary.ravel(order=order)
            flat = dflat.copy_to_host()
            assert dary is not dflat
            assert flat.ndim == 1
            assert expect == pytest.approx(flat)

    @skip_on_cudasim("CUDA Array Interface is not supported in the simulator")
    def test_ravel_stride_f(self):
        ary = np.arange(60)
        reshaped = np.asfortranarray(ary.reshape(2, 5, 2, 3))
        dary = cuda.to_device(reshaped)
        darystride = dary[::2, ::2, ::2, ::2]
        dary_data = dary.__cuda_array_interface__["data"][0]
        ddarystride_data = darystride.__cuda_array_interface__["data"][0]
        assert dary_data == ddarystride_data
        with pytest.raises(NotImplementedError):
            darystride.ravel()

    def test_reshape_c(self):
        ary = np.arange(10)
        expect = ary.reshape(2, 5)
        dary = cuda.to_device(ary)
        dary_reshaped = dary.reshape(2, 5)
        got = dary_reshaped.copy_to_host()
        assert expect == pytest.approx(got)

    def test_reshape_f(self):
        ary = np.arange(10)
        expect = ary.reshape(2, 5, order="F")
        dary = cuda.to_device(ary)
        dary_reshaped = dary.reshape(2, 5, order="F")
        got = dary_reshaped.copy_to_host()
        assert expect == pytest.approx(got)
