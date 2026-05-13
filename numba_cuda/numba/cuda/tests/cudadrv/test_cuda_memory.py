# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

import ctypes

import numpy as np
import pytest

from numba.cuda.cudadrv import driver, devices
from numba.cuda.testing import skip_on_cudasim
from numba.cuda.tests.support import cuda_test_setup


@skip_on_cudasim("CUDA Memory API unsupported in the simulator")
class TestCudaMemory:
    @pytest.fixture(autouse=True)
    def configure(self, cuda_test_setup):
        self.context = devices.get_context()

        yield

        self.context.reset()
        del self.context

    def _template(self, obj):
        assert driver.is_device_memory(obj)
        driver.require_device_memory(obj)
        expected_class = ctypes.c_size_t
        assert isinstance(obj.device_ctypes_pointer, expected_class)

    def test_device_memory(self):
        devmem = self.context.memalloc(1024)
        self._template(devmem)

    def test_device_view(self):
        devmem = self.context.memalloc(1024)
        self._template(devmem.view(10))

    def test_host_alloc(self):
        devmem = self.context.memhostalloc(1024, mapped=True)
        self._template(devmem)

    def test_pinned_memory(self):
        ary = np.arange(10)
        devmem = self.context.mempin(
            ary, ary.ctypes.data, ary.size * ary.dtype.itemsize, mapped=True
        )
        self._template(devmem)

    def test_managed_memory(self):
        devmem = self.context.memallocmanaged(1024)
        self._template(devmem)

    def test_derived_pointer(self):
        # Use MemoryPointer.view to create derived pointer

        def handle_val(mem):
            return int(mem.handle)

        def check(m, offset):
            # create view
            v1 = m.view(offset)
            assert handle_val(v1.owner) == handle_val(m)
            assert m.refct == 2
            assert handle_val(v1) - offset == handle_val(v1.owner)
            # create a view
            v2 = v1.view(offset)
            assert handle_val(v2.owner) == handle_val(m)
            assert handle_val(v2.owner) == handle_val(m)
            assert handle_val(v2) - offset * 2 == handle_val(v2.owner)
            assert m.refct == 3
            del v2
            assert m.refct == 2
            del v1
            assert m.refct == 1

        m = self.context.memalloc(1024)
        check(m=m, offset=0)
        check(m=m, offset=1)

    def test_user_extension(self):
        # User can use MemoryPointer to wrap externally defined pointers.
        # This test checks if the finalizer is invokded at correct time
        fake_ptr = ctypes.c_void_p(0xDEADBEEF)
        dtor_invoked = [0]

        def dtor():
            dtor_invoked[0] += 1

        # Ensure finalizer is called when pointer is deleted
        ptr = driver.MemoryPointer(
            context=self.context, pointer=fake_ptr, size=40, finalizer=dtor
        )
        assert dtor_invoked[0] == 0
        del ptr
        assert dtor_invoked[0] == 1

        # Ensure removing derived pointer doesn't call finalizer
        ptr = driver.MemoryPointer(
            context=self.context, pointer=fake_ptr, size=40, finalizer=dtor
        )
        owned = ptr.own()
        del owned
        assert dtor_invoked[0] == 1
        del ptr
        assert dtor_invoked[0] == 2


class TestCudaMemoryFunctions:
    @pytest.fixture(autouse=True)
    def configure(self, cuda_test_setup):
        self.context = devices.get_context()

        yield

        self.context.reset()
        del self.context

    def test_memcpy(self):
        hstary = np.arange(100, dtype=np.uint32)
        hstary2 = np.arange(100, dtype=np.uint32)
        sz = hstary.size * hstary.dtype.itemsize
        devary = self.context.memalloc(sz)

        driver.host_to_device(devary, hstary, sz)
        driver.device_to_host(hstary2, devary, sz)

        assert np.all(hstary == hstary2)

    def test_memset(self):
        dtype = np.dtype("uint32")
        n = 10
        sz = dtype.itemsize * 10
        devary = self.context.memalloc(sz)
        driver.device_memset(devary, 0xAB, sz)

        hstary = np.empty(n, dtype=dtype)
        driver.device_to_host(hstary, devary, sz)

        hstary2 = np.array([0xABABABAB] * n, dtype=np.dtype("uint32"))
        assert np.all(hstary == hstary2)

    def test_d2d(self):
        hst = np.arange(100, dtype=np.uint32)
        hst2 = np.empty_like(hst)
        sz = hst.size * hst.dtype.itemsize
        dev1 = self.context.memalloc(sz)
        dev2 = self.context.memalloc(sz)
        driver.host_to_device(dev1, hst, sz)
        driver.device_to_device(dev2, dev1, sz)
        driver.device_to_host(hst2, dev2, sz)
        assert np.all(hst == hst2)


@skip_on_cudasim("CUDA Memory API unsupported in the simulator")
class TestMVExtent:
    def test_c_contiguous_array(self):
        ary = np.arange(100)
        arysz = ary.dtype.itemsize * ary.size
        s, e = driver.host_memory_extents(ary)
        assert ary.ctypes.data == s
        assert arysz == driver.host_memory_size(ary)

    def test_f_contiguous_array(self):
        ary = np.asfortranarray(np.arange(100).reshape(2, 50))
        arysz = ary.dtype.itemsize * np.prod(ary.shape)
        s, e = driver.host_memory_extents(ary)
        assert ary.ctypes.data == s
        assert arysz == driver.host_memory_size(ary)

    def test_single_element_array(self):
        ary = np.asarray(np.uint32(1234))
        arysz = ary.dtype.itemsize
        s, e = driver.host_memory_extents(ary)
        assert ary.ctypes.data == s
        assert arysz == driver.host_memory_size(ary)

    def test_ctypes_struct(self):
        class mystruct(ctypes.Structure):
            _fields_ = [("x", ctypes.c_int), ("y", ctypes.c_int)]

        data = mystruct(x=123, y=432)
        sz = driver.host_memory_size(data)
        assert ctypes.sizeof(data) == sz

    def test_ctypes_double(self):
        data = ctypes.c_double(1.234)
        sz = driver.host_memory_size(data)
        assert ctypes.sizeof(data) == sz
