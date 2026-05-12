# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

import numbers

import pytest

from numba import cuda
from numba.cuda.testing import skip_on_cudasim
from numba.cuda.cudadrv import driver
from numba.cuda.tests.support import cuda_test_setup


class TestContextStack:
    @pytest.fixture(autouse=True)
    def context_stack_setup(cuda_test_setup):
        cuda.current_context().reset()

    def test_gpus_len(self):
        assert len(cuda.gpus) > 0

    def test_gpus_iter(self):
        gpulist = list(cuda.gpus)
        assert len(gpulist) > 0

    def test_gpus_cudevice_indexing(self):
        """Test that CUdevice objects can be used to index into cuda.gpus"""
        # When using the CUDA Python bindings, the device ids are CUdevice
        # objects, otherwise they are integers. We test that the device id is
        # usable as an index into cuda.gpus.
        device_ids = [device.id for device in cuda.list_devices()]
        for device_id in device_ids:
            with cuda.gpus[device_id]:
                assert cuda.gpus.current.id == device_id


class TestContextAPI:
    @pytest.fixture(autouse=True)
    def context_stack_setup(cuda_test_setup):
        cuda.current_context().reset()

    def test_context_memory(self):
        try:
            mem = cuda.current_context().get_memory_info()
        except NotImplementedError:
            pytest.skip(
                reason="EMM Plugin does not implement get_memory_info()"
            )

        assert isinstance(mem.free, numbers.Number)
        assert mem.free == mem[0]

        assert isinstance(mem.total, numbers.Number)
        assert mem.total == mem[1]

        assert mem.free <= mem.total

    @pytest.mark.skipif(len(cuda.gpus) < 2, reason="need more than 1 gpus")
    @skip_on_cudasim("CUDA HW required")
    def test_forbidden_context_switch(self):
        # Cannot switch context inside a `cuda.require_context`
        @cuda.require_context
        def switch_gpu():
            with cuda.gpus[1]:
                pass

        with cuda.gpus[0]:
            with pytest.raises(RuntimeError) as raises:
                switch_gpu()

            assert "Cannot switch CUDA-context." in str(raises.value)

    @pytest.mark.skipif(len(cuda.gpus) < 2, reason="need more than 1 gpus")
    def test_accepted_context_switch(self):
        def switch_gpu():
            with cuda.gpus[1]:
                return cuda.current_context().device.id

        with cuda.gpus[0]:
            devid = switch_gpu()
        assert int(devid) == 1


@skip_on_cudasim("CUDA HW required")
class TestContextLeak:
    """Regression tests for context leaks from the gpu context manager."""

    def test_gpus_context_manager_does_not_leak(self):
        # Regression test: ``with cuda.gpus[N]`` must not leave a CUDA
        # context on the thread after the block exits.
        the_driver = driver.driver

        # Drain any pre-existing contexts from the stack.
        while the_driver.pop_active_context() is not None:
            pass

        with cuda.gpus[0]:
            pass

        # After exiting the context manager the current context must be null.
        with the_driver.get_active_context() as ac:
            assert ac.context_handle is None, (
                "CUDA context leaked after exiting cuda.gpus context manager"
            )

    def test_gpus_context_manager_restores_previous_context(self):
        # If a context is already active before entering the context manager,
        # it must be restored on exit.
        the_driver = driver.driver

        # Ensure device-0 context exists and is pushed.
        outer_ctx = cuda.current_context()
        outer_handle = int(outer_ctx.handle)

        with cuda.gpus[0]:
            pass

        with the_driver.get_active_context() as ac:
            assert ac.context_handle is not None
            assert int(ac.context_handle) == outer_handle, (
                "Previous context was not restored after exiting "
                "cuda.gpus context manager",
            )


@skip_on_cudasim("CUDA HW required")
class Test3rdPartyContext:
    @pytest.fixture(autouse=True)
    def setUp(cuda_test_setup):
        cuda.current_context().reset()

    def test_attached_primary(self, extra_work=lambda: None):
        # Emulate primary context creation by 3rd party
        the_driver = driver.driver
        dev = driver.binding.CUdevice(0)
        hctx = the_driver.cuDevicePrimaryCtxRetain(dev)
        ctx = driver.Context(dev, hctx)
        try:
            ctx.push()
            # Check that the context from numba matches the created primary
            # context.
            my_ctx = cuda.current_context()
            assert int(my_ctx.handle) == int(ctx.handle)

            extra_work()
        finally:
            ctx.pop()
            the_driver.cuDevicePrimaryCtxRelease(dev)

    def test_attached_non_primary(self):
        # Emulate non-primary context creation by 3rd party
        the_driver = driver.driver
        flags = 0
        dev = driver.binding.CUdevice(0)

        result, version = driver.binding.cuDriverGetVersion()
        assert result == driver.binding.CUresult.CUDA_SUCCESS, (
            "Error getting CUDA driver version"
        )

        # CUDA 13's cuCtxCreate has an optional parameter prepended. The
        # version of cuCtxCreate in use depends on the cuda.bindings major
        # version rather than the installed driver version on the machine
        # we're running on.
        from cuda import bindings

        bindings_version = int(bindings.__version__.split(".")[0])
        if bindings_version in (11, 12):
            args = (flags, dev)
        else:
            args = (None, flags, dev)

        hctx = the_driver.cuCtxCreate(*args)
        try:
            cuda.current_context()
        except RuntimeError as e:
            # Expecting an error about non-primary CUDA context
            assert "Numba cannot operate on non-primary CUDA context " in str(e)
        else:
            pytest.fail("No RuntimeError raised")
        finally:
            the_driver.cuCtxDestroy(hctx)

    def test_cudajit_in_attached_primary_context(self):
        def do():
            from numba import cuda

            @cuda.jit
            def foo(a):
                for i in range(a.size):
                    a[i] = i

            a = cuda.device_array(10)
            foo[1, 1](a)
            assert list(a.copy_to_host()) == list(range(10))

        self.test_attached_primary(do)
