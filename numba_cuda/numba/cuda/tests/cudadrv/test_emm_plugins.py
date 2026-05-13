# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

import ctypes
import weakref

import numpy as np
import pytest

from numba import cuda
from numba.cuda.core import config
from numba.cuda.testing import skip_on_cudasim
from numba.cuda.tests.support import linux_only, cuda_test_setup

if not config.ENABLE_CUDASIM:

    class DeviceOnlyEMMPlugin(cuda.HostOnlyCUDAMemoryManager):
        """
        Dummy EMM Plugin implementation for testing. It memorises which plugin
        API methods have been called so that the tests can check that Numba
        called into the plugin as expected.
        """

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

            # For tracking our dummy allocations
            self.allocations = {}
            self.count = 0

            # For tracking which methods have been called
            self.initialized = False
            self.memalloc_called = False
            self.reset_called = False
            self.get_memory_info_called = False
            self.get_ipc_handle_called = False

        def memalloc(self, size):
            # We maintain a list of allocations and keep track of them, so that
            # we can test that the finalizers of objects returned by memalloc
            # get called.

            # Numba should have initialized the memory manager when preparing
            # the context for use, prior to any memalloc call.
            if not self.initialized:
                raise RuntimeError("memalloc called before initialize")
            self.memalloc_called = True

            # Create an allocation and record it
            self.count += 1
            alloc_count = self.count
            self.allocations[alloc_count] = size

            # The finalizer deletes the record from our internal dict of
            # allocations.
            finalizer_allocs = self.allocations

            def finalizer():
                del finalizer_allocs[alloc_count]

            # We use an AutoFreePointer so that the finalizer will be run when
            # the reference count drops to zero.
            ctx = weakref.proxy(self.context)
            ptr = ctypes.c_void_p(alloc_count)
            return cuda.cudadrv.driver.AutoFreePointer(
                ctx, ptr, size, finalizer=finalizer
            )

        def initialize(self):
            # No special initialization needed.
            self.initialized = True

        def reset(self):
            # We remove all allocations on reset, just as a real EMM Plugin
            # would do. Note that our finalizers in memalloc don't check
            # whether the allocations are still alive, so running them after
            # reset will detect any allocations that are floating around at
            # exit time; however, the atexit finalizer for weakref will only
            # print a traceback, not terminate the interpreter abnormally.
            self.reset_called = True

        def get_memory_info(self):
            # Return some dummy memory information
            self.get_memory_info_called = True
            return cuda.MemoryInfo(free=32, total=64)

        def get_ipc_handle(self, memory):
            # The dummy IPC handle is only a string, so it is important that
            # the tests don't try to do too much with it (e.g. open / close
            # it).
            self.get_ipc_handle_called = True
            return "Dummy IPC handle for alloc %s" % memory.device_pointer_value

        @property
        def interface_version(self):
            # The expected version for an EMM Plugin.
            return 1

    class BadVersionEMMPlugin(DeviceOnlyEMMPlugin):
        """A plugin that claims to implement a different interface version"""

        @property
        def interface_version(self):
            return 2


@skip_on_cudasim("EMM Plugins not supported on CUDA simulator")
class TestDeviceOnlyEMMPlugin:
    """
    Tests that the API of an EMM Plugin that implements device allocations
    only is used correctly by Numba.
    """

    @pytest.fixture(autouse=True)
    def configure(self, cuda_test_setup):
        # Always start afresh with a new context and memory manager
        ctx = cuda.current_context()
        ctx.reset()
        self._initial_memory_manager = ctx.memory_manager
        ctx.memory_manager = DeviceOnlyEMMPlugin(context=ctx)

        yield

        ctx = cuda.current_context()
        ctx.reset()
        ctx.memory_manager = self._initial_memory_manager

    def test_memalloc(self):
        mgr = cuda.current_context().memory_manager

        # Allocate an array and check that memalloc was called with the correct
        # size.
        arr_1 = np.arange(10)
        d_arr_1 = cuda.device_array_like(arr_1)
        assert mgr.memalloc_called

        assert mgr.count == 1
        assert mgr.allocations[1] == arr_1.nbytes

        # Allocate again, with a different size, and check that it is also
        # correct.
        arr_2 = np.arange(5)
        d_arr_2 = cuda.device_array_like(arr_2)
        assert mgr.count == 2
        assert mgr.allocations[2] == arr_2.nbytes

        # Remove the first array, and check that our finalizer was called for
        # the first array only.
        del d_arr_1
        assert 1 not in mgr.allocations
        assert 2 in mgr.allocations

        # Remove the second array and check that its finalizer was also
        # called.
        del d_arr_2
        assert 2 not in mgr.allocations

    def test_initialized_in_context(self):
        # If we have a CUDA context, it should already have initialized its
        # memory manager.
        assert cuda.current_context().memory_manager.initialized

    def test_reset(self):
        ctx = cuda.current_context()
        ctx.reset()
        assert ctx.memory_manager.reset_called

    def test_get_memory_info(self):
        ctx = cuda.current_context()
        meminfo = ctx.get_memory_info()
        assert ctx.memory_manager.get_memory_info_called
        assert meminfo.free == 32
        assert meminfo.total == 64

    @linux_only
    def test_get_ipc_handle(self):
        # We don't attempt to close the IPC handle in this test because Numba
        # will be expecting a real IpcHandle object to have been returned from
        # get_ipc_handle, and it would cause problems to do so.
        arr = np.arange(2)
        d_arr = cuda.device_array_like(arr)
        ipch = d_arr.get_ipc_handle()
        ctx = cuda.current_context()
        assert ctx.memory_manager.get_ipc_handle_called
        assert "Dummy IPC handle for alloc 1" in ipch._ipc_handle


@skip_on_cudasim("EMM Plugins not supported on CUDA simulator")
class TestBadEMMPluginVersion:
    """
    Ensure that Numba rejects EMM Plugins with incompatible version
    numbers.
    """

    def test_bad_plugin_version(self):
        with pytest.raises(RuntimeError) as raises:
            cuda.set_memory_manager(BadVersionEMMPlugin)
        assert "version 1 required" in str(raises.value)
