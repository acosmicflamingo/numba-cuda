# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

import numpy as np
import pytest

from numba import cuda
from numba.cuda.np.numpy_support import from_dtype
from numba.cuda.testing import skip_on_cudasim


class TestAlignment:
    def test_record_alignment(self):
        rec_dtype = np.dtype([("a", "int32"), ("b", "float64")], align=True)
        rec = from_dtype(rec_dtype)

        @cuda.jit((rec[:],))
        def foo(a):
            i = cuda.grid(1)
            a[i].a = a[i].b

        a_recarray = np.recarray(3, dtype=rec_dtype)
        for i in range(a_recarray.size):
            a_rec = a_recarray[i]
            a_rec.a = 0
            a_rec.b = (i + 1) * 123

        foo[1, 3](a_recarray)

        assert np.all(a_recarray.a == a_recarray.b)

    @skip_on_cudasim("Simulator does not check alignment")
    def test_record_alignment_error(self):
        rec_dtype = np.dtype([("a", "int32"), ("b", "float64")])
        rec = from_dtype(rec_dtype)

        with pytest.raises(Exception) as raises:

            @cuda.jit((rec[:],))
            def foo(a):
                i = cuda.grid(1)
                a[i].a = a[i].b

        assert "type float64 is not aligned" in str(raises.value)
