#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cmath>
#include <iostream>
#include <stdexcept>

namespace needle
{
  namespace cpu
  {

#define ALIGNMENT 256
#define TILE 8
    typedef float scalar_t;
    const size_t ELEM_SIZE = sizeof(scalar_t);

    /**
     * This is a utility structure for maintaining an array aligned to ALIGNMENT boundaries in
     * memory.  This alignment should be at least TILE * ELEM_SIZE, though we make it even larger
     * here by default.
     */
    struct AlignedArray
    {
      AlignedArray(const size_t size)
      {
        int ret = posix_memalign((void **)&ptr, ALIGNMENT, size * ELEM_SIZE);
        if (ret != 0)
          throw std::bad_alloc();
        this->size = size;
      }
      ~AlignedArray() { free(ptr); }
      size_t ptr_as_int() { return (size_t)ptr; }
      scalar_t *ptr;
      size_t size;
    };

    void Fill(AlignedArray *out, scalar_t val)
    {
      /**
       * Fill the values of an aligned array with val
       */
      for (int i = 0; i < out->size; i++)
      {
        out->ptr[i] = val;
      }
    }

    void Compact(const AlignedArray &a, AlignedArray *out, std::vector<uint32_t> shape,
                 std::vector<uint32_t> strides, size_t offset)
    {
      /**
       * Compact an array in memory
       *
       * Args:
       *   a: non-compact representation of the array, given as input
       *   out: compact version of the array to be written
       *   shape: shapes of each dimension for a and out
       *   strides: strides of the *a* array (not out, which has compact strides)
       *   offset: offset of the *a* array (not out, which has zero offset, being compact)
       *
       * Returns:
       *  void (you need to modify out directly, rather than returning anything; this is true for all the
       *  function will implement here, so we won't repeat this note.)
       */
      /// BEGIN YOUR SOLUTION
      std::vector<uint32_t> idx(shape.size(), 0);
      int total_elem = 1;
      for (int i = 0; i < shape.size(); i++)
      {
        total_elem *= shape[i];
      }
      for (int cnt = 0; cnt < total_elem; cnt++)
      {
        int cur_idx = offset;
        for (auto i = 0; i < strides.size(); i++)
        {
          cur_idx += strides[i] * idx[i];
        }
        out->ptr[cnt] = a.ptr[cur_idx];
        for (auto i = shape.size() - 1; i >= 0; i--)
        {
          if (idx[i] == shape[i] - 1)
          {
            if (i == 0)
              return;
            idx[i] = 0;
          }
          else
          {
            idx[i] += 1;
            break;
          }
        }
      }

      /// END YOUR SOLUTION
    }

    void EwiseSetitem(const AlignedArray &a, AlignedArray *out, std::vector<uint32_t> shape,
                      std::vector<uint32_t> strides, size_t offset)
    {
      /**
       * Set items in a (non-compact) array
       *
       * Args:
       *   a: _compact_ array whose items will be written to out
       *   out: non-compact array whose items are to be written
       *   shape: shapes of each dimension for a and out
       *   strides: strides of the *out* array (not a, which has compact strides)
       *   offset: offset of the *out* array (not a, which has zero offset, being compact)
       */
      /// BEGIN YOUR SOLUTION
      std::vector<uint32_t> idx(shape.size(), 0);
      int total_elem = 1;
      for (int i = 0; i < shape.size(); i++)
      {
        total_elem *= shape[i];
      }
      for (int cnt = 0; cnt < total_elem; cnt++)
      {
        int cur_idx = offset;
        for (auto i = 0; i < strides.size(); i++)
        {
          cur_idx += strides[i] * idx[i];
        }
        if (cur_idx < out->size)
          out->ptr[cur_idx] = a.ptr[cnt];
        for (auto i = shape.size() - 1; i >= 0; i--)
        {
          if (idx[i] == shape[i] - 1)
          {
            if (i == 0)
              return;
            idx[i] = 0;
          }
          else
          {
            idx[i] += 1;
            break;
          }
        }
      }
      /// END YOUR SOLUTION
    }

    void ScalarSetitem(const size_t size, scalar_t val, AlignedArray *out, std::vector<uint32_t> shape,
                       std::vector<uint32_t> strides, size_t offset)
    {
      /**
       * Set items is a (non-compact) array
       *
       * Args:
       *   size: number of elements to write in out array (note that this will note be the same as
       *         out.size, because out is a non-compact subset array);  it _will_ be the same as the
       *         product of items in shape, but convenient to just pass it here.
       *   val: scalar value to write to
       *   out: non-compact array whose items are to be written
       *   shape: shapes of each dimension of out
       *   strides: strides of the out array
       *   offset: offset of the out array
       */

      /// BEGIN YOUR SOLUTION
      std::vector<uint32_t> idx(shape.size(), 0);
      int cnt = 0;
      int cur_idx;
      while (1)
      {
        cur_idx = 0;
        for (auto i = 0; i < strides.size(); i++)
        {
          cur_idx += strides[i] * idx[i];
        }
        out->ptr[cur_idx + offset] = val;
        for (auto i = shape.size() - 1; i >= 0; i--)
        {
          if (idx[i] == shape[i] - 1)
          {
            if (i == 0)
              return;
            idx[i] = 0;
          }
          else
          {
            idx[i] += 1;
            break;
          }
        }
        cnt++;
      }
      /// END YOUR SOLUTION
    }

    void EwiseAdd(const AlignedArray &a, const AlignedArray &b, AlignedArray *out)
    {
      /**
       * Set entries in out to be the sum of correspondings entires in a and b.
       */
      for (size_t i = 0; i < a.size; i++)
      {
        out->ptr[i] = a.ptr[i] + b.ptr[i];
      }
    }

    void ScalarAdd(const AlignedArray &a, scalar_t val, AlignedArray *out)
    {
      /**
       * Set entries in out to be the sum of corresponding entry in a plus the scalar val.
       */
      for (size_t i = 0; i < a.size; i++)
      {
        out->ptr[i] = a.ptr[i] + val;
      }
    }

    /**
     * In the code the follows, use the above template to create analogous element-wise
     * and and scalar operators for the following functions.  See the numpy backend for
     * examples of how they should work.
     *   - EwiseMul, ScalarMul
     *   - EwiseDiv, ScalarDiv
     *   - ScalarPower
     *   - EwiseMaximum, ScalarMaximum
     *   - EwiseEq, ScalarEq
     *   - EwiseGe, ScalarGe
     *   - EwiseLog
     *   - EwiseExp
     *   - EwiseTanh
     *
     * If you implement all these naively, there will be a lot of repeated code, so
     * you are welcome (but not required), to use macros or templates to define these
     * functions (however you want to do so, as long as the functions match the proper)
     * signatures above.
     */

    /// BEGIN YOUR SOLUTION
    enum class _EwiseOp
    {
      MUL,
      DIV,
      MAX,
      EQ,
      GE,
      LOG,
      EXP,
      TANH
    };

    enum class _ScalarOp
    {
      MUL,
      DIV,
      MAX,
      POWER,
      EQ,
      GE
    };

    template <_EwiseOp op>
    void EwiseOp(const AlignedArray &a, const AlignedArray &b, AlignedArray *out)
    {
      for (int idx = 0; idx < a.size; idx++)
      {
        if constexpr (op == _EwiseOp::MUL)
        {
          out->ptr[idx] = a.ptr[idx] * b.ptr[idx];
        }
        if constexpr (op == _EwiseOp::DIV)
        {
          out->ptr[idx] = a.ptr[idx] / b.ptr[idx];
        }
        if constexpr (op == _EwiseOp::MAX)
        {
          out->ptr[idx] = std::max(a.ptr[idx], b.ptr[idx]);
        }
        if constexpr (op == _EwiseOp::EQ)
        {
          out->ptr[idx] = (scalar_t)(a.ptr[idx] == b.ptr[idx]);
        }
        if constexpr (op == _EwiseOp::GE)
        {
          out->ptr[idx] = (scalar_t)(a.ptr[idx] >= b.ptr[idx]);
        }
      }
    }
    template <_EwiseOp op>
    void EwiseFunc(const AlignedArray &a, AlignedArray *out)
    {
      for (int idx = 0; idx < a.size; idx++)
      {
        if constexpr (op == _EwiseOp::LOG)
        {
          out->ptr[idx] = std::log(a.ptr[idx]);
        }
        if constexpr (op == _EwiseOp::EXP)
        {
          out->ptr[idx] = std::exp(a.ptr[idx]);
        }
        if constexpr (op == _EwiseOp::TANH)
        {
          out->ptr[idx] = std::tanh(a.ptr[idx]);
        }
      }
    }

    template <_ScalarOp op>
    void ScalarOp(const AlignedArray &a, scalar_t val, AlignedArray *out)
    {
      for (int idx = 0; idx < a.size; idx++)
      {
        if constexpr (op == _ScalarOp::MUL)
        {
          out->ptr[idx] = a.ptr[idx] * val;
        }
        if constexpr (op == _ScalarOp::DIV)
        {
          out->ptr[idx] = a.ptr[idx] / val;
        }
        if constexpr (op == _ScalarOp::MAX)
        {
          out->ptr[idx] = std::max(a.ptr[idx], val);
        }
        if constexpr (op == _ScalarOp::EQ)
        {
          out->ptr[idx] = (scalar_t)(a.ptr[idx] == val);
        }
        if constexpr (op == _ScalarOp::GE)
        {
          out->ptr[idx] = (scalar_t)(a.ptr[idx] >= val);
        }
        if constexpr (op == _ScalarOp::POWER)
        {
          out->ptr[idx] = std::pow(a.ptr[idx], val);
        }
      }
    }
    /// END YOUR SOLUTION

    void Matmul(const AlignedArray &a, const AlignedArray &b, AlignedArray *out, uint32_t m, uint32_t n,
                uint32_t p)
    {
      /**
       * Multiply two (compact) matrices into an output (also compact) matrix.  For this implementation
       * you can use the "naive" three-loop algorithm.
       *
       * Args:
       *   a: compact 2D array of size m x n
       *   b: compact 2D array of size n x p
       *   out: compact 2D array of size m x p to write the output to
       *   m: rows of a / out
       *   n: columns of a / rows of b
       *   p: columns of b / out
       */

      /// BEGIN YOUR SOLUTION
      for (uint32_t i = 0; i < m; i++)
      {
        for (uint32_t j = 0; j < p; j++)
        {
          scalar_t val = 0;
          for (uint32_t k = 0; k < n; k++)
          {
            val += a.ptr[i * n + k] * b.ptr[k * p + j];
          }
          out->ptr[i * p + j] = val;
        }
      }
      /// END YOUR SOLUTION
    }

    inline void AlignedDot(const float *__restrict__ a,
                           const float *__restrict__ b,
                           float *__restrict__ out)
    {

      /**
       * Multiply together two TILE x TILE matrices, and _add _the result to out (it is important to add
       * the result to the existing out, which you should not set to zero beforehand).  We are including
       * the compiler flags here that enable the compile to properly use vector operators to implement
       * this function.  Specifically, the __restrict__ keyword indicates to the compile that a, b, and
       * out don't have any overlapping memory (which is necessary in order for vector operations to be
       * equivalent to their non-vectorized counterparts (imagine what could happen otherwise if a, b,
       * and out had overlapping memory).  Similarly the __builtin_assume_aligned keyword tells the
       * compiler that the input array will be aligned to the appropriate blocks in memory, which also
       * helps the compiler vectorize the code.
       *
       * Args:
       *   a: compact 2D array of size TILE x TILE
       *   b: compact 2D array of size TILE x TILE
       *   out: compact 2D array of size TILE x TILE to write to
       */

      a = (const float *)__builtin_assume_aligned(a, TILE * ELEM_SIZE);
      b = (const float *)__builtin_assume_aligned(b, TILE * ELEM_SIZE);
      out = (float *)__builtin_assume_aligned(out, TILE * ELEM_SIZE);

      /// BEGIN YOUR SOLUTION
      for (uint32_t i = 0; i < TILE; i++)
      {
        for (uint32_t j = 0; j < TILE; j++)
        {
          scalar_t val = out[i * TILE + j];
          for (uint32_t k = 0; k < TILE; k++)
          {
            val += a[i * TILE + k] * b[k * TILE + j];
          }
          out[i * TILE + j] = val;
        }
      }
      /// END YOUR SOLUTION
    }

    void MatmulTiled(const AlignedArray &a, const AlignedArray &b, AlignedArray *out, uint32_t m,
                     uint32_t n, uint32_t p)
    {
      /**
       * Matrix multiplication on tiled representations of array.  In this setting, a, b, and out
       * are all *4D* compact arrays of the appropriate size, e.g. a is an array of size
       *   a[m/TILE][n/TILE][TILE][TILE]
       * You should do the multiplication tile-by-tile to improve performance of the array (i.e., this
       * function should call `AlignedDot()` implemented above).
       *
       * Note that this function will only be called when m, n, p are all multiples of TILE, so you can
       * assume that this division happens without any remainder.
       *
       * Args:
       *   a: compact 4D array of size m/TILE x n/TILE x TILE x TILE
       *   b: compact 4D array of size n/TILE x p/TILE x TILE x TILE
       *   out: compact 4D array of size m/TILE x p/TILE x TILE x TILE to write to
       *   m: rows of a / out
       *   n: columns of a / rows of b
       *   p: columns of b / out
       *
       */
      /// BEGIN YOUR SOLUTION
      const size_t TILE2 = TILE * TILE;
      float *A = new float[TILE * n];
      float *B = new float[TILE * n];
      float *C = new float[TILE2];
      for (int i = 0; i < m / TILE; ++i)
      {
        for (int ii = 0; ii < n / TILE; ++ii)
        {
          for (int num = 0; num < TILE2; ++num)
          {
            A[ii * TILE2 + num] = a.ptr[i * n * TILE + ii * TILE2 + num];
          }
        }
        for (int j = 0; j < p / TILE; ++j)
        {
          for (int ii = 0; ii < n / TILE; ++ii)
          {
            for (int num = 0; num < TILE2; ++num)
            {
              B[ii * TILE2 + num] = b.ptr[ii * TILE * p + j * TILE2 + num];
            }
          }
          for (int k = 0; k < TILE2; ++k)
          {
            C[k] = 0.0;
          }
          for (int k = 0; k < n / TILE; ++k)
          {
            AlignedDot(&A[k * TILE2], &B[k * TILE2], C);
          }
          for (int k = 0; k < TILE2; ++k)
          {
            out->ptr[i * p * TILE + j * TILE2 + k] = C[k];
          }
        }
      }
      delete[] A;
      A = nullptr;
      delete[] B;
      B = nullptr;
      delete[] C;
      C = nullptr;
      /// END YOUR SOLUTION
    }

    void ReduceMax(const AlignedArray &a, AlignedArray *out, size_t reduce_size)
    {
      /**
       * Reduce by taking maximum over `reduce_size` contiguous blocks.
       *
       * Args:
       *   a: compact array of size a.size = out.size * reduce_size to reduce over
       *   out: compact array to write into
       *   reduce_size: size of the dimension to reduce over
       */

      /// BEGIN YOUR SOLUTION
      for (int i = 0; i < out->size; i++)
      {
        scalar_t max_n = a.ptr[i * reduce_size];
        for (int idx = 1; idx < reduce_size; idx++)
        {
          max_n = std::max(a.ptr[i * reduce_size + idx], max_n);
        }
        out->ptr[i] = max_n;
      }
      /// END YOUR SOLUTION
    }

    void ReduceSum(const AlignedArray &a, AlignedArray *out, size_t reduce_size)
    {
      /**
       * Reduce by taking sum over `reduce_size` contiguous blocks.
       *
       * Args:
       *   a: compact array of size a.size = out.size * reduce_size to reduce over
       *   out: compact array to write into
       *   reduce_size: size of the dimension to reduce over
       */

      /// BEGIN YOUR SOLUTION
      for (int i = 0; i < out->size; i++)
      {
        scalar_t sum_n = 0;
        for (int idx = 0; idx < reduce_size; idx++)
        {
          sum_n += a.ptr[i * reduce_size + idx];
        }
        out->ptr[i] = sum_n;
      }
      /// END YOUR SOLUTION
    }

  } // namespace cpu
} // namespace needle

PYBIND11_MODULE(ndarray_backend_cpu, m)
{
  namespace py = pybind11;
  using namespace needle;
  using namespace cpu;

  m.attr("__device_name__") = "cpu";
  m.attr("__tile_size__") = TILE;

  py::class_<AlignedArray>(m, "Array")
      .def(py::init<size_t>(), py::return_value_policy::take_ownership)
      .def("ptr", &AlignedArray::ptr_as_int)
      .def_readonly("size", &AlignedArray::size);

  // return numpy array (with copying for simplicity, otherwise garbage
  // collection is a pain)
  m.def("to_numpy", [](const AlignedArray &a, std::vector<size_t> shape,
                       std::vector<size_t> strides, size_t offset)
        {
    std::vector<size_t> numpy_strides = strides;
    std::transform(numpy_strides.begin(), numpy_strides.end(), numpy_strides.begin(),
                   [](size_t& c) { return c * ELEM_SIZE; });
    return py::array_t<scalar_t>(shape, numpy_strides, a.ptr + offset); });

  // convert from numpy (with copying)
  m.def("from_numpy", [](py::array_t<scalar_t> a, AlignedArray *out)
        { std::memcpy(out->ptr, a.request().ptr, out->size * ELEM_SIZE); });

  m.def("fill", Fill);
  m.def("compact", Compact);
  m.def("ewise_setitem", EwiseSetitem);
  m.def("scalar_setitem", ScalarSetitem);
  m.def("ewise_add", EwiseAdd);
  m.def("scalar_add", ScalarAdd);

  m.def("ewise_mul", EwiseOp<_EwiseOp::MUL>);
  m.def("scalar_mul", ScalarOp<_ScalarOp::MUL>);
  m.def("ewise_div", EwiseOp<_EwiseOp::DIV>);
  m.def("scalar_div", ScalarOp<_ScalarOp::DIV>);
  m.def("scalar_power", ScalarOp<_ScalarOp::POWER>);

  m.def("ewise_maximum", EwiseOp<_EwiseOp::MAX>);
  m.def("scalar_maximum", ScalarOp<_ScalarOp::MAX>);
  m.def("ewise_eq", EwiseOp<_EwiseOp::EQ>);
  m.def("scalar_eq", ScalarOp<_ScalarOp::EQ>);
  m.def("ewise_ge", EwiseOp<_EwiseOp::GE>);
  m.def("scalar_ge", ScalarOp<_ScalarOp::GE>);

  m.def("ewise_log", EwiseFunc<_EwiseOp::LOG>);
  m.def("ewise_exp", EwiseFunc<_EwiseOp::EXP>);
  m.def("ewise_tanh", EwiseFunc<_EwiseOp::TANH>);

  m.def("matmul", Matmul);
  m.def("matmul_tiled", MatmulTiled);

  m.def("reduce_max", ReduceMax);
  m.def("reduce_sum", ReduceSum);
}
