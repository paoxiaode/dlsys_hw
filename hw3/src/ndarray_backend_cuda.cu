#include <cuda_runtime.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <iostream>
#include <sstream>

namespace needle {
namespace cuda {

#define BASE_THREAD_NUM 256

#define TILE 4
typedef float scalar_t;
const size_t ELEM_SIZE = sizeof(scalar_t);

struct CudaArray {
  CudaArray(const size_t size) {
    cudaError_t err = cudaMalloc(&ptr, size * ELEM_SIZE);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
    this->size = size;
  }
  ~CudaArray() { cudaFree(ptr); }
  size_t ptr_as_int() { return (size_t)ptr; }
  
  scalar_t* ptr;
  size_t size;
};

struct CudaDims {
  dim3 block, grid;
};

CudaDims CudaOneDim(size_t size) {
  /**
   * Utility function to get cuda dimensions for 1D call
   */
  CudaDims dim;
  size_t num_blocks = (size + BASE_THREAD_NUM - 1) / BASE_THREAD_NUM;
  dim.block = dim3(BASE_THREAD_NUM, 1, 1);
  dim.grid = dim3(num_blocks, 1, 1);
  return dim;
}

#define MAX_VEC_SIZE 8
struct CudaVec {
  uint32_t size;
  uint32_t data[MAX_VEC_SIZE];
};

CudaVec VecToCuda(const std::vector<uint32_t>& x) {
  CudaVec shape;
  if (x.size() > MAX_VEC_SIZE) throw std::runtime_error("Exceeded CUDA supported max dimesions");
  shape.size = x.size();
  for (size_t i = 0; i < x.size(); i++) {
    shape.data[i] = x[i];
  }
  return shape;
}

////////////////////////////////////////////////////////////////////////////////
// Fill call
////////////////////////////////////////////////////////////////////////////////

__global__ void FillKernel(scalar_t* out, scalar_t val, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = val;
}

void Fill(CudaArray* out, scalar_t val) {
  CudaDims dim = CudaOneDim(out->size);
  FillKernel<<<dim.grid, dim.block>>>(out->ptr, val, out->size);
}

////////////////////////////////////////////////////////////////////////////////
// Compact and setitem cals
////////////////////////////////////////////////////////////////////////////////

// Untility function to convert contiguous index i to memory location from strides




__global__ void CompactKernel(const scalar_t* a, scalar_t* out, size_t size, CudaVec shape,
                              CudaVec strides, size_t offset, size_t dim) {
  /**
   * The CUDA kernel for the compact opeation.  This should effectively map a single entry in the 
   * non-compact input a, to the corresponding item (at location gid) in the compact array out.
   * 
   * Args:
   *   a: CUDA pointer to a array
   *   out: CUDA point to out array
   *   size: size of out array
   *   shape: vector of shapes of a and out arrays (of type CudaVec, for past passing to CUDA kernel)
   *   strides: vector of strides of out array
   *   offset: offset of out array
   */
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;

  /// BEGIN YOUR SOLUTION
  size_t base = 1;
  for (int i = dim - 1; i >= 0; --i) {
    int cur = gid / base;
    cur %= shape.data[i];
    offset += cur * strides.data[i];
    base *= shape.data[i];
  }

  if (gid < size)
    out[gid] = a[offset];
  /// END YOUR SOLUTION
}

void Compact(const CudaArray& a, CudaArray* out, std::vector<uint32_t> shape,
             std::vector<uint32_t> strides, size_t offset) {
  /**
   * Compact an array in memory.  Unlike the C++ version, in CUDA this will primarily call the 
   * relevant CUDA kernel.  In this case, we illustrate how you should set this up (i.e., we give 
   * you the code for this fuction, and also the prototype for the CompactKernel() function).  For
   * the functions after this, however, you'll need to define these kernels as you see fit to 
   * execute the underlying function.
   * 
   * Args:
   *   a: non-compact represntation of the array, given as input
   *   out: compact version of the array to be written
   *   shape: shapes of each dimension for a and out
   *   strides: strides of the *a* array (not out, which has compact strides)
   *   offset: offset of the *a* array (not out, which has zero offset, being compact)
   */

  // Nothing needs to be added here
  CudaDims dim = CudaOneDim(out->size);
  size_t _dim = shape.size();
  CompactKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size, VecToCuda(shape),
                                         VecToCuda(strides), offset, _dim);
}


__global__ void EwiseSetitemKernel(const scalar_t* a, scalar_t* out, size_t size, CudaVec shape,
                              CudaVec strides, size_t offset, size_t dim) {
  /**
   * Set items in a (non-compact) array using CUDA.  Yyou will most likely want to implement a
   * EwiseSetitemKernel() function, similar to those above, that will do the actual work.
   * 
   * Args:
   *   a: _compact_ array whose items will be written to out
   *   out: non-compact array whose items are to be written
   *   shape: shapes of each dimension for a and out
   *   strides: strides of the *out* array (not a, which has compact strides)
   *   offset: offset of the *out* array (not a, which has zero offset, being compact)
   */
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;

  /// BEGIN YOUR SOLUTION
  size_t base = 1;
  for (int i = dim - 1; i >= 0; --i) {
    int cur = gid / base;
    cur %= shape.data[i];
    offset += cur * strides.data[i];
    base *= shape.data[i];
  }

  if (gid < size)
    out[offset] = a[gid];
  /// END YOUR SOLUTION
}


void EwiseSetitem(const CudaArray& a, CudaArray* out, std::vector<uint32_t> shape,
                  std::vector<uint32_t> strides, size_t offset) {
  /**
   * Set items in a (non-compact) array using CUDA.  Yyou will most likely want to implement a
   * EwiseSetitemKernel() function, similar to those above, that will do the actual work.
   * 
   * Args:
   *   a: _compact_ array whose items will be written to out
   *   out: non-compact array whose items are to be written
   *   shape: shapes of each dimension for a and out
   *   strides: strides of the *out* array (not a, which has compact strides)
   *   offset: offset of the *out* array (not a, which has zero offset, being compact)
   */
  /// BEGIN YOUR SOLUTION
  CudaDims dim = CudaOneDim(out->size);
  size_t _dim = shape.size();
  EwiseSetitemKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, a.size, VecToCuda(shape),
                                         VecToCuda(strides), offset, _dim);
  /// END YOUR SOLUTION
}


__global__ void ScalarSetitemKernel(scalar_t* out, size_t size, scalar_t val, CudaVec shape,
                              CudaVec strides, size_t offset, size_t dim) {
  /**
   * Set items is a (non-compact) array
   * 
   * Args:
   *   size: number of elements to write in out array (note that this will note be the same as
   *         out.size, because out is a non-compact subset array);  it _will_ be the same as the 
   *         product of items in shape, but covenient to just pass it here.
   *   val: scalar value to write to
   *   out: non-compact array whose items are to be written
   *   shape: shapes of each dimension of out
   *   strides: strides of the out array
   *   offset: offset of the out array
   */
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;

  /// BEGIN YOUR SOLUTION
  size_t base = 1;
  for (int i = dim - 1; i >= 0; --i) {
    int cur = gid / base;
    cur %= shape.data[i];
    offset += cur * strides.data[i];
    base *= shape.data[i];
  }

  if (gid < size)
    out[offset] = val;
  /// END YOUR SOLUTION
}


void ScalarSetitem(size_t size, scalar_t val, CudaArray* out, std::vector<uint32_t> shape,
                   std::vector<uint32_t> strides, size_t offset) {
  /**
   * Set items is a (non-compact) array
   * 
   * Args:
   *   size: number of elements to write in out array (note that this will note be the same as
   *         out.size, because out is a non-compact subset array);  it _will_ be the same as the 
   *         product of items in shape, but covenient to just pass it here.
   *   val: scalar value to write to
   *   out: non-compact array whose items are to be written
   *   shape: shapes of each dimension of out
   *   strides: strides of the out array
   *   offset: offset of the out array
   */
  /// BEGIN YOUR SOLUTION
  CudaDims dim = CudaOneDim(out->size);
  size_t _dim = shape.size();
  ScalarSetitemKernel<<<dim.grid, dim.block>>>(out->ptr, size, val, VecToCuda(shape),
                                         VecToCuda(strides), offset, _dim);
  /// END YOUR SOLUTION
}

////////////////////////////////////////////////////////////////////////////////
// Elementwise and scalar operations
////////////////////////////////////////////////////////////////////////////////

__global__ void EwiseAddKernel(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = a[gid] + b[gid];
}

void EwiseAdd(const CudaArray& a, const CudaArray& b, CudaArray* out) {
  /**
   * Add together two CUDA array
   */
  CudaDims dim = CudaOneDim(out->size);
  EwiseAddKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size);
}

__global__ void ScalarAddKernel(const scalar_t* a, scalar_t val, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = a[gid] + val;
}

void ScalarAdd(const CudaArray& a, scalar_t val, CudaArray* out) {
  /**
   * Add together a CUDA array and a scalar value.
   */
  CudaDims dim = CudaOneDim(out->size);
  ScalarAddKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);
}

/**
 * In the code the follows, use the above template to create analogous elementise
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

__global__ void EwiseOpKernel(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t size, _EwiseOp op) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size){
    if(op == _EwiseOp::MUL) out[gid] = a[gid] * b[gid];
    if(op == _EwiseOp::DIV) out[gid] = a[gid] / b[gid];
    if(op == _EwiseOp::MAX) out[gid] = fmaxf(a[gid], b[gid]);
    if(op == _EwiseOp::EQ) out[gid] = (scalar_t)(a[gid] == b[gid]);
    if(op == _EwiseOp::GE) out[gid] = (scalar_t)(a[gid] >= b[gid]);
  }
}

template<_EwiseOp op>
void EwiseOp(const CudaArray& a, const CudaArray& b, CudaArray* out) {
  /**
   * Add together two CUDA array
   */
  CudaDims dim = CudaOneDim(out->size);
  EwiseOpKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size, op);
}

__global__ void EwiseFuncKernel(const scalar_t* a, scalar_t* out, size_t size, _EwiseOp op) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size){
    if(op == _EwiseOp::LOG) out[gid] = logf(a[gid]);
    if(op == _EwiseOp::EXP) out[gid] = expf(a[gid]);
    if(op == _EwiseOp::TANH) out[gid] = tanhf(a[gid]);
  }
}

template<_EwiseOp op>
void EwiseFunc(const CudaArray& a, CudaArray* out) {
  /**
   * Add together two CUDA array
   */
  CudaDims dim = CudaOneDim(out->size);
  EwiseFuncKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size, op);
}

__global__ void ScalarOpKernel(const scalar_t* a, scalar_t val, scalar_t* out, size_t size, _ScalarOp op) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size){
    if(op == _ScalarOp::MUL) out[gid] = a[gid] * val;
    if(op == _ScalarOp::DIV) out[gid] = a[gid] / val;
    if(op == _ScalarOp::MAX) out[gid] = fmaxf(a[gid], val);
    if(op == _ScalarOp::POWER) out[gid] = powf(a[gid], val);
    if(op == _ScalarOp::EQ) out[gid] = (scalar_t)(a[gid] == val);
    if(op == _ScalarOp::GE) out[gid] = (scalar_t)(a[gid] >= val);
  }
}

template<_ScalarOp op>
void ScalarOp(const CudaArray& a, scalar_t val, CudaArray* out) {
  /**
   * Add together two CUDA array
   */
  CudaDims dim = CudaOneDim(out->size);
  ScalarOpKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size, op);
}


/// END YOUR SOLUTION

////////////////////////////////////////////////////////////////////////////////
// Elementwise and scalar operations
////////////////////////////////////////////////////////////////////////////////


const int TILE_SIZE = 8;

__global__ void MatMulKernel_Tiling(const scalar_t* A, const scalar_t* B, scalar_t* C, 
            uint32_t M, uint32_t N, uint32_t P) {
	/* Basic tiling implementation of matrix multiplication.
	 * Based on a more mathematically reasonable indexing method.
	 */
	int bx = blockIdx.x, by = blockIdx.y;
	int tx = threadIdx.x, ty = threadIdx.y;

	__shared__ float As[TILE_SIZE][TILE_SIZE];
	__shared__ float Bs[TILE_SIZE][TILE_SIZE];

	int aBegin = N * blockDim.y * by;
	int aEnd = aBegin + N - 1;
	int aStep =  blockDim.x;

	int bBegin = blockDim.x * bx;
	int bStep = blockDim.y * P;

	float Csub = 0;

	for (int i = aBegin, j = bBegin; i <= aEnd; i += aStep, j += bStep) {
        // calculate tile C_ij
        // record tile A_ik and tile B00 B_kj
		As[ty][tx] = A[i + N * ty + tx];
		Bs[tx][ty] = B[j + P * tx + ty];

		__syncthreads();

		for (int k = 0; k < TILE_SIZE; ++k) {
			Csub += As[ty][k]*Bs[k][tx];
		}
		
		__syncthreads();
	}
	int cIdx = P * blockDim.y * by + blockDim.x * bx;
	C[cIdx + P * ty + tx] = Csub;
}

__global__ void MatMulKernel_naive(const scalar_t* A, const scalar_t* B, scalar_t* C, 
            uint32_t M, uint32_t N, uint32_t P) {
	/* Basic tiling implementation of matrix multiplication.
	 * Based on a more mathematically reasonable indexing method.
	 */
	size_t tx = blockDim.x * blockIdx.x + threadIdx.x;
  size_t ty = blockDim.y * blockIdx.y + threadIdx.y;
  size_t gid = tx * P + ty;
  if (tx < M && ty < P) {
    scalar_t val = 0.0;
    for (size_t i = 0; i < N; ++i) {
      val += A[tx * N + i] * B[ty + i * P];
    }
    C[gid] = val;
  }
}

bool div_exa(uint32_t n){
  return (n % TILE_SIZE == 0);
}

void Matmul(const CudaArray& a, const CudaArray& b, CudaArray* out, uint32_t M, uint32_t N,
            uint32_t P) {
  /**
   * Multiply two (compact) matrices into an output (also comapct) matrix.  You will want to look
   * at the lecture and notes on GPU-based linear algebra to see how to do this.  Since ultimately
   * mugrade is just evaluating correctness, you _can_ implement a version that simply parallelizes
   * over (i,j) entries in the output array.  However, to really get the full benefit of this
   * problem, we would encourage you to use cooperative fetching, shared memory register tiling, 
   * and other ideas covered in the class notes.  Note that unlike the tiled matmul function in
   * the CPU backend, here you should implement a single function that works across all size
   * matrices, whether or not they are a multiple of a tile size.  As with previous CUDA
   * implementations, this function here will largely just set up the kernel call, and you should
   * implement the logic in a separate MatmulKernel() call.
   * 
   *
   * Args:
   *   a: compact 2D array of size m x n
   *   b: comapct 2D array of size n x p
   *   out: compact 2D array of size m x p to write the output to
   *   M: rows of a / out
   *   N: columns of a / rows of b
   *   P: columns of b / out
   */

  /// BEGIN YOUR SOLUTION
  std::vector<uint32_t> dim{M, N, P};
  if(std::all_of(dim.begin(), dim.end(), div_exa)){
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((M + block.x - 1)/block.x, (P + block.y - 1)/block.y);
    MatMulKernel_Tiling<<<grid, block>>>(a.ptr, b.ptr, out->ptr, M, N, P);
  }
  else{
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((M + block.x - 1)/block.x, (P + block.y - 1)/block.y);
    MatMulKernel_naive<<<grid, block>>>(a.ptr, b.ptr, out->ptr, M, N, P);
  }
  /// END YOUR SOLUTION
}

////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////

enum class _ReduceOp{
  Max, Sum
};

__global__ void ReduceOpKernel(const scalar_t* a, scalar_t* out, size_t size, size_t reduce_size, _ReduceOp op) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  size_t gid_block = gid * reduce_size;
  scalar_t cur = a[gid_block];
  if(op == _ReduceOp::Max){
    for(int i = 1; i < reduce_size; i++){
      cur = fmaxf(cur, a[gid_block+i]);
    }
  }
  if(op == _ReduceOp::Sum){
    for(int i = 1; i < reduce_size; i++){
      cur += a[gid_block+i];
    }
  }
  if (gid < size) out[gid] = cur;
}

template<_ReduceOp op>
void ReduceOp(const CudaArray& a, CudaArray* out, size_t reduce_size) {
  /**
   * Reduce by taking maximum over `reduce_size` contiguous blocks.  Even though it is inefficient,
   * for simplicity you can perform each reduction in a single CUDA thread.
   * 
   * Args:
   *   a: compact array of size a.size = out.size * reduce_size to reduce over
   *   out: compact array to write into
   *   redice_size: size of the dimension to reduce over
   */
  /// BEGIN YOUR SOLUTION
  CudaDims dim = CudaOneDim(out->size);
  ReduceOpKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size, reduce_size, op);
  /// END YOUR SOLUTION
}



}  // namespace cuda
}  // namespace needle

PYBIND11_MODULE(ndarray_backend_cuda, m) {
  namespace py = pybind11;
  using namespace needle;
  using namespace cuda;

  m.attr("__device_name__") = "cuda";
  m.attr("__tile_size__") = TILE;

  py::class_<CudaArray>(m, "Array")
      .def(py::init<size_t>(), py::return_value_policy::take_ownership)
      .def_readonly("size", &CudaArray::size)
      .def("ptr", &CudaArray::ptr_as_int);

  // return numpy array, copying from CPU
  m.def("to_numpy", [](const CudaArray& a, std::vector<size_t> shape, std::vector<size_t> strides,
                       size_t offset) {
    std::vector<size_t> numpy_strides = strides;
    std::transform(numpy_strides.begin(), numpy_strides.end(), numpy_strides.begin(),
                   [](size_t& c) { return c * ELEM_SIZE; });

    // copy memory to host
    scalar_t* host_ptr = (scalar_t*)std::malloc(a.size * ELEM_SIZE);
    if (host_ptr == 0) throw std::bad_alloc();
    cudaError_t err = cudaMemcpy(host_ptr, a.ptr, a.size * ELEM_SIZE, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));

    // return numpy array
    py::capsule deallocate_buffer(host_ptr, [](void* p) { free(p); });
    return py::array_t<scalar_t>(shape, numpy_strides, host_ptr + offset, deallocate_buffer);
  });

  // copy numpy array to GPU
  m.def("from_numpy", [](py::array_t<scalar_t> a, CudaArray* out) {
    cudaError_t err =
        cudaMemcpy(out->ptr, a.request().ptr, out->size * ELEM_SIZE, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
  });

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

  m.def("reduce_max", ReduceOp<_ReduceOp::Max>);
  m.def("reduce_sum", ReduceOp<_ReduceOp::Sum>);
}
