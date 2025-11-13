#include <cuda.h>
#include <iomanip>
#include <iostream>
#include <unordered_map>
#include <memory>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>

// Boxes
// =====
//
// Objects to hold and manage the lifetimes of CUDA resources. A CUDA
// resource is anthing allocated from the CUDA driver that requires a
// matching call to a deallocation function. Boxes are named like the
// resources they contain, in camel case, without the CU prefix: e.g.,
// Stream, MemPool, Deviceptr.
//
// Contents:
//
//   - CUDA resource
//       The boxed CUDA resource. E.g., CUstream, CUdeviceptr, CUmemoryPool
//
//   - Resource owner holders
//       Owners of the resource, whose lifetimes should be extended according
//       to the resource lifetime. For example, a memory pool holder
//       (MemPoolH) for the memory pool that owns a device memory allocation
//       (Deviceptr).
//
//   - Descructor arguments
//       Additional arguments and resource holders needed to call the
//       dealloation function. E.g., for Deviceptr, a stream holder (StreamH)
//       specifying the stream to deallocate on when using cuMemFreeAsyc.
//
// Properties:
//
//   - Default constructible
//       A default-constructed box contains a default resource instance. This
//       could be an invalid resource or a valid global/static resource, such
//       as the default context or stream. Whether the resource is valid or not,
//       a default-constructed box is a valid box, which is necessary for wrapping
//       boxes as Python objects.
//
//   - Copyable/Moveable
//       Boxes should use default destructors that do not free the boxed
//       resource, but do destroy and free and resource holders referring to
//       owners or destructor arguments.
//
//
// Holders
// =======
//
// A holder is a shared pointer to a box, specifically something like:
//
//     using StreamH = std::shared_ptr<Stream>;
//
// The naming convention is as indicated above: the name of a holder type
// matches that of the corresponding box with 'H' appended.
//
// Properties:
//
//   - Deletion is resource release
//       The shared_ptr deleter frees the boxed CUDA resource.
//
//   - Resettable
//       Resetting the holder to a default-initialized box drops a reference,
//       potentially freeing the boxed CUDA resource, while retaining a valid
//       box. This allows holders to serve as pybind11 holders.
//
//
// Python Holders
// ==============
//
// Boxes are exposed as Python objects in a holders module. Stream is exposed
// as holders.Stream, MemPool as holders.MemPool, and so on. These objects can
// be held in Python/Cython code to manage CUDA resource lifetimes outside of
// Python.
//
// Properties:
//
//   - Python holders are constructed from int-like handles to CUDA resource,
//     plus additional argument describing owners and destructor arguments.
//
//   - CUDA resource handles can be obtained as int-like object either by
//     converting a Python holder to int, or accessing its `value` method.
//
//   - Python holders can be closed by calling the `close` method. This
//     converts the Python object into a default instance (corresponding to a
//     default-constructed box).
//
//   - If applicable, these objects may expose an interface for updating
//     destructor arguments (e.g., `set_stream` for memory allocations).


namespace py = pybind11;

#define ENABLE_DIAGNOSTICS

#ifdef ENABLE_DIAGNOSTICS
  #define MESSAGE(body) std::cerr << body << std::endl;
  #define USAGE(expr) g_usage.expr
#else
  #define MESSAGE(body)
  #define USAGE(expr)
#endif

#define CUDA_CHECK(call) do { \
    CUresult const result = call; \
    if (result != CUDA_SUCCESS) { ::raise_cuda_error(result); } \
} while(0)

namespace
{
  [[noreturn]] void raise_cuda_error(CUresult result)
  {
    char const * cuda_msg = nullptr;
    cuGetErrorString(result, &cuda_msg);
    auto msg =
        std::string("CUDA error ")
      + std::to_string(static_cast<int>(result)) + ": "
      + cuda_msg;
    throw std::runtime_error(msg);
  }

  template<typename Action>
  auto on_scope_exit(Action && action)
  {
    auto deleter = [action = std::move(action)](void*) { action(); };
    return std::unique_ptr<void, decltype(deleter)>((void *) 0x00c0ffee, deleter);
  }

  template <typename T>
  uintptr_t to_uintptr(T v) {
      if constexpr (std::is_pointer_v<T>)
          return reinterpret_cast<uintptr_t>(v);
      else
          return static_cast<uintptr_t>(v);
  }

  #ifdef ENABLE_DIAGNOSTICS
  static struct CudaResourceUsage
  {
    int streams = 0;
    int mempools = 0;
    int devptrs = 0;

    void report()
    {
      std::cerr << "\n"
                   "CUDA Core Resource Usage Report\n"
                   "===============================\n"
                   "Currently in use:\n"
                << "    #streams : " << this->streams  << "\n"
                << "    #mempools: " << this->mempools << "\n"
                << "    #devptrs : " << this->devptrs  << "\n"
      ;
    }

    ~CudaResourceUsage() { this->report(); }
  } g_usage;
  #endif

  // Boxes
  struct Stream;
  struct MemPool;
  struct Deviceptr;

  // Holders
  using StreamH = std::shared_ptr<Stream>;
  using MemPoolH = std::shared_ptr<MemPool>;
  using DeviceptrH = std::shared_ptr<Deviceptr>;

  template<typename Box> using Cache =
      std::unordered_map<uintptr_t, std::weak_ptr<Box>>;

  // Box definitions
  struct Stream
  {
    CUstream res = CU_STREAM_PER_THREAD;

    static Cache<Stream> cache;
    static constexpr char const * class_name = "Stream";
    static constexpr char const * cuda_resource_name = "CUstream";

    Stream() = default;
    Stream(CUstream res) : res{res} {}

    uintptr_t as_int() const { return to_uintptr(res); }

    static auto capture(uintptr_t i_res) -> StreamH
    {
      USAGE(streams += 1);
      MESSAGE("Capturing Stream 0x" << std::hex << i_res);
      auto res = reinterpret_cast<CUstream>(i_res);
      return StreamH(new Stream(res), [](auto * box)
        {
          USAGE(streams -= 1);
          MESSAGE("Releasing Stream 0x" << std::hex << box->as_int());
          auto _ = on_scope_exit([=]{ delete box; });
          CUDA_CHECK(cuStreamDestroy(box->res));
        });
    }

    static auto capture_static(uintptr_t i_res) -> StreamH
    {
      MESSAGE("Wrapping static Stream 0x" << std::hex << i_res);
      auto res = reinterpret_cast<CUstream>(i_res);
      return StreamH(new Stream(res));
    }
  };

  Cache<Stream> Stream::cache;

  struct MemPool
  {
    CUmemoryPool res = nullptr;

    static Cache<MemPool> cache;
    static constexpr char const * class_name = "MemPool";
    static constexpr char const * cuda_resource_name = "CUmemoryPool";

    MemPool() = default;
    MemPool(CUmemoryPool res) : res{res} {}

    uintptr_t as_int() const { return to_uintptr(res); }

    static auto capture(uintptr_t i_res) -> MemPoolH
    {
      USAGE(mempools += 1);
      MESSAGE("Capturing MemPool 0x" << std::hex << i_res);
      auto res = reinterpret_cast<CUmemoryPool>(i_res);
      return MemPoolH(new MemPool(res), [](auto * box)
        {
          USAGE(mempools -= 1);
          MESSAGE("Releasing MemPool 0x" << std::hex << box->as_int());
          auto _ = on_scope_exit([=]{ delete box; });
          CUDA_CHECK(cuMemPoolDestroy(box->res));
        });
    }

    static auto capture_static(uintptr_t i_res) -> MemPoolH
    {
      MESSAGE("Wrapping static MemPool 0x" << std::hex << i_res);
      auto res = reinterpret_cast<CUmemoryPool>(i_res);
      return MemPoolH(new MemPool(res));
    }
  };

  Cache<MemPool> MemPool::cache;

  struct Deviceptr
  {
    CUdeviceptr res = 0;
    MemPoolH h_pool;
    StreamH h_stream;
    static Cache<Deviceptr> cache;
    static constexpr char const * class_name = "Deviceptr";
    static constexpr char const * cuda_resource_name = "CUdeviceptr";

    Deviceptr() = default;
    Deviceptr(
        CUdeviceptr res
      , MemPoolH const & h_pool = MemPoolH{}
      , StreamH const & h_stream = StreamH{}
      )
      : res{res}, h_pool{h_pool}, h_stream{h_stream}
    {}

    uintptr_t as_int() const { return to_uintptr(res); }

    static auto capture(
        uintptr_t i_res, MemPoolH const & h_pool, StreamH const & h_stream
      ) -> DeviceptrH
    {
      USAGE(devptrs += 1);
      MESSAGE("Capturing Deviceptr 0x" << std::hex << i_res);
      auto res = static_cast<CUdeviceptr>(i_res);
      return DeviceptrH(new Deviceptr(res, h_pool, h_stream), [](auto * box)
        {
          USAGE(devptrs -= 1);
          MESSAGE("Releasing Deviceptr 0x" << std::hex << box->as_int());
          auto _ = on_scope_exit([=]{ delete box; });
          CUDA_CHECK(cuMemFreeAsync(box->res, box->h_stream->res));
        });
    }

    static auto capture_static(uintptr_t i_res) -> DeviceptrH
    {
      MESSAGE("Wrapping static Deviceptr 0x" << std::hex << i_res);
      auto res = static_cast<CUdeviceptr>(i_res);
      return DeviceptrH(new Deviceptr(res));
    }
  };

  Cache<Deviceptr> Deviceptr::cache;

  template<typename Box, typename ... Args>
  auto capture_cached(uintptr_t i_res, Args && ... args)
  {
    using Holder = std::shared_ptr<Box>;
    auto it = Box::cache.find(i_res);
    if (it != Box::cache.end()) {
        auto sp = it->second.lock();
        if (sp) {
            MESSAGE("Returning cached " << Box::class_name << " 0x" << std::hex << i_res);
            return Holder(sp);
        } else {
            Box::cache.erase(it);
        }
    }

    auto sp = Box::capture(i_res, std::forward<Args&&>(args)...);
    Box::cache[i_res] = std::weak_ptr<Box>(sp);
    return sp;
  }

  // Make a Python class wrapping a CUDA resource box that exposes the resource
  // (as an integer), is showable and resettable, and provided make_static.
  template<typename Box>
  auto py_class(py::module & m)
  {
    using Holder = std::shared_ptr<Box>;
    return py::class_<Box, Holder>(m, Box::class_name)
      .def("__int__", &Box::as_int)
      .def_property_readonly("value", &Box::as_int)
      .def("reset", [](Holder & self) { self.reset(new Box{}); })
      .def("__repr__", [=](Box const & self) {
          std::ostringstream oss;
          oss << Box::cuda_resource_name << "=0x" << std::hex << self.as_int();
          return py::str(oss.str());
      });
  }
}


PYBIND11_MODULE(cuda_core_holders_demo, m)
{
  m.doc() = "Provides CUDA resource holders";

  #ifdef ENABLE_DIAGNOSTICS
  m.def("report_usage", [](){ g_usage.report(); });
  #endif

  py_class<Stream>(m)
    .def_static("capture", &Stream::capture)
    .def_static("capture_static", &Stream::capture_static)
    ;

  py_class<MemPool>(m)
    .def_static("capture", &MemPool::capture)
    .def_static("capture_cached", (MemPoolH(*)(uintptr_t)) &capture_cached<MemPool>)
    .def_static("capture_static", &MemPool::capture_static)
    ;

  py_class<Deviceptr>(m)
    .def_static("capture", &Deviceptr::capture)
    .def_static("capture_static", &Deviceptr::capture_static)
    .def("set_stream", [](DeviceptrH const & h_devp, StreamH const & h_stream)
        { h_devp->h_stream = h_stream; })
    ;
}

