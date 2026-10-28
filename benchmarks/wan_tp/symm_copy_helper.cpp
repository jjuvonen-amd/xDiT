// Copy paths for symm_mem_copy_paths.py, loaded with ctypes:
//   hipcc -O2 -shared -fPIC -o symm_copy_helper.so symm_copy_helper.cpp -lhsa-runtime64
//
// Issues peer copies the ways a framework can on ROCm (HIP memcpy kinds, hipMemcpyPeerAsync,
// ROCr's hsa_amd_memory_async_copy, optionally on a chosen SDMA engine) and reports how HIP and ROCr
// classify a pointer. Functions return 0 on success, a negative code otherwise.
#include <hip/hip_runtime.h>
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>

#include <cstdint>
#include <vector>

namespace {
struct Gpu {
  hsa_agent_t agent;
  uint32_t bus;
};
std::vector<Gpu> g_gpus;
std::vector<hsa_signal_t> g_signals;
std::vector<hipStream_t> g_streams;

hsa_status_t collect(hsa_agent_t agent, void*) {
  hsa_device_type_t type;
  hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &type);
  if (type == HSA_DEVICE_TYPE_GPU) {
    uint32_t bdf = 0;
    hsa_agent_get_info(agent, (hsa_agent_info_t)HSA_AMD_AGENT_INFO_BDFID, &bdf);
    g_gpus.push_back({agent, (bdf >> 8) & 0xff});
  }
  return HSA_STATUS_SUCCESS;
}

int agent_of(int hip_dev, hsa_agent_t* out) {
  hipDeviceProp_t prop;
  if (hipGetDeviceProperties(&prop, hip_dev) != hipSuccess) return -1;
  for (const Gpu& g : g_gpus)
    if (g.bus == (uint32_t)prop.pciBusID) {
      *out = g.agent;
      return 0;
    }
  return -2;
}

int bus_of(hsa_agent_t agent) {
  for (const Gpu& g : g_gpus)
    if (g.agent.handle == agent.handle) return (int)g.bus;
  return -1;
}
}  // namespace

extern "C" {

// Call after the process has selected its device: streams are created on it.
int sch_init(int num_signals, int num_streams) {
  if (hsa_init() != HSA_STATUS_SUCCESS) return -1;
  if (hsa_iterate_agents(collect, nullptr) != HSA_STATUS_SUCCESS) return -2;
  g_signals.resize(num_signals);
  for (auto& s : g_signals)
    if (hsa_signal_create(1, 0, nullptr, &s) != HSA_STATUS_SUCCESS) return -3;
  g_streams.resize(num_streams);
  for (auto& s : g_streams)
    if (hipStreamCreateWithFlags(&s, hipStreamNonBlocking) != hipSuccess) return -4;
  return 0;
}

int sch_hip_copy(void* dst, const void* src, size_t bytes, int stream, int kind) {
  return hipMemcpyAsync(dst, src, bytes, (hipMemcpyKind)kind, g_streams[stream]) == hipSuccess ? 0 : -1;
}

int sch_hip_copy_peer(void* dst, int dst_dev, const void* src, int src_dev, size_t bytes, int stream) {
  return hipMemcpyPeerAsync(dst, dst_dev, src, src_dev, bytes, g_streams[stream]) == hipSuccess ? 0 : -1;
}

int sch_hip_sync(int num_streams) {
  for (int i = 0; i < num_streams; ++i)
    if (hipStreamSynchronize(g_streams[i]) != hipSuccess) return -1;
  return 0;
}

// ROCr async copy between the named agents; ROCr picks the engine. With the same GPU as
// both agents ROCr treats it as a same-device copy; concurrent ones run one at a time.
int sch_hsa_copy(void* dst, int dst_dev, const void* src, int src_dev, size_t bytes, int slot) {
  hsa_agent_t dst_agent, src_agent;
  if (agent_of(dst_dev, &dst_agent) || agent_of(src_dev, &src_agent)) return -1;
  hsa_signal_store_relaxed(g_signals[slot], 1);
  hsa_status_t s = hsa_amd_memory_async_copy(dst, dst_agent, src, src_agent, bytes, 0, nullptr, g_signals[slot]);
  return s == HSA_STATUS_SUCCESS ? 0 : -(int)s;
}

// ROCr async copy forced onto SDMA engine `engine` (bit index) of the named agents.
int sch_hsa_copy_on_engine(void* dst, int dst_dev, const void* src, int src_dev, size_t bytes, int slot,
                           int engine) {
  hsa_agent_t dst_agent, src_agent;
  if (agent_of(dst_dev, &dst_agent) || agent_of(src_dev, &src_agent)) return -1;
  hsa_signal_store_relaxed(g_signals[slot], 1);
  hsa_status_t s = hsa_amd_memory_async_copy_on_engine(dst, dst_agent, src, src_agent, bytes, 0, nullptr,
                                                       g_signals[slot], (hsa_amd_sdma_engine_id_t)(1u << engine), true);
  return s == HSA_STATUS_SUCCESS ? 0 : -(int)s;
}

// The engine (bit index) ROCr prefers for copies from `src_dev` into `dst_dev`: the lowest
// bit of the preferred mask, else of the available mask. Negative on failure.
int sch_engine_for(int dst_dev, int src_dev) {
  hsa_agent_t dst_agent, src_agent;
  if (agent_of(dst_dev, &dst_agent) || agent_of(src_dev, &src_agent)) return -1;
  uint32_t mask = 0;
  if (hsa_amd_memory_get_preferred_copy_engine(dst_agent, src_agent, &mask) != HSA_STATUS_SUCCESS || mask == 0)
    if (hsa_amd_memory_copy_engine_status(dst_agent, src_agent, &mask) != HSA_STATUS_SUCCESS) return -2;
  for (int e = 0; e < 32; ++e)
    if (mask & (1u << e)) return e;
  return -3;
}

// Mask of the SDMA engines available for copies from `src_dev` into `dst_dev`. Negative on failure.
int sch_engine_mask(int dst_dev, int src_dev) {
  hsa_agent_t dst_agent, src_agent;
  if (agent_of(dst_dev, &dst_agent) || agent_of(src_dev, &src_agent)) return -1;
  uint32_t mask = 0;
  if (hsa_amd_memory_copy_engine_status(dst_agent, src_agent, &mask) != HSA_STATUS_SUCCESS) return -2;
  return (int)(mask & 0x7fffffff);
}

int sch_hsa_wait(int num_slots) {
  for (int i = 0; i < num_slots; ++i)
    hsa_signal_wait_scacquire(g_signals[i], HSA_SIGNAL_CONDITION_LT, 1, UINT64_MAX, HSA_WAIT_STATE_ACTIVE);
  return 0;
}

// HIP's view: hipMemoryType (0 unregistered, 1 host, 2 device, 3 managed, ...) and device.
int sch_hip_pointer(const void* p, int* type, int* device) {
  hipPointerAttribute_t attr{};
  hipError_t e = hipPointerGetAttributes(&attr, p);
  if (e != hipSuccess) {
    (void)hipGetLastError();
    *type = -1;
    *device = -1;
    return -1;
  }
  *type = (int)attr.type;
  *device = attr.device;
  return 0;
}

// ROCr's view: hsa_amd_pointer_type_t (0 unknown, 1 HSA, 2 locked, 3 graphics, 4 IPC,
// 5 reserved address, 6 HSA VMEM) and the owning agent's PCI bus.
int sch_hsa_pointer(const void* p, int* type, int* owner_bus) {
  hsa_amd_pointer_info_t info{};
  info.size = sizeof(info);
  if (hsa_amd_pointer_info(p, &info, nullptr, nullptr, nullptr) != HSA_STATUS_SUCCESS) return -1;
  *type = (int)info.type;
  *owner_bus = bus_of(info.agentOwner);
  return 0;
}

int sch_bus_of_device(int hip_dev) {
  hipDeviceProp_t prop;
  return hipGetDeviceProperties(&prop, hip_dev) == hipSuccess ? prop.pciBusID : -1;
}
}
