// ROCr SDMA transport for xfuser::tp_sdma_all_gather_linear (see sdma_all_gather.py), loaded with ctypes.
//
// Each peer's shard is copied by its own SDMA engine, all peers at once. Each copy names the
// local GPU on both sides: the peer's symmetric-memory buffer is mapped in this process for
// the local GPU only. A 4-byte copy on the same engine, dependent on the data copy, then sets
// that peer's flag, which the consuming GEMM waits on. The data copies depend on a trigger
// signal that a one-thread kernel releases on the compute stream, after the symmetric-memory
// barrier, so no copy starts before every peer's shard is in place.
//
// Functions return 0 on success and a negative code otherwise.
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

struct Slot {
  hsa_signal_t trigger;
  volatile hsa_signal_value_t* trigger_value;
  std::vector<hsa_signal_t> data;
  std::vector<hsa_signal_t> flag;
};

std::vector<Gpu> g_gpus;
std::vector<Slot> g_slots;
hsa_agent_t g_local;

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

void wait_done(hsa_signal_t s) {
  hsa_signal_wait_scacquire(s, HSA_SIGNAL_CONDITION_LT, 1, UINT64_MAX, HSA_WAIT_STATE_ACTIVE);
}

__global__ void release_trigger(hsa_signal_value_t* value) {
  __hip_atomic_store(value, (hsa_signal_value_t)0, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
}

}  // namespace

extern "C" {

// `slots` calls may be in flight at once; a slot is reused only after its copies finished.
int sag_init(int local_dev, int max_peers, int slots) {
  if (!g_slots.empty()) return 0;
  if (hsa_init() != HSA_STATUS_SUCCESS) return -1;
  if (hsa_iterate_agents(collect, nullptr) != HSA_STATUS_SUCCESS) return -2;
  if (agent_of(local_dev, &g_local)) return -3;
  g_slots.resize(slots);
  for (Slot& s : g_slots) {
    // The trigger is written by a kernel, so its value must be addressable: GPU-only signal.
    if (hsa_amd_signal_create(0, 0, nullptr, HSA_AMD_SIGNAL_AMD_GPU_ONLY, &s.trigger) != HSA_STATUS_SUCCESS)
      return -4;
    if (hsa_amd_signal_value_pointer(s.trigger, &s.trigger_value) != HSA_STATUS_SUCCESS) return -5;
    s.data.resize(max_peers);
    s.flag.resize(max_peers);
    for (int i = 0; i < max_peers; ++i)
      if (hsa_signal_create(0, 0, nullptr, &s.data[i]) != HSA_STATUS_SUCCESS ||
          hsa_signal_create(0, 0, nullptr, &s.flag[i]) != HSA_STATUS_SUCCESS)
        return -6;
  }
  return 0;
}

// One SDMA engine (bit index) per peer for copies into `local_dev`: ROCr's preferred engine for
// that link, and a peer whose preferred engine is already taken gets the lowest available
// engine that no peer prefers (then any free one).
int sag_engine_plan(int local_dev, const int* peer_devs, int n, int* engines) {
  hsa_agent_t local, peer;
  if (agent_of(local_dev, &local)) return -1;
  std::vector<uint32_t> avail(n), preferred(n, 0);
  for (int i = 0; i < n; ++i) {
    if (agent_of(peer_devs[i], &peer)) return -2;
    if (hsa_amd_memory_copy_engine_status(local, peer, &avail[i]) != HSA_STATUS_SUCCESS) return -3;
    uint32_t mask = 0;
    if (hsa_amd_memory_get_preferred_copy_engine(local, peer, &mask) == HSA_STATUS_SUCCESS) preferred[i] = mask;
  }
  auto lowest = [](uint32_t mask) {
    for (int e = 0; e < 32; ++e)
      if (mask >> e & 1) return e;
    return -1;
  };
  uint32_t used = 0, wanted = 0;
  for (int i = 0; i < n; ++i) {
    engines[i] = lowest(preferred[i] & avail[i]);
    if (engines[i] >= 0) wanted |= 1u << engines[i];
  }
  for (int i = 0; i < n; ++i) {
    if (engines[i] >= 0 && !(used >> engines[i] & 1)) {
      used |= 1u << engines[i];
      continue;
    }
    int e = lowest(avail[i] & ~used & ~wanted);
    if (e < 0) e = lowest(avail[i] & ~used);
    if (e < 0) return -4;
    engines[i] = e;
    used |= 1u << e;
  }
  return 0;
}

// Arms `slot` for a call: waits for its previous use to finish and sets every signal to 1.
// Call before sag_release and sag_issue.
int sag_arm(int slot) {
  Slot& s = g_slots[slot];
  for (size_t i = 0; i < s.data.size(); ++i) {
    wait_done(s.data[i]);
    wait_done(s.flag[i]);
  }
  hsa_signal_store_relaxed(s.trigger, 1);
  for (size_t i = 0; i < s.data.size(); ++i) {
    hsa_signal_store_relaxed(s.data[i], 1);
    hsa_signal_store_relaxed(s.flag[i], 1);
  }
  return 0;
}

// Enqueues the release of `slot`'s trigger on `stream`: copies gated on it start once the
// stream reaches this point.
int sag_release(int slot, hipStream_t stream) {
  hipLaunchKernelGGL(release_trigger, dim3(1), dim3(1), 0, stream,
                     const_cast<hsa_signal_value_t*>(g_slots[slot].trigger_value));
  return hipGetLastError() == hipSuccess ? 0 : -1;
}

// Copies peer i's `bytes` from src[i] to dst[i] on SDMA engine engines[i], then writes the 4
// bytes at `one` to flags + 4 * flag_index[i]. With `gated`, the data copies wait for the
// slot's trigger; otherwise they start at once.
int sag_issue(int slot, const uint64_t* dst, const uint64_t* src, uint64_t bytes, const int* engines,
              const int* flag_index, int n, uint64_t flags, uint64_t one, int gated) {
  Slot& s = g_slots[slot];
  if (n > (int)s.data.size()) return -1;
  if (!gated) hsa_signal_store_relaxed(s.trigger, 0);
  for (int i = 0; i < n; ++i) {
    hsa_amd_sdma_engine_id_t engine = (hsa_amd_sdma_engine_id_t)(1u << engines[i]);
    hsa_status_t st = hsa_amd_memory_async_copy_on_engine(
        (void*)dst[i], g_local, (const void*)src[i], g_local, bytes, gated ? 1 : 0, gated ? &s.trigger : nullptr,
        s.data[i], engine, true);
    if (st != HSA_STATUS_SUCCESS) return -100 - (int)st;
    st = hsa_amd_memory_async_copy_on_engine((void*)(flags + 4 * (uint64_t)flag_index[i]), g_local,
                                             (const void*)one, g_local, 4, 1, &s.data[i], s.flag[i], engine,
                                             true);
    if (st != HSA_STATUS_SUCCESS) return -200 - (int)st;
  }
  return 0;
}

// Enqueues a wait on `stream` until the 32-bit value at `ptr` equals `value`.
int sag_stream_wait_value32(hipStream_t stream, uint64_t ptr, uint32_t value) {
  return hipStreamWaitValue32(stream, (void*)ptr, value, hipStreamWaitValueEq, 0xFFFFFFFFu) == hipSuccess ? 0 : -1;
}
}
