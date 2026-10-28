// ROCr SDMA transport for xfuser::tp_sdma_all_gather_linear (see sdma_all_gather.py), loaded with ctypes.
//
// Each peer's shard is copied by its own SDMA engine, all peers at once, in pieces queued in
// order on that engine. Each copy names the local GPU on both sides: the peer's
// symmetric-memory buffer is mapped in this process for the local GPU only. A 4-byte copy on
// the same engine, dependent on each piece, then sets that piece's flag, which the compute
// stream waits on before the piece's GEMM. The data copies depend on a trigger signal that a
// one-thread kernel releases on the compute stream after the symmetric-memory barrier, so no
// copy starts before every peer's shard is in place.
//
// Functions return 0 on success and a negative code otherwise.
#include <hip/hip_runtime.h>
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>

#include <chrono>
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
  std::vector<hsa_signal_t> data;  // one per copy of a call
  std::vector<hsa_signal_t> flag;
  int armed = 0;  // copies the last call on this slot made
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

// Waits until `s` drops below 1 (its copy finished); false after kWaitSeconds, so a copy
// that never runs fails the call instead of hanging it.
constexpr double kWaitSeconds = 60.0;

bool wait_done(hsa_signal_t s) {
  auto deadline = std::chrono::steady_clock::now() + std::chrono::duration<double>(kWaitSeconds);
  while (hsa_signal_load_scacquire(s) >= 1) {
    if (std::chrono::steady_clock::now() > deadline) return false;
    hsa_signal_wait_scacquire(s, HSA_SIGNAL_CONDITION_LT, 1, 1000000, HSA_WAIT_STATE_ACTIVE);
  }
  return true;
}

int lowest(uint32_t mask) {
  for (int e = 0; e < 32; ++e)
    if (mask >> e & 1) return e;
  return -1;
}

uint32_t pair_of(uint32_t mask) { return ((mask & 0x55555555u) << 1) | ((mask & 0xAAAAAAAAu) >> 1); }

__global__ void release_trigger(hsa_signal_value_t* value) {
  __hip_atomic_store(value, (hsa_signal_value_t)0, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
}

}  // namespace

extern "C" {

// `slots` calls may be in flight at once, each of at most `max_copies` copies; a slot is
// reused only after its copies finished.
int sag_init(int local_dev, int max_copies, int slots) {
  if (!g_slots.empty()) return (int)g_slots[0].data.size() >= max_copies ? 0 : -7;
  if (hsa_init() != HSA_STATUS_SUCCESS) return -1;
  if (hsa_iterate_agents(collect, nullptr) != HSA_STATUS_SUCCESS) return -2;
  if (agent_of(local_dev, &g_local)) return -3;
  g_slots.resize(slots);
  for (Slot& s : g_slots) {
    // The trigger is written by a kernel, so its value must be addressable: GPU-only signal.
    if (hsa_amd_signal_create(0, 0, nullptr, HSA_AMD_SIGNAL_AMD_GPU_ONLY, &s.trigger) != HSA_STATUS_SUCCESS)
      return -4;
    if (hsa_amd_signal_value_pointer(s.trigger, &s.trigger_value) != HSA_STATUS_SUCCESS) return -5;
    s.data.resize(max_copies);
    s.flag.resize(max_copies);
    for (int i = 0; i < max_copies; ++i)
      if (hsa_signal_create(0, 0, nullptr, &s.data[i]) != HSA_STATUS_SUCCESS ||
          hsa_signal_create(0, 0, nullptr, &s.flag[i]) != HSA_STATUS_SUCCESS)
        return -6;
  }
  return 0;
}

// One SDMA engine (bit index) per peer for copies into `local_dev`.
//   mode 0 (fixed): ROCr's preferred engine for that link; a peer whose preferred engine is
//     taken gets the lowest available engine no peer prefers (then any free one).
//   mode 1 (pairfree): the same, but no two peers use engines 2k and 2k+1, which may share
//     bandwidth; falls back to mode 0's choice when no such engine is left.
int sag_engine_plan(int local_dev, const int* peer_devs, int n, int mode, int* engines) {
  hsa_agent_t local, peer;
  if (agent_of(local_dev, &local)) return -1;
  std::vector<uint32_t> avail(n), preferred(n, 0);
  for (int i = 0; i < n; ++i) {
    if (agent_of(peer_devs[i], &peer)) return -2;
    if (hsa_amd_memory_copy_engine_status(local, peer, &avail[i]) != HSA_STATUS_SUCCESS) return -3;
    uint32_t mask = 0;
    if (hsa_amd_memory_get_preferred_copy_engine(local, peer, &mask) == HSA_STATUS_SUCCESS) preferred[i] = mask;
  }
  uint32_t used = 0, wanted = 0;
  for (int i = 0; i < n; ++i) {
    engines[i] = lowest(preferred[i] & avail[i]);
    if (engines[i] >= 0) wanted |= 1u << engines[i];
  }
  for (int i = 0; i < n; ++i) {
    uint32_t bit = engines[i] >= 0 ? 1u << engines[i] : 0;
    uint32_t blocked = used | (mode == 1 ? pair_of(used) : 0);
    if (bit && !(blocked & bit)) {
      used |= bit;
      continue;
    }
    engines[i] = -1;
  }
  for (int i = 0; i < n; ++i) {
    if (engines[i] >= 0) continue;
    int e = -1;
    if (mode == 1) e = lowest(avail[i] & ~used & ~pair_of(used) & ~wanted & ~pair_of(wanted));
    if (e < 0) e = lowest(avail[i] & ~used & ~wanted);
    if (e < 0) e = lowest(avail[i] & ~used);
    if (e < 0) return -4;
    engines[i] = e;
    used |= 1u << e;
  }
  return 0;
}

// Arms `slot` for a call of `n` copies: waits for the copies of its previous use to finish
// and sets the trigger and the `n` copies' signals to 1. Call before sag_release and sag_issue.
int sag_arm(int slot, int n) {
  Slot& s = g_slots[slot];
  if (n > (int)s.data.size()) return -1;
  for (int i = 0; i < s.armed; ++i) {
    if (!wait_done(s.data[i])) return -1000 - i;  // copy i's data never landed
    if (!wait_done(s.flag[i])) return -2000 - i;  // copy i's flag never landed
  }
  hsa_signal_store_relaxed(s.trigger, 1);
  for (int i = 0; i < n; ++i) {
    hsa_signal_store_relaxed(s.data[i], 1);
    hsa_signal_store_relaxed(s.flag[i], 1);
  }
  s.armed = n;
  return 0;
}

// Enqueues the release of `slot`'s trigger on `stream`: copies gated on it start once the
// stream reaches this point. Without it (host release), sag_open must be called instead.
int sag_release(int slot, hipStream_t stream) {
  hipLaunchKernelGGL(release_trigger, dim3(1), dim3(1), 0, stream,
                     const_cast<hsa_signal_value_t*>(g_slots[slot].trigger_value));
  return hipGetLastError() == hipSuccess ? 0 : -1;
}

// Releases `slot`'s trigger from the host, for copies issued after the host waited for the barrier.
int sag_open(int slot) {
  hsa_signal_store_screlease(g_slots[slot].trigger, 0);
  return 0;
}

// Copies i = 0..n-1 of the call, using the slot's signals first + i: `bytes` from src[i] to
// dst[i] on SDMA engine engines[i] once the slot's trigger is released, then the 4 bytes at
// `one` to flags + 4 * flag_index[i]. Copies on one engine run in the order issued.
int sag_issue(int slot, int first, const uint64_t* dst, const uint64_t* src, uint64_t bytes, const int* engines,
              const int* flag_index, int n, uint64_t flags, uint64_t one) {
  Slot& s = g_slots[slot];
  if (first + n > s.armed) return -1;
  for (int i = 0; i < n; ++i) {
    hsa_amd_sdma_engine_id_t engine = (hsa_amd_sdma_engine_id_t)(1u << engines[i]);
    hsa_signal_t& data = s.data[first + i];
    hsa_status_t st = hsa_amd_memory_async_copy_on_engine((void*)dst[i], g_local, (const void*)src[i], g_local,
                                                          bytes, 1, &s.trigger, data, engine, true);
    if (st != HSA_STATUS_SUCCESS) return -100 - (int)st;
    st = hsa_amd_memory_async_copy_on_engine((void*)(flags + 4 * (uint64_t)flag_index[i]), g_local,
                                             (const void*)one, g_local, 4, 1, &data, s.flag[first + i], engine,
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
