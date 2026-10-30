// ROCr SDMA transport for xfuser::tp_sdma_all_gather_linear (see sdma_all_gather.py), loaded with ctypes.
//
// Every copy is issued by one rank on its own SDMA engines and names the local GPU on both
// sides: peers' symmetric-memory buffers are mapped in this process for the local GPU only.
// Shards move in pieces queued in order on one engine per peer, all peers at once.
//
// pull (sag_issue): this rank reads piece c of every peer. Once all have landed, a 4-byte copy
//   on a spare engine, dependent on all of them, sets piece c's flag; the spare engine keeps
//   that wait from holding up the peers' later pieces.
// push (sag_issue_push): this rank writes its piece c into every peer's inbox; once each lands,
//   a 4-byte write on the spare engine sets this rank's own flag word for piece c at that peer
//   (one word per sender, so no two engines write the same word). On the data engines, a flag
//   write that waits for its piece would stall the next piece behind it.
// The compute stream waits for piece c's flags with sag_wait_flags, a one-thread kernel whose
// wait is bounded: a flag that never lands is recorded (sag_take_error) and the stream moves
// on, instead of hanging the GPU. Every data copy also waits for a trigger signal that a
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

// A bounded flag wait that ran out of time, written by sag_wait_flags into host-visible memory.
struct SagError {
  unsigned long long count;    // waits that timed out since the last sag_take_error
  unsigned long long missing;  // first one: bit i set if flag word i never read 1
  int piece;                   // first one: piece and call id it waited for
  int call;
};

namespace {

struct Gpu {
  hsa_agent_t agent;
  uint32_t bus;
};

constexpr int kTriggers = 16;  // per slot: one per piece at most (the all-gather uses only the first)

struct Slot {
  hsa_signal_t trigger[kTriggers];
  volatile hsa_signal_value_t* trigger_value[kTriggers];
  std::vector<hsa_signal_t> data;  // one per data copy of a call
  std::vector<hsa_signal_t> flag;  // one per piece
  int armed_data = 0;              // what the last call on this slot used
  int armed_flags = 0;
};

std::vector<Gpu> g_gpus;
std::vector<Slot> g_slots;
hsa_agent_t g_local;
SagError* g_error = nullptr;       // host-mapped
int* g_scratch = nullptr;          // device word that a dropped flag write goes to
int g_drop_piece = -1, g_drop_index = -1;
unsigned long long g_ticks_per_s = 0;

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

__global__ void release_trigger(hsa_signal_value_t* value) {
  __hip_atomic_store(value, (hsa_signal_value_t)0, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
}

// Waits until all `count` words at `words` read 1 (acquire, system scope), or `budget` wall-clock
// ticks have passed; then records which words were missing and returns.
__global__ void wait_flags(const int* words, int count, unsigned long long budget, SagError* err, int piece,
                           int call) {
  unsigned long long start = wall_clock64();
  while (true) {
    unsigned long long missing = 0;
    for (int i = 0; i < count; ++i)
      if (__hip_atomic_load(words + i, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM) != 1) missing |= 1ull << i;
    if (!missing) return;
    if (wall_clock64() - start > budget) {
      if (__hip_atomic_fetch_add(&err->count, 1ull, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM) == 0) {
        err->missing = missing;
        err->piece = piece;
        err->call = call;
      }
      __threadfence_system();
      return;
    }
    __builtin_amdgcn_s_sleep(2);
  }
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
  if (hipHostMalloc((void**)&g_error, sizeof(SagError), hipHostMallocMapped | hipHostMallocCoherent) != hipSuccess)
    return -8;
  *g_error = SagError{};
  if (hipMalloc((void**)&g_scratch, 64) != hipSuccess) return -9;
  int khz = 0;
  if (hipDeviceGetAttribute(&khz, hipDeviceAttributeWallClockRate, local_dev) != hipSuccess || khz <= 0) return -10;
  g_ticks_per_s = (unsigned long long)khz * 1000ull;
  g_slots.resize(slots);
  for (Slot& s : g_slots) {
    // Triggers are written by a kernel, so their values must be addressable: GPU-only signals.
    for (int k = 0; k < kTriggers; ++k) {
      if (hsa_amd_signal_create(0, 0, nullptr, HSA_AMD_SIGNAL_AMD_GPU_ONLY, &s.trigger[k]) != HSA_STATUS_SUCCESS)
        return -4;
      if (hsa_amd_signal_value_pointer(s.trigger[k], &s.trigger_value[k]) != HSA_STATUS_SUCCESS) return -5;
    }
    s.data.resize(max_copies);
    s.flag.resize(max_copies);
    for (int i = 0; i < max_copies; ++i)
      if (hsa_signal_create(0, 0, nullptr, &s.data[i]) != HSA_STATUS_SUCCESS ||
          hsa_signal_create(0, 0, nullptr, &s.flag[i]) != HSA_STATUS_SUCCESS)
        return -6;
  }
  return 0;
}

// One SDMA engine (bit index) per peer, in engines[0..n-1]: ROCr's preferred engine for the
// link the data crosses (into `local_dev` from the peer when pulling, from `local_dev` to the
// peer when pushing); a peer whose preferred engine is taken gets the lowest available engine
// no peer prefers (then any free one). engines[n] is a spare engine, used by no peer, for the
// pull flag copies.
int sag_engine_plan(int local_dev, const int* peer_devs, int n, int push, int* engines) {
  hsa_agent_t local, peer;
  if (agent_of(local_dev, &local)) return -1;
  std::vector<uint32_t> avail(n), preferred(n, 0);
  for (int i = 0; i < n; ++i) {
    if (agent_of(peer_devs[i], &peer)) return -2;
    hsa_agent_t dst = push ? peer : local, src = push ? local : peer;
    if (hsa_amd_memory_copy_engine_status(dst, src, &avail[i]) != HSA_STATUS_SUCCESS) return -3;
    uint32_t mask = 0;
    if (hsa_amd_memory_get_preferred_copy_engine(dst, src, &mask) == HSA_STATUS_SUCCESS) preferred[i] = mask;
  }
  uint32_t used = 0, wanted = 0, everywhere = ~0u;
  for (int i = 0; i < n; ++i) {
    engines[i] = lowest(preferred[i] & avail[i]);
    if (engines[i] >= 0) wanted |= 1u << engines[i];
    everywhere &= avail[i];
  }
  for (int i = 0; i < n; ++i) {
    uint32_t bit = engines[i] >= 0 ? 1u << engines[i] : 0;
    if (bit && !(used & bit)) {
      used |= bit;
      continue;
    }
    engines[i] = -1;
  }
  for (int i = 0; i < n; ++i) {
    if (engines[i] >= 0) continue;
    int e = lowest(avail[i] & ~used & ~wanted);
    if (e < 0) e = lowest(avail[i] & ~used);
    if (e < 0) return -4;
    engines[i] = e;
    used |= 1u << e;
  }
  // The flag copies name the local GPU on both sides, like the data copies; any engine that
  // every link offers will do, preferably one no peer uses.
  engines[n] = lowest(everywhere & ~used);
  if (engines[n] < 0) engines[n] = lowest(everywhere);
  return engines[n] < 0 ? -5 : 0;
}

// Arms `slot` for a call of `n_data` data copies and `n_flags` flag copies: waits for the
// copies of its previous use to finish and sets its triggers and the signals the call will use
// to 1. Call before sag_release and sag_issue.
int sag_arm(int slot, int n_data, int n_flags) {
  Slot& s = g_slots[slot];
  if (n_data > (int)s.data.size() || n_flags > (int)s.flag.size()) return -1;
  for (int i = 0; i < s.armed_data; ++i)
    if (!wait_done(s.data[i])) return -1000 - i;  // data copy i never landed
  for (int i = 0; i < s.armed_flags; ++i)
    if (!wait_done(s.flag[i])) return -2000 - i;  // piece i's flag never landed
  for (int k = 0; k < kTriggers; ++k) hsa_signal_store_relaxed(s.trigger[k], 1);
  for (int i = 0; i < n_data; ++i) hsa_signal_store_relaxed(s.data[i], 1);
  for (int i = 0; i < n_flags; ++i) hsa_signal_store_relaxed(s.flag[i], 1);
  s.armed_data = n_data;
  s.armed_flags = n_flags;
  return 0;
}

// Enqueues the release of `slot`'s trigger `k` on `stream`: copies gated on it start once the
// stream reaches this point. Without it (host release), sag_open must be called instead.
int sag_release(int slot, int k, hipStream_t stream) {
  if (k < 0 || k >= kTriggers) return -2;
  hipLaunchKernelGGL(release_trigger, dim3(1), dim3(1), 0, stream,
                     const_cast<hsa_signal_value_t*>(g_slots[slot].trigger_value[k]));
  return hipGetLastError() == hipSuccess ? 0 : -1;
}

// Releases `slot`'s trigger `k` from the host, for copies issued after the host waited for the
// work they depend on.
int sag_open(int slot, int k) {
  if (k < 0 || k >= kTriggers) return -2;
  hsa_signal_store_screlease(g_slots[slot].trigger[k], 0);
  return 0;
}

// Pull, piece `piece` of the call: n data copies, using the slot's data signals piece * n + i,
// each `bytes` from src[i] to dst[i] on SDMA engine engines[i] once the slot's trigger k is released
// (copies on one engine run in the order issued, so piece 0's copies must be issued first);
// then, on engine engines[n] and once all n have landed, the 4 bytes at `one` to `flag`.
int sag_issue(int slot, int piece, int k, const uint64_t* dst, const uint64_t* src, uint64_t bytes,
              const int* engines, int n, uint64_t flag, uint64_t one) {
  Slot& s = g_slots[slot];
  int first = piece * n;
  if (first + n > s.armed_data || piece >= s.armed_flags || k < 0 || k >= kTriggers) return -1;
  // Every piece waits for the trigger itself, rather than relying on queueing behind piece 0.
  for (int i = 0; i < n; ++i) {
    hsa_amd_sdma_engine_id_t engine = (hsa_amd_sdma_engine_id_t)(1u << engines[i]);
    hsa_status_t st = hsa_amd_memory_async_copy_on_engine((void*)dst[i], g_local, (const void*)src[i], g_local,
                                                          bytes, 1, &s.trigger[k], s.data[first + i], engine, true);
    if (st != HSA_STATUS_SUCCESS) return -100 - (int)st;
  }
  hsa_status_t st = hsa_amd_memory_async_copy_on_engine(
      (void*)flag, g_local, (const void*)one, g_local, 4, n, &s.data[first], s.flag[piece],
      (hsa_amd_sdma_engine_id_t)(1u << engines[n]), true);
  return st == HSA_STATUS_SUCCESS ? 0 : -200 - (int)st;
}

// Push, piece `piece` of the call: n data copies, using the slot's data and flag signals
// piece * n + i, each `bytes` from src[i] to dst[i] on SDMA engine engines[i] once the slot's
// trigger k is released; then, on the spare engine engines[n] and once copy i has landed, the
// 4 bytes at `one` to flag_dst[i] (4-byte aligned).
int sag_issue_push(int slot, int piece, int k, const uint64_t* dst, const uint64_t* src, uint64_t bytes,
                   const int* engines, int n, const uint64_t* flag_dst, uint64_t one) {
  Slot& s = g_slots[slot];
  int first = piece * n;
  if (first + n > s.armed_data || first + n > s.armed_flags || k < 0 || k >= kTriggers) return -1;
  // Every piece waits for the trigger itself, rather than relying on queueing behind piece 0.
  for (int i = 0; i < n; ++i) {
    hsa_amd_sdma_engine_id_t engine = (hsa_amd_sdma_engine_id_t)(1u << engines[i]);
    hsa_signal_t& data = s.data[first + i];
    hsa_status_t st = hsa_amd_memory_async_copy_on_engine((void*)dst[i], g_local, (const void*)src[i], g_local,
                                                          bytes, 1, &s.trigger[k], data, engine, true);
    if (st != HSA_STATUS_SUCCESS) return -100 - (int)st;
  }
  hsa_amd_sdma_engine_id_t spare = (hsa_amd_sdma_engine_id_t)(1u << engines[n]);
  for (int i = 0; i < n; ++i) {
    void* flag = (void*)flag_dst[i];
    if (piece == g_drop_piece && i == g_drop_index) {  // test hook: this flag goes nowhere
      flag = g_scratch;
      g_drop_piece = g_drop_index = -1;
    }
    hsa_status_t st = hsa_amd_memory_async_copy_on_engine(flag, g_local, (const void*)one, g_local, 4, 1,
                                                          &s.data[first + i], s.flag[first + i], spare, true);
    if (st != HSA_STATUS_SUCCESS) return -200 - (int)st;
  }
  return 0;
}

// Enqueues on `stream` a wait until the `count` 32-bit flags at `words` all read 1, bounded by
// `seconds`; a wait that runs out is recorded for sag_take_error and the stream moves on.
int sag_wait_flags(hipStream_t stream, uint64_t words, int count, double seconds, int piece, int call) {
  if (count > 64) return -1;
  unsigned long long budget = (unsigned long long)(seconds * (double)g_ticks_per_s);
  hipLaunchKernelGGL(wait_flags, dim3(1), dim3(1), 0, stream, (const int*)words, count, budget, g_error, piece, call);
  return hipGetLastError() == hipSuccess ? 0 : -2;
}

// Returns how many bounded waits ran out since the last call (0 if none) and the first one's
// piece, call id and missing-flag mask; clears the record.
int sag_take_error(int* piece, int* call, uint64_t* missing) {
  volatile SagError* e = g_error;
  unsigned long long count = e->count;
  if (count == 0) return 0;
  *piece = e->piece;
  *call = e->call;
  *missing = e->missing;
  e->count = 0;
  return (int)(count > 0x7fffffff ? 0x7fffffff : count);
}

// Test hook: the next push of piece `piece` diverts its flag write for peer index `index` to a
// scratch word, as if it were lost.
int sag_debug_drop_flag(int piece, int index) {
  g_drop_piece = piece;
  g_drop_index = index;
  return 0;
}

}
