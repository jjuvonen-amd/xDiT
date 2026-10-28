// Peer-to-peer reads on AMD GPUs with explicit SDMA engine choice: does pulling from every
// peer at once scale when each peer's copy gets its own copy engine?
//
//   hipcc -O2 -o /tmp/sdma_p2p benchmarks/wan_tp/sdma_p2p_bandwidth.cpp -lhsa-runtime64
//   /tmp/sdma_p2p [--mib 92] [--iters 20] [--warmup 3]
//
// One process drives every visible GPU. Each GPU d holds a source buffer filled with byte
// d + 1 and pulls the buffers of its peers into a local destination (dst on d, src on the
// peer), with copies issued through ROCr's async copy API:
//   default    hsa_amd_memory_async_copy: the runtime picks the engine
//   preferred  hsa_amd_memory_async_copy_on_engine on the lowest engine of
//              hsa_amd_memory_get_preferred_copy_engine (or of the available mask)
//   distinct   ..._on_engine with a different engine per peer: pair (d, p) takes engine
//              slot (p - d - 1) mod n of its candidate list, so engine slots differ both
//              across one receiver's peers and across one sender's receivers
// for one peer, and for all peers at once with every GPU receiving (an all-gather's
// traffic) or with GPU 0 receiving alone. GB/s = bytes received per receiving GPU / wall
// time (median). Every mode checks what it received.
#include <hip/hip_runtime.h>
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define HIP_CHECK(x)                                                                         \
  do {                                                                                       \
    hipError_t e_ = (x);                                                                     \
    if (e_ != hipSuccess) {                                                                  \
      fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, hipGetErrorString(e_));      \
      exit(1);                                                                               \
    }                                                                                        \
  } while (0)
#define HSA_CHECK(x)                                                                         \
  do {                                                                                       \
    hsa_status_t s_ = (x);                                                                   \
    if (s_ != HSA_STATUS_SUCCESS) {                                                          \
      const char* m_ = nullptr;                                                              \
      hsa_status_string(s_, &m_);                                                            \
      fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, m_ ? m_ : "?");              \
      exit(1);                                                                               \
    }                                                                                        \
  } while (0)

struct Gpu {
  hsa_agent_t agent{};
  uint32_t bus = 0;
  uint32_t sdma = 0, sdma_xgmi = 0;
  char* src = nullptr;
  char* dst = nullptr;  // (n - 1) slots, slot i holds peer peers[i]
};

static std::vector<hsa_agent_t> g_agents;

static hsa_status_t collect_gpu(hsa_agent_t agent, void*) {
  hsa_device_type_t type;
  hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &type);
  if (type == HSA_DEVICE_TYPE_GPU) g_agents.push_back(agent);
  return HSA_STATUS_SUCCESS;
}

static std::vector<int> bits(uint32_t mask) {
  std::vector<int> out;
  for (int b = 0; b < 32; ++b)
    if (mask & (1u << b)) out.push_back(b);
  return out;
}

static std::string mask_str(uint32_t mask) {
  std::string s;
  for (int b : bits(mask)) s += (s.empty() ? "" : ",") + std::to_string(b);
  return s.empty() ? "-" : s;
}

int main(int argc, char** argv) {
  size_t mib = 92;
  int iters = 20, warmup = 3;
  for (int i = 1; i + 1 < argc; i += 2) {
    if (!strcmp(argv[i], "--mib")) mib = strtoull(argv[i + 1], nullptr, 10);
    else if (!strcmp(argv[i], "--iters")) iters = atoi(argv[i + 1]);
    else if (!strcmp(argv[i], "--warmup")) warmup = atoi(argv[i + 1]);
  }
  const size_t bytes = mib << 20;

  int n = 0;
  HIP_CHECK(hipGetDeviceCount(&n));
  if (n < 2) {
    fprintf(stderr, "need at least 2 GPUs\n");
    return 1;
  }
  HSA_CHECK(hsa_init());
  HSA_CHECK(hsa_iterate_agents(collect_gpu, nullptr));

  // Match HIP device order to HSA agents by PCI bus.
  std::vector<Gpu> gpu(n);
  for (int d = 0; d < n; ++d) {
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, d));
    bool found = false;
    for (auto a : g_agents) {
      uint32_t bdf = 0;
      HSA_CHECK(hsa_agent_get_info(a, (hsa_agent_info_t)HSA_AMD_AGENT_INFO_BDFID, &bdf));
      if (((bdf >> 8) & 0xff) == (uint32_t)prop.pciBusID) {
        gpu[d].agent = a;
        gpu[d].bus = prop.pciBusID;
        found = true;
      }
    }
    if (!found) {
      fprintf(stderr, "no HSA agent for HIP device %d (bus %x)\n", d, prop.pciBusID);
      return 1;
    }
    hsa_agent_get_info(gpu[d].agent, (hsa_agent_info_t)HSA_AMD_AGENT_INFO_NUM_SDMA_ENG, &gpu[d].sdma);
    hsa_agent_get_info(gpu[d].agent, (hsa_agent_info_t)HSA_AMD_AGENT_INFO_NUM_SDMA_XGMI_ENG, &gpu[d].sdma_xgmi);
  }

  for (int d = 0; d < n; ++d) {
    HIP_CHECK(hipSetDevice(d));
    for (int p = 0; p < n; ++p)
      if (p != d) {
        hipError_t e = hipDeviceEnablePeerAccess(p, 0);
        if (e != hipSuccess && e != hipErrorPeerAccessAlreadyEnabled) HIP_CHECK(e);
      }
    HIP_CHECK(hipMalloc(&gpu[d].src, bytes));
    HIP_CHECK(hipMemset(gpu[d].src, d + 1, bytes));
    HIP_CHECK(hipMalloc(&gpu[d].dst, bytes * (n - 1)));
  }
  for (int d = 0; d < n; ++d) {
    HIP_CHECK(hipSetDevice(d));
    HIP_CHECK(hipDeviceSynchronize());
  }

  auto peer_of = [&](int d, int i) { return (d + 1 + i) % n; };  // slot i of receiver d

  // Engine candidates per pair: ROCr's preferred engines, else every available one.
  std::vector<std::vector<uint32_t>> avail(n, std::vector<uint32_t>(n)), pref(n, std::vector<uint32_t>(n));
  for (int d = 0; d < n; ++d)
    for (int p = 0; p < n; ++p)
      if (p != d) {
        if (hsa_amd_memory_copy_engine_status(gpu[d].agent, gpu[p].agent, &avail[d][p]) != HSA_STATUS_SUCCESS)
          avail[d][p] = 0;
        if (hsa_amd_memory_get_preferred_copy_engine(gpu[d].agent, gpu[p].agent, &pref[d][p]) != HSA_STATUS_SUCCESS)
          pref[d][p] = 0;
      }
  auto candidates = [&](int d, int p) { return bits(pref[d][p] ? pref[d][p] : avail[d][p]); };
  auto all_avail = [&](int d, int p) { return bits(avail[d][p]); };

  // Engine per (receiver, peer) for each explicit-engine strategy; -1 = runtime default.
  auto engine_for = [&](const std::string& strategy, int d, int p) -> int {
    if (strategy == "default") return -1;
    if (strategy == "preferred") {
      auto c = candidates(d, p);
      return c.empty() ? -1 : c[0];
    }
    // distinct: Latin-square slot over the pair's candidates; widen to every available
    // engine when the preferred list is shorter than the number of peers.
    auto c = candidates(d, p);
    if ((int)c.size() < n - 1) c = all_avail(d, p);
    if (c.empty()) return -1;
    int slot = ((p - d - 1) % n + n) % n;
    return c[slot % c.size()];
  };

  struct Copy {
    int d, slot, p, engine;
  };
  auto plan = [&](const std::string& strategy, bool all_peers, bool all_receivers) {
    std::vector<Copy> copies;
    // Interleave receivers so every GPU's queues get work right away.
    for (int i = 0; i < (all_peers ? n - 1 : 1); ++i)
      for (int d = 0; d < (all_receivers ? n : 1); ++d)
        copies.push_back({d, i, peer_of(d, i), engine_for(strategy, d, peer_of(d, i))});
    return copies;
  };

  std::vector<hsa_signal_t> signals(n * n);
  for (auto& s : signals) HSA_CHECK(hsa_signal_create(1, 0, nullptr, &s));

  auto run_once = [&](const std::vector<Copy>& copies) {
    for (size_t k = 0; k < copies.size(); ++k) hsa_signal_store_relaxed(signals[k], 1);
    auto t0 = std::chrono::steady_clock::now();
    for (size_t k = 0; k < copies.size(); ++k) {
      const Copy& c = copies[k];
      void* dst = gpu[c.d].dst + (size_t)c.slot * bytes;
      const void* src = gpu[c.p].src;
      if (c.engine < 0) {
        HSA_CHECK(hsa_amd_memory_async_copy(dst, gpu[c.d].agent, src, gpu[c.p].agent, bytes, 0, nullptr,
                                            signals[k]));
      } else {
        HSA_CHECK(hsa_amd_memory_async_copy_on_engine(dst, gpu[c.d].agent, src, gpu[c.p].agent, bytes, 0, nullptr,
                                                      signals[k], (hsa_amd_sdma_engine_id_t)(1u << c.engine),
                                                      true));
      }
    }
    for (size_t k = 0; k < copies.size(); ++k)
      hsa_signal_wait_scacquire(signals[k], HSA_SIGNAL_CONDITION_LT, 1, UINT64_MAX, HSA_WAIT_STATE_ACTIVE);
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  };

  auto verify = [&](const std::vector<Copy>& copies) {
    for (const Copy& c : copies) {
      unsigned char probe[3];
      char* base = gpu[c.d].dst + (size_t)c.slot * bytes;
      HIP_CHECK(hipSetDevice(c.d));
      HIP_CHECK(hipMemcpy(&probe[0], base, 1, hipMemcpyDeviceToHost));
      HIP_CHECK(hipMemcpy(&probe[1], base + bytes / 2, 1, hipMemcpyDeviceToHost));
      HIP_CHECK(hipMemcpy(&probe[2], base + bytes - 1, 1, hipMemcpyDeviceToHost));
      for (unsigned char v : probe)
        if (v != c.p + 1) return false;
    }
    return true;
  };
  auto clear = [&]() {
    for (int d = 0; d < n; ++d) {
      HIP_CHECK(hipSetDevice(d));
      HIP_CHECK(hipMemset(gpu[d].dst, 0, bytes * (n - 1)));
      HIP_CHECK(hipDeviceSynchronize());
    }
  };

  // Report the engine landscape first.
  hipDeviceProp_t prop0;
  HIP_CHECK(hipGetDeviceProperties(&prop0, 0));
  printf("# SDMA peer-read bandwidth, %d x %s, %zu MiB per peer\n\n", n, prop0.name, mib);
  printf("SDMA engines per GPU: %u (+ %u xGMI-optimized), GPU 0.\n\n", gpu[0].sdma, gpu[0].sdma_xgmi);
  printf("Engines for GPU 0 pulling from each peer (bit = engine id):\n\n");
  printf("| peer | available | preferred | distinct pick |\n|---|---|---|---|\n");
  for (int i = 0; i < n - 1; ++i) {
    int p = peer_of(0, i);
    printf("| %d | %s | %s | %d |\n", p, mask_str(avail[0][p]).c_str(), mask_str(pref[0][p]).c_str(),
           engine_for("distinct", 0, p));
  }
  printf("\n");

  struct Mode {
    const char* name;
    bool all_peers, all_receivers;
    const char* strategy;
  };
  const Mode modes[] = {
      {"one peer, all GPUs receiving", false, true, "default"},
      {"one peer, all GPUs receiving", false, true, "preferred"},
      {"all peers at once, GPU 0 receiving alone", true, false, "default"},
      {"all peers at once, GPU 0 receiving alone", true, false, "preferred"},
      {"all peers at once, GPU 0 receiving alone", true, false, "distinct"},
      {"all peers at once, all GPUs receiving", true, true, "default"},
      {"all peers at once, all GPUs receiving", true, true, "preferred"},
      {"all peers at once, all GPUs receiving", true, true, "distinct"},
  };
  printf("| pattern | engine choice | engines used (GPU 0) | ms | GB/s per receiving GPU | data ok |\n");
  printf("|---|---|---|---|---|---|\n");
  for (const Mode& m : modes) {
    auto copies = plan(m.strategy, m.all_peers, m.all_receivers);
    std::vector<int> used;
    for (const Copy& c : copies)
      if (c.d == 0) used.push_back(c.engine);
    std::sort(used.begin(), used.end());
    std::string used_s;
    for (size_t k = 0; k < used.size(); ++k)
      if (k == 0 || used[k] != used[k - 1]) used_s += (used_s.empty() ? "" : ",") + (used[k] < 0 ? std::string("auto") : std::to_string(used[k]));

    clear();
    run_once(copies);
    bool ok = verify(copies);
    for (int w = 0; w < warmup; ++w) run_once(copies);
    std::vector<double> times;
    for (int it = 0; it < iters; ++it) times.push_back(run_once(copies));
    std::sort(times.begin(), times.end());
    double ms = times[times.size() / 2];
    int per_receiver = m.all_peers ? n - 1 : 1;
    double gbps = per_receiver * (double)bytes / (ms * 1e-3) / 1e9;
    printf("| %s | %s | %s | %.2f | %.0f | %s |\n", m.name, m.strategy, used_s.c_str(), ms, gbps, ok ? "yes" : "NO");
    fflush(stdout);
  }

  for (auto& s : signals) hsa_signal_destroy(s);
  for (int d = 0; d < n; ++d) {
    HIP_CHECK(hipSetDevice(d));
    HIP_CHECK(hipFree(gpu[d].src));
    HIP_CHECK(hipFree(gpu[d].dst));
  }
  return 0;
}
