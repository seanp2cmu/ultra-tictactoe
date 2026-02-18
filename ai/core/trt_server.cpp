/*
 * C++ TensorRT inference server.
 * Replaces the Python _trt_server_loop with native C++ for minimal latency.
 *
 * Communication with Python parent process:
 *   - SharedMemory for input/output tensors (same as before)
 *   - Atomic int in shared memory for signalling (replaces Python Queue)
 *     cmd_flag: 0=idle, 1=infer, 2=shutdown, 3=rebuild
 *     resp_flag: 0=idle, 1=done, 2=error
 *     batch_size: int
 *
 * Build:
 *   g++ -shared -fPIC -O3 -o trt_server.so trt_server.cpp \
 *       -I/usr/include/x86_64-linux-gnu \
 *       -L/usr/lib/x86_64-linux-gnu -lnvinfer -lcudart -lpthread
 */

#include <NvInfer.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <atomic>
#include <thread>
#include <fstream>
#include <vector>

using namespace nvinfer1;

// ── Logger ──
class TrtLogger : public ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING)
            fprintf(stderr, "[TRT-C++] %s\n", msg);
    }
};

// ── Server state ──
struct ServerState {
    // TRT
    IRuntime* runtime = nullptr;
    ICudaEngine* engine = nullptr;
    IExecutionContext* context = nullptr;
    TrtLogger logger;

    // CUDA
    cudaStream_t stream = nullptr;
    void* d_in = nullptr;
    void* d_pol = nullptr;
    void* d_val = nullptr;
    int alloc_bs = 0;
    int max_batch = 0;

    // Host pointers (into shared memory)
    float* h_in = nullptr;
    float* h_pol = nullptr;
    float* h_val = nullptr;

    // Signalling (in shared memory)
    volatile int* cmd_flag = nullptr;   // parent writes, server reads
    volatile int* resp_flag = nullptr;  // server writes, parent reads
    volatile int* batch_size = nullptr; // parent writes batch size

    // Rebuild info (in shared memory)
    // We use a simple char buffer for paths
    char* rebuild_buf = nullptr;  // onnx_path\0engine_path\0
};

static ServerState g;

static bool load_engine(const char* path) {
    std::ifstream f(path, std::ios::binary);
    if (!f.good()) return false;
    f.seekg(0, std::ios::end);
    size_t sz = f.tellg();
    f.seekg(0, std::ios::beg);
    std::vector<char> buf(sz);
    f.read(buf.data(), sz);
    f.close();

    if (g.runtime) delete g.runtime;
    g.runtime = createInferRuntime(g.logger);
    if (!g.runtime) return false;
    g.engine = g.runtime->deserializeCudaEngine(buf.data(), sz);
    if (!g.engine) return false;
    g.context = g.engine->createExecutionContext();
    return g.context != nullptr;
}

static bool build_from_onnx(const char* onnx_path, const char* engine_path,
                             int max_bs, bool fp16) {
    // We don't link nvonnxparser to keep deps minimal.
    // Instead, Python builds the engine and we just load it.
    // This function is a placeholder — rebuild is done by Python.
    (void)onnx_path; (void)engine_path; (void)max_bs; (void)fp16;
    return false;
}

static void ensure_dev(int bs) {
    if (bs <= g.alloc_bs) return;
    if (g.d_in)  cudaFree(g.d_in);
    if (g.d_pol) cudaFree(g.d_pol);
    if (g.d_val) cudaFree(g.d_val);
    g.alloc_bs = (bs > g.max_batch) ? bs : g.max_batch;
    cudaMalloc(&g.d_in,  (size_t)g.alloc_bs * 7 * 9 * 9 * sizeof(float));
    cudaMalloc(&g.d_pol, (size_t)g.alloc_bs * 81 * sizeof(float));
    cudaMalloc(&g.d_val, (size_t)g.alloc_bs * 1  * sizeof(float));

    // Set tensor addresses once
    if (g.context) {
        g.context->setTensorAddress("input",  g.d_in);
        g.context->setTensorAddress("policy", g.d_pol);
        g.context->setTensorAddress("value",  g.d_val);
    }
}

static void do_infer(int bs) {
    ensure_dev(bs);
    size_t in_bytes  = (size_t)bs * 7 * 9 * 9 * sizeof(float);
    size_t pol_bytes = (size_t)bs * 81 * sizeof(float);
    size_t val_bytes = (size_t)bs * 1  * sizeof(float);

    // H2D
    cudaMemcpyAsync(g.d_in, g.h_in, in_bytes,
                    cudaMemcpyHostToDevice, g.stream);
    // Execute
    g.context->setInputShape("input", Dims4{bs, 7, 9, 9});
    g.context->enqueueV3(g.stream);
    // D2H
    cudaMemcpyAsync(g.h_pol, g.d_pol, pol_bytes,
                    cudaMemcpyDeviceToHost, g.stream);
    cudaMemcpyAsync(g.h_val, g.d_val, val_bytes,
                    cudaMemcpyDeviceToHost, g.stream);
    cudaStreamSynchronize(g.stream);
}

// ── Exported C functions called from Python ──
extern "C" {

/*
 * Start the C++ server loop. Called from Python in a spawned process.
 * Uses spin-wait on atomic flags in shared memory for minimal latency.
 *
 * shm_ctrl: shared memory for control (cmd_flag, resp_flag, batch_size, rebuild paths)
 *   layout: [cmd_flag:4][resp_flag:4][batch_size:4][rebuild_buf:512]
 * shm_in/pol/val: shared memory for tensors (same as Python version)
 * engine_path: path to .trt file
 * max_batch: max batch size
 *
 * Returns 0 on success, -1 on failure.
 */
int trt_server_run(
    void* shm_ctrl_ptr, size_t ctrl_size,
    void* shm_in_ptr, void* shm_pol_ptr, void* shm_val_ptr,
    const char* engine_path,
    int max_batch
) {
    // Map control region
    g.cmd_flag   = (volatile int*)shm_ctrl_ptr;
    g.resp_flag  = (volatile int*)((char*)shm_ctrl_ptr + 4);
    g.batch_size = (volatile int*)((char*)shm_ctrl_ptr + 8);
    g.rebuild_buf = (char*)shm_ctrl_ptr + 12;

    // Map tensor buffers
    g.h_in  = (float*)shm_in_ptr;
    g.h_pol = (float*)shm_pol_ptr;
    g.h_val = (float*)shm_val_ptr;
    g.max_batch = max_batch;

    // Init CUDA
    cudaSetDevice(0);
    cudaStreamCreate(&g.stream);

    // Load engine
    bool ok = load_engine(engine_path);
    if (ok) {
        ensure_dev(max_batch);
        fprintf(stderr, "[TRT-C++] loaded %s\n", engine_path);
    } else {
        fprintf(stderr, "[TRT-C++] failed to load %s\n", engine_path);
    }

    // Signal ready
    *g.resp_flag = ok ? 1 : 2;  // 1=ready, 2=error
    __sync_synchronize();

    if (!ok) return -1;

    // ── Main loop: spin-wait on cmd_flag ──
    while (true) {
        // Spin-wait with backoff
        while (__sync_val_compare_and_swap((int*)g.cmd_flag, 0, 0) == 0) {
            // Yield to avoid burning CPU when idle
            std::this_thread::yield();
        }

        int cmd = *g.cmd_flag;
        __sync_synchronize();

        if (cmd == 2) {
            // Shutdown
            break;
        }
        else if (cmd == 1) {
            // Infer
            int bs = *g.batch_size;
            do_infer(bs);
            *g.resp_flag = 1;  // done
            __sync_synchronize();
            *g.cmd_flag = 0;   // reset
            __sync_synchronize();
        }
        else if (cmd == 3) {
            // Rebuild: engine_path is in rebuild_buf
            const char* new_engine_path = g.rebuild_buf;
            if (g.context) { delete g.context; g.context = nullptr; }
            if (g.engine)  { delete g.engine;  g.engine = nullptr; }
            ok = load_engine(new_engine_path);
            if (ok) {
                g.alloc_bs = 0;  // force realloc
                ensure_dev(max_batch);
            }
            *g.resp_flag = ok ? 1 : 2;
            __sync_synchronize();
            *g.cmd_flag = 0;
            __sync_synchronize();
        }
    }

    // Cleanup
    if (g.d_in)  cudaFree(g.d_in);
    if (g.d_pol) cudaFree(g.d_pol);
    if (g.d_val) cudaFree(g.d_val);
    if (g.stream) cudaStreamDestroy(g.stream);
    if (g.context) delete g.context;
    if (g.engine)  delete g.engine;
    if (g.runtime) delete g.runtime;
    fprintf(stderr, "[TRT-C++] shutdown\n");
    return 0;
}

} // extern "C"
