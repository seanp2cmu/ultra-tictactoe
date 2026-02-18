"""TensorRT inference server in a separate process.

Avoids CUDA context conflicts with PyTorch by running TensorRT in a
dedicated spawned process.  Communication uses SharedMemory (zero-copy
numpy arrays) + multiprocessing Queues for signalling.
"""
import os
import numpy as np
from typing import Tuple, Optional
from multiprocessing import get_context
from multiprocessing.shared_memory import SharedMemory

TRT_AVAILABLE = False
trt = None

try:
    import tensorrt as trt
    TRT_AVAILABLE = True
except ImportError:
    pass

# 'spawn' gives the child a fresh CUDA context (no inherited PyTorch state)
_mp = get_context('spawn')


# ──────────────────────────────────────────────────────────────────────────────
# Server process  (NO PyTorch – owns its own CUDA context)
# ──────────────────────────────────────────────────────────────────────────────

def _trt_server_loop(
    req_q, resp_q,
    shm_in_name, shm_pol_name, shm_val_name,
    max_batch, engine_path, onnx_path, fp16,
):
    """Main loop of the TensorRT inference server (runs in child process)."""
    import ctypes
    import tensorrt as _trt

    # ── tiny CUDA runtime wrapper via ctypes (no PyTorch needed) ──
    _rt = None
    for lib in ('libcudart.so', 'libcudart.so.12', 'libcudart.so.11.0'):
        try:
            _rt = ctypes.CDLL(lib)
            break
        except OSError:
            continue
    if _rt is None:
        resp_q.put(('ready', False))
        return

    HTOD, DTOH = 1, 2

    def _ck(ret, tag=''):
        if ret != 0:
            raise RuntimeError(f'CUDA {tag} error {ret}')

    def cu_malloc(n):
        p = ctypes.c_void_p()
        _ck(_rt.cudaMalloc(ctypes.byref(p), ctypes.c_size_t(n)), 'malloc')
        return p

    def cu_free(p):
        if p:
            _rt.cudaFree(p)

    def cu_h2d(d, h_ptr, n):
        _ck(_rt.cudaMemcpy(d, h_ptr, ctypes.c_size_t(n), ctypes.c_int(HTOD)), 'h2d')

    def cu_d2h(h_ptr, d, n):
        _ck(_rt.cudaMemcpy(h_ptr, d, ctypes.c_size_t(n), ctypes.c_int(DTOH)), 'd2h')

    def cu_h2d_async(d, h_ptr, n, s):
        _ck(_rt.cudaMemcpyAsync(d, h_ptr, ctypes.c_size_t(n), ctypes.c_int(HTOD), s), 'h2d_a')

    def cu_d2h_async(h_ptr, d, n, s):
        _ck(_rt.cudaMemcpyAsync(h_ptr, d, ctypes.c_size_t(n), ctypes.c_int(DTOH), s), 'd2h_a')

    _ck(_rt.cudaSetDevice(ctypes.c_int(0)), 'setdev')
    stream = ctypes.c_void_p()
    _ck(_rt.cudaStreamCreate(ctypes.byref(stream)), 'stream')

    # ── open shared-memory numpy views ──
    shm_in  = SharedMemory(name=shm_in_name,  create=False)
    shm_pol = SharedMemory(name=shm_pol_name, create=False)
    shm_val = SharedMemory(name=shm_val_name, create=False)

    buf_in  = np.ndarray((max_batch, 7, 9, 9), dtype=np.float32, buffer=shm_in.buf)
    buf_pol = np.ndarray((max_batch, 81),       dtype=np.float32, buffer=shm_pol.buf)
    buf_val = np.ndarray((max_batch, 1),        dtype=np.float32, buffer=shm_val.buf)

    # Stable host pointers for cudaMemcpy (point into shared memory)
    h_in  = ctypes.c_void_p(buf_in.ctypes.data)
    h_pol = ctypes.c_void_p(buf_pol.ctypes.data)
    h_val = ctypes.c_void_p(buf_val.ctypes.data)

    # ── TensorRT state ──
    logger  = _trt.Logger(_trt.Logger.WARNING)
    engine  = None
    context = None
    d_in = d_pol = d_val = None
    alloc_bs = 0
    _addrs_set = False

    def _build_from_onnx(ox, ep, mbs, f16):
        nonlocal engine, context
        if not os.path.exists(ox):
            print(f'[TRT-Srv] ONNX not found: {ox}')
            return False
        builder = _trt.Builder(logger)
        net = builder.create_network(
            1 << int(_trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        )
        parser = _trt.OnnxParser(net, logger)
        with open(ox, 'rb') as f:
            if not parser.parse(f.read()):
                for i in range(parser.num_errors):
                    print(f'[TRT-Srv] parse err: {parser.get_error(i)}')
                return False
        cfg = builder.create_builder_config()
        cfg.set_memory_pool_limit(_trt.MemoryPoolType.WORKSPACE, 1 << 32)
        if f16 and builder.platform_has_fast_fp16:
            cfg.set_flag(_trt.BuilderFlag.FP16)
            # print('[TRT-Srv] FP16 enabled')
        prof = builder.create_optimization_profile()
        opt_bs = min(128, mbs)
        prof.set_shape(
            'input', (1, 7, 9, 9), (opt_bs, 7, 9, 9), (mbs, 7, 9, 9)
        )
        cfg.add_optimization_profile(prof)
        # print(f'[TRT-Srv] Building engine (max_batch={mbs}) ...')
        blob = builder.build_serialized_network(net, cfg)
        if blob is None:
            print('[TRT-Srv] build failed')
            return False
        os.makedirs(os.path.dirname(ep) or '.', exist_ok=True)
        with open(ep, 'wb') as f:
            f.write(blob)
        # print(f'[TRT-Srv] saved {ep}')
        rt = _trt.Runtime(logger)
        engine = rt.deserialize_cuda_engine(blob)
        context = engine.create_execution_context()
        return True

    def _load_engine(ep):
        nonlocal engine, context
        if not os.path.exists(ep):
            return False
        rt = _trt.Runtime(logger)
        with open(ep, 'rb') as f:
            engine = rt.deserialize_cuda_engine(f.read())
        if engine is None:
            return False
        context = engine.create_execution_context()
        print(f'[TRT-Srv] loaded {ep}')
        return True

    def _ensure_dev(bs):
        nonlocal d_in, d_pol, d_val, alloc_bs, _addrs_set
        if bs <= alloc_bs:
            return
        cu_free(d_in)
        cu_free(d_pol)
        cu_free(d_val)
        alloc_bs = max(bs, max_batch)
        d_in  = cu_malloc(alloc_bs * 7 * 9 * 9 * 4)
        d_pol = cu_malloc(alloc_bs * 81 * 4)
        d_val = cu_malloc(alloc_bs * 1  * 4)
        _addrs_set = False  # addresses changed, need to re-set

    # ── initial load / build ──
    ok = _load_engine(engine_path) if os.path.exists(engine_path) else False
    if not ok and os.path.exists(onnx_path):
        ok = _build_from_onnx(onnx_path, engine_path, max_batch, fp16)
    if ok:
        _ensure_dev(max_batch)
    resp_q.put(('ready', ok))

    # ── event loop ──
    while True:
        try:
            msg = req_q.get()
        except Exception:
            break

        cmd = msg[0]

        if cmd == 'shutdown':
            break

        elif cmd == 'infer':
            bs = msg[1]
            if engine is None:
                resp_q.put(('error', 'no engine'))
                continue
            try:
                _ensure_dev(bs)

                if not _addrs_set:
                    context.set_tensor_address('input',  d_in.value)
                    context.set_tensor_address('policy', d_pol.value)
                    context.set_tensor_address('value',  d_val.value)
                    _addrs_set = True

                ib = bs * 7 * 9 * 9 * 4
                pb = bs * 81 * 4
                vb = bs * 1  * 4
                cu_h2d_async(d_in, h_in, ib, stream.value)
                context.set_input_shape('input', (bs, 7, 9, 9))
                context.execute_async_v3(stream.value)
                cu_d2h_async(h_pol, d_pol, pb, stream.value)
                cu_d2h_async(h_val, d_val, vb, stream.value)
                _rt.cudaStreamSynchronize(stream)

                resp_q.put(('done',))
            except Exception as e:
                resp_q.put(('error', str(e)))

        elif cmd == 'rebuild':
            _, ox, ep, mbs, f16 = msg
            context = None
            engine = None
            _addrs_set = False
            try:
                ok = _build_from_onnx(ox, ep, mbs, f16)
                if ok:
                    _ensure_dev(mbs)
                resp_q.put(('rebuilt', ok))
            except Exception as e:
                resp_q.put(('error', str(e)))

    # ── cleanup ──
    cu_free(d_in)
    cu_free(d_pol)
    cu_free(d_val)
    _rt.cudaStreamDestroy(stream)
    shm_in.close()
    shm_pol.close()
    shm_val.close()
    print('[TRT-Srv] shutdown')


# ──────────────────────────────────────────────────────────────────────────────
# Client  (runs in the main PyTorch process)
# ──────────────────────────────────────────────────────────────────────────────

class TensorRTProcessClient:
    """Communicates with a dedicated TensorRT server process via SharedMemory."""

    def __init__(self, max_batch_size: int = 4096):
        self.max_batch_size = max_batch_size
        self._ready = False
        self._proc = None
        self._req_q = None
        self._resp_q = None
        self._shms: list = []
        self._input_buf = None
        self._policy_buf = None
        self._value_buf = None

    # ── lifecycle ──

    def start(
        self,
        engine_path: str = './model/model.trt',
        onnx_path: str = './model/model.onnx',
        fp16: bool = True,
    ) -> bool:
        """Start the TensorRT server process."""
        if self._proc is not None and self._proc.is_alive():
            return self._ready

        pid = os.getpid()
        names = [f'trt_in_{pid}', f'trt_pol_{pid}', f'trt_val_{pid}']
        sizes = [
            self.max_batch_size * 7 * 9 * 9 * 4,
            self.max_batch_size * 81 * 4,
            self.max_batch_size * 1  * 4,
        ]

        # Clean any stale shared memory from previous runs
        for n in names:
            try:
                s = SharedMemory(name=n, create=False)
                s.close()
                s.unlink()
            except FileNotFoundError:
                pass

        shms = [SharedMemory(name=n, create=True, size=sz)
                for n, sz in zip(names, sizes)]
        self._shms = shms

        self._input_buf  = np.ndarray(
            (self.max_batch_size, 7, 9, 9), np.float32, buffer=shms[0].buf)
        self._policy_buf = np.ndarray(
            (self.max_batch_size, 81),       np.float32, buffer=shms[1].buf)
        self._value_buf  = np.ndarray(
            (self.max_batch_size, 1),        np.float32, buffer=shms[2].buf)

        self._req_q  = _mp.Queue()
        self._resp_q = _mp.Queue()

        self._proc = _mp.Process(
            target=_trt_server_loop,
            args=(
                self._req_q, self._resp_q,
                names[0], names[1], names[2],
                self.max_batch_size, engine_path, onnx_path, fp16,
            ),
            daemon=True,
        )
        self._proc.start()

        # Wait for server initialisation (may build engine – up to 10 min)
        status, ok = self._resp_q.get(timeout=600)
        self._ready = (status == 'ready' and ok)
        print(f'[TRT-Client] server {"ready" if self._ready else "FAILED"}')
        return self._ready

    def shutdown(self):
        """Stop the server process and release shared memory."""
        if self._proc is not None and self._proc.is_alive():
            try:
                self._req_q.put(('shutdown',))
                self._proc.join(timeout=5)
            except Exception:
                self._proc.kill()
        for s in self._shms:
            try:
                s.close()
                s.unlink()
            except Exception:
                pass
        self._shms.clear()
        self._proc = None
        self._ready = False

    def __del__(self):
        self.shutdown()

    # ── inference ──

    def infer(self, input_array: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Run inference via the server process (zero-copy shared memory)."""
        bs = self.infer_async(input_array)
        return self.infer_wait(bs)

    def infer_async(self, input_array: np.ndarray) -> int:
        """Submit inference request (non-blocking). Returns batch size for wait()."""
        bs = input_array.shape[0]
        self._input_buf[:bs] = input_array
        self._req_q.put(('infer', bs))
        return bs

    def infer_wait(self, bs: int) -> Tuple[np.ndarray, np.ndarray]:
        """Wait for inference result. Call after infer_async()."""
        resp = self._resp_q.get(timeout=30)
        if resp[0] == 'error':
            raise RuntimeError(f'TRT infer: {resp[1]}')
        return self._policy_buf[:bs].copy(), self._value_buf[:bs].copy()

    def rebuild(
        self,
        onnx_path: str,
        engine_path: str,
        max_batch_size: int = 4096,
        fp16: bool = True,
    ) -> bool:
        """Rebuild TensorRT engine (ONNX must already be re-exported)."""
        if not self.is_ready():
            return False
        self._req_q.put(('rebuild', onnx_path, engine_path,
                         max_batch_size, fp16))
        resp = self._resp_q.get(timeout=600)
        if resp[0] == 'error':
            print(f'[TRT-Client] rebuild error: {resp[1]}')
            self._ready = False
            return False
        self._ready = resp[1]
        return self._ready

    def is_ready(self) -> bool:
        """Check if the server is alive and engine is loaded."""
        return (self._ready
                and self._proc is not None
                and self._proc.is_alive())


# ──────────────────────────────────────────────────────────────────────────────
# Helpers  (run in the main PyTorch process)
# ──────────────────────────────────────────────────────────────────────────────

def export_onnx(model, onnx_path: str = './model/model.onnx') -> bool:
    """Export PyTorch model to ONNX (must run in the PyTorch process)."""
    import torch

    raw = model._orig_mod if hasattr(model, '_orig_mod') else model
    raw.eval()
    dummy = torch.randn(1, 7, 9, 9).cuda()
    try:
        with torch.no_grad():
            torch.onnx.export(
                raw, dummy, onnx_path,
                export_params=True,
                opset_version=13,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['policy', 'value'],
                dynamic_axes={
                    'input':  {0: 'batch_size'},
                    'policy': {0: 'batch_size'},
                    'value':  {0: 'batch_size'},
                },
                dynamo=False,
            )
        # print(f'[TRT] ONNX exported → {onnx_path}')
        return True
    except Exception as e:
        print(f'[TRT] ONNX export failed: {e}')
        return False


# ──────────────────────────────────────────────────────────────────────────────
# C++ TRT Server Client  (near-zero latency IPC via shared memory flags)
# ──────────────────────────────────────────────────────────────────────────────

def _cpp_server_wrapper(shm_ctrl_name, shm_in_name, shm_pol_name, shm_val_name,
                        engine_path, max_batch, so_path):
    """Thin wrapper to call the C++ server from a spawned process."""
    import ctypes
    from multiprocessing.shared_memory import SharedMemory as _SHM

    shm_ctrl = _SHM(name=shm_ctrl_name, create=False)
    shm_in   = _SHM(name=shm_in_name,   create=False)
    shm_pol  = _SHM(name=shm_pol_name,  create=False)
    shm_val  = _SHM(name=shm_val_name,  create=False)

    lib = ctypes.CDLL(so_path)
    lib.trt_server_run.restype = ctypes.c_int
    lib.trt_server_run.argtypes = [
        ctypes.c_void_p, ctypes.c_size_t,  # shm_ctrl
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,  # in/pol/val
        ctypes.c_char_p, ctypes.c_int,  # engine_path, max_batch
    ]

    ret = lib.trt_server_run(
        ctypes.c_void_p(ctypes.addressof(ctypes.c_char.from_buffer(shm_ctrl.buf))),
        ctypes.c_size_t(shm_ctrl.size),
        ctypes.c_void_p(ctypes.addressof(ctypes.c_char.from_buffer(shm_in.buf))),
        ctypes.c_void_p(ctypes.addressof(ctypes.c_char.from_buffer(shm_pol.buf))),
        ctypes.c_void_p(ctypes.addressof(ctypes.c_char.from_buffer(shm_val.buf))),
        engine_path.encode('utf-8'),
        ctypes.c_int(max_batch),
    )

    shm_ctrl.close()
    shm_in.close()
    shm_pol.close()
    shm_val.close()


class TensorRTCppClient:
    """TRT client using C++ server with atomic shared memory flags.

    Same interface as TensorRTProcessClient but ~4x lower inference latency
    by eliminating Python Queue pickle overhead and Python TRT binding calls.

    Control shared memory layout (524 bytes):
      [0:4]   cmd_flag   — 0=idle, 1=infer, 2=shutdown, 3=rebuild
      [4:8]   resp_flag  — 0=idle, 1=done, 2=error
      [8:12]  batch_size — int
      [12:524] rebuild_buf — engine path for rebuild (512 bytes)
    """

    _CTRL_SIZE = 524

    def __init__(self, max_batch_size: int = 4096):
        self.max_batch_size = max_batch_size
        self._ready = False
        self._proc = None
        self._shms: list = []
        self._input_buf = None
        self._policy_buf = None
        self._value_buf = None
        self._ctrl_buf = None  # raw memoryview
        self._ctrl_arr = None  # numpy int32 view for cmd/resp/bs

    def start(
        self,
        engine_path: str = './model/model.trt',
        onnx_path: str = './model/model.onnx',
        fp16: bool = True,
    ) -> bool:
        if self._proc is not None and self._proc.is_alive():
            return self._ready

        so_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               'trt_server.so')
        if not os.path.exists(so_path):
            print('[TRT-C++] trt_server.so not found')
            return False

        pid = os.getpid()
        names = [f'trtc_ctrl_{pid}', f'trtc_in_{pid}',
                 f'trtc_pol_{pid}', f'trtc_val_{pid}']
        sizes = [
            self._CTRL_SIZE,
            self.max_batch_size * 7 * 9 * 9 * 4,
            self.max_batch_size * 81 * 4,
            self.max_batch_size * 1  * 4,
        ]

        for n in names:
            try:
                s = SharedMemory(name=n, create=False)
                s.close(); s.unlink()
            except FileNotFoundError:
                pass

        shms = [SharedMemory(name=n, create=True, size=sz)
                for n, sz in zip(names, sizes)]
        self._shms = shms

        # Zero out control region
        shms[0].buf[:self._CTRL_SIZE] = b'\x00' * self._CTRL_SIZE

        self._ctrl_arr = np.ndarray(3, dtype=np.int32, buffer=shms[0].buf)
        self._input_buf  = np.ndarray(
            (self.max_batch_size, 7, 9, 9), np.float32, buffer=shms[1].buf)
        self._policy_buf = np.ndarray(
            (self.max_batch_size, 81),       np.float32, buffer=shms[2].buf)
        self._value_buf  = np.ndarray(
            (self.max_batch_size, 1),        np.float32, buffer=shms[3].buf)

        self._proc = _mp.Process(
            target=_cpp_server_wrapper,
            args=(names[0], names[1], names[2], names[3],
                  engine_path, self.max_batch_size, so_path),
            daemon=True,
        )
        self._proc.start()

        # Wait for C++ server to signal ready (resp_flag)
        import time
        deadline = time.monotonic() + 600
        while time.monotonic() < deadline:
            resp = int(self._ctrl_arr[1])
            if resp == 1:
                self._ready = True
                print('[TRT-C++] server ready')
                # Reset resp_flag
                self._ctrl_arr[1] = 0
                return True
            elif resp == 2:
                print('[TRT-C++] server FAILED')
                return False
            time.sleep(0.01)

        print('[TRT-C++] server timeout')
        return False

    def shutdown(self):
        if self._proc is not None and self._proc.is_alive():
            try:
                self._ctrl_arr[0] = 2  # cmd=shutdown
                self._proc.join(timeout=5)
            except Exception:
                self._proc.kill()
        for s in self._shms:
            try:
                s.close(); s.unlink()
            except Exception:
                pass
        self._shms.clear()
        self._proc = None
        self._ready = False

    def __del__(self):
        self.shutdown()

    def infer(self, input_array: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        bs = input_array.shape[0]
        self._input_buf[:bs] = input_array
        # Write batch_size, then cmd=infer (order matters for C++ side)
        self._ctrl_arr[2] = bs        # batch_size
        self._ctrl_arr[1] = 0         # clear resp
        self._ctrl_arr[0] = 1         # cmd=infer (triggers C++ loop)
        # Spin-wait for response
        while self._ctrl_arr[1] == 0:
            pass
        resp = int(self._ctrl_arr[1])
        if resp == 2:
            raise RuntimeError('TRT-C++ infer error')
        return self._policy_buf[:bs].copy(), self._value_buf[:bs].copy()

    def infer_async(self, input_array: np.ndarray) -> int:
        bs = input_array.shape[0]
        self._input_buf[:bs] = input_array
        self._ctrl_arr[2] = bs
        self._ctrl_arr[1] = 0
        self._ctrl_arr[0] = 1
        return bs

    def infer_wait(self, bs: int) -> Tuple[np.ndarray, np.ndarray]:
        while self._ctrl_arr[1] == 0:
            pass
        resp = int(self._ctrl_arr[1])
        if resp == 2:
            raise RuntimeError('TRT-C++ infer error')
        return self._policy_buf[:bs].copy(), self._value_buf[:bs].copy()

    def rebuild(self, onnx_path: str, engine_path: str,
                max_batch_size: int = 4096, fp16: bool = True) -> bool:
        if not self.is_ready():
            return False
        # Write engine path into rebuild_buf (offset 12 in ctrl shm)
        path_bytes = engine_path.encode('utf-8')[:511]
        self._shms[0].buf[12:12+len(path_bytes)] = path_bytes
        self._shms[0].buf[12+len(path_bytes)] = 0  # null terminator
        self._ctrl_arr[1] = 0  # clear resp
        self._ctrl_arr[0] = 3  # cmd=rebuild
        # Wait for response
        import time
        deadline = time.monotonic() + 600
        while time.monotonic() < deadline:
            resp = int(self._ctrl_arr[1])
            if resp == 1:
                self._ready = True
                return True
            elif resp == 2:
                self._ready = False
                return False
            time.sleep(0.001)
        self._ready = False
        return False

    def is_ready(self) -> bool:
        return (self._ready
                and self._proc is not None
                and self._proc.is_alive())


# ──────────────────────────────────────────────────────────────────────────────

def get_tensorrt_process_client(
    model=None,
    engine_path: str = './model/model.trt',
    onnx_path: str = './model/model.onnx',
    max_batch_size: int = 4096,
    force_rebuild: bool = False,
) -> Optional[TensorRTProcessClient]:
    """Get a TensorRT inference client backed by a dedicated server process.

    Prefers C++ server (TensorRTCppClient) for lower latency.
    Falls back to Python server (TensorRTProcessClient) if C++ .so not available.
    """
    if not TRT_AVAILABLE:
        print('[TRT] TensorRT not available')
        return None

    # Export ONNX if model provided and needed
    if model is not None and (force_rebuild or not os.path.exists(onnx_path)):
        if not export_onnx(model, onnx_path):
            return None

    # Delete stale engine so server rebuilds from fresh ONNX
    if force_rebuild and os.path.exists(engine_path):
        os.remove(engine_path)

    # Try C++ server first (need engine to already exist since C++ can't build from ONNX)
    so_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'trt_server.so')
    if os.path.exists(so_path) and os.path.exists(engine_path):
        cpp_client = TensorRTCppClient(max_batch_size)
        if cpp_client.start(engine_path=engine_path, onnx_path=onnx_path):
            return cpp_client
        cpp_client.shutdown()
        print('[TRT] C++ server failed, falling back to Python server')

    # Fallback: Python server (can build engine from ONNX)
    client = TensorRTProcessClient(max_batch_size)
    if client.start(engine_path=engine_path, onnx_path=onnx_path):
        return client
    client.shutdown()
    return None
