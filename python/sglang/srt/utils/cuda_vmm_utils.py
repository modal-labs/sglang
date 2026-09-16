"""Subset of upstream cuda_vmm_utils used by the stream-ordered CUDA-IPC lease pool; the VMM transport itself is not ported."""

_drv = None


def _get_cuda_driver():
    """Lazily import cuda.bindings.driver (cached after first call)."""
    global _drv
    if _drv is None:
        from cuda.bindings import driver

        _drv = driver
    return _drv


def check_drv(result_tuple, label):
    """Check a cuda.bindings driver call result and return the value."""
    if not isinstance(result_tuple, tuple):
        result_tuple = (result_tuple,)
    err = result_tuple[0]
    drv = _get_cuda_driver()
    if err != drv.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"{label}: {err}")
    return result_tuple[1] if len(result_tuple) > 1 else None
