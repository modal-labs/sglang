"""
EAGLE3 Execution Tracing Utilities

This module provides detailed execution tracing for EAGLE3 speculative decoding.
Enable tracing by setting environment variable: EAGLE_DEBUG_TRACE=1
"""

import os
import functools
import torch
from typing import Any, Dict, List, Optional, Union, Tuple
import inspect

# Global tracing control
TRACE_ENABLED = os.environ.get("EAGLE_DEBUG_TRACE", "0") == "1"
TRACE_INDENT = 0

def trace_enabled():
    """Check if tracing is enabled."""
    return TRACE_ENABLED

def get_tensor_info(tensor: torch.Tensor) -> str:
    """Get tensor shape and basic information."""
    if tensor is None:
        return "None"
    
    # Only show shape, dtype, and device - no tensor value operations
    return f"shape={list(tensor.shape)}, dtype={tensor.dtype}, device={tensor.device}"

def get_object_info(obj: Any) -> str:
    """Get information about any object."""
    if obj is None:
        return "None"
    elif isinstance(obj, torch.Tensor):
        return get_tensor_info(obj)
    elif isinstance(obj, (list, tuple)):
        if len(obj) == 0:
            return f"{type(obj).__name__}(len=0)"
        elif len(obj) > 0 and isinstance(obj[0], torch.Tensor):
            return f"{type(obj).__name__}(len={len(obj)}, tensor_shapes=[{', '.join([f'shape={list(t.shape)}' for t in obj])}])"
        else:
            return f"{type(obj).__name__}(len={len(obj)})"
    elif isinstance(obj, dict):
        return f"dict(len={len(obj)})"
    else:
        return f"{type(obj).__name__}"

def trace_call(func_name: str, *args, **kwargs):
    """Trace function call with arguments."""
    if not TRACE_ENABLED:
        return
    
    global TRACE_INDENT
    indent = "  " * TRACE_INDENT
    
    print(f"\n{indent}🔵 ENTER: {func_name}")
    
    # Print arguments
    if args:
        print(f"{indent}  📥 ARGS:")
        for i, arg in enumerate(args):
            print(f"{indent}    [{i}] {get_object_info(arg)}")
    
    if kwargs:
        print(f"{indent}  📥 KWARGS:")
        for key, value in kwargs.items():
            print(f"{indent}    {key}: {get_object_info(value)}")

def trace_return(func_name: str, result: Any):
    """Trace function return value."""
    if not TRACE_ENABLED:
        return
    
    global TRACE_INDENT
    indent = "  " * TRACE_INDENT
    
    print(f"{indent}  📤 RETURN: {get_object_info(result)}")
    print(f"{indent}🔴 EXIT: {func_name}\n")

def trace_intermediate(step_name: str, **values):
    """Trace intermediate values within a function."""
    if not TRACE_ENABLED:
        return
    
    global TRACE_INDENT
    indent = "  " * (TRACE_INDENT + 1)
    
    print(f"{indent}⚡ {step_name}:")
    for name, value in values.items():
        print(f"{indent}  {name}: {get_object_info(value)}")

def trace_gpu_kernel(kernel_name: str, inputs: Dict[str, Any], outputs: Dict[str, Any]):
    """Trace GPU kernel calls (black box)."""
    if not TRACE_ENABLED:
        return
    
    global TRACE_INDENT
    indent = "  " * (TRACE_INDENT + 1)
    
    print(f"{indent}⚡ GPU_KERNEL: {kernel_name}")
    print(f"{indent}  📥 INPUTS:")
    for name, value in inputs.items():
        print(f"{indent}    {name}: {get_object_info(value)}")
    print(f"{indent}  📤 OUTPUTS:")
    for name, value in outputs.items():
        print(f"{indent}    {name}: {get_object_info(value)}")

def trace_memory_op(operation: str, details: Dict[str, Any]):
    """Trace memory operations."""
    if not TRACE_ENABLED:
        return
    
    global TRACE_INDENT
    indent = "  " * (TRACE_INDENT + 1)
    
    print(f"{indent}💾 MEMORY: {operation}")
    for name, value in details.items():
        print(f"{indent}  {name}: {get_object_info(value)}")

def eagle_trace(func):
    """Decorator to automatically trace function calls."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if not TRACE_ENABLED:
            return func(*args, **kwargs)
        
        global TRACE_INDENT
        
        # Get function name with class if it's a method
        func_name = func.__name__
        if args and hasattr(args[0], '__class__'):
            func_name = f"{args[0].__class__.__name__}.{func_name}"
        
        trace_call(func_name, *args[1:] if args and hasattr(args[0], '__class__') else args, **kwargs)
        TRACE_INDENT += 1
        
        try:
            result = func(*args, **kwargs)
            trace_return(func_name, result)
            return result
        finally:
            TRACE_INDENT -= 1
    
    return wrapper

def print_execution_summary():
    """Print execution summary at the end."""
    if not TRACE_ENABLED:
        return
    
    print("\n" + "="*80)
    print("EAGLE3 EXECUTION TRACE COMPLETE")
    print("="*80)