import sys
import socket
import json
import time
import cupy as cp
import numpy as np
import base64
import torch
import logging

NUM_ELEMS = 2 ** 31
PORT = 12345
SRC_GPU = 0
DST_GPU = 1
MAX_RETRIES = 10
INITIAL_RETRY_DELAY = 0.1  # seconds
MAX_RETRY_DELAY = 5.0  # seconds


def connect_with_retry(addr):
    for attempt in range(MAX_RETRIES):
        try:
            return socket.create_connection(addr)
        except (ConnectionRefusedError, OSError) as e:
            delay = min(INITIAL_RETRY_DELAY * (2 ** attempt), MAX_RETRY_DELAY)
            logging.info(f"[Sender] Connection failed: {e}. Retrying in {delay:.1f}s...")
            time.sleep(delay)
    raise ConnectionRefusedError(f"Failed to connect after {MAX_RETRIES} attempts.")


def sender():
    src_buf = torch.arange(NUM_ELEMS, dtype=torch.float32, device=f"cuda:{SRC_GPU}")
    nbytes = src_buf.numel() * src_buf.element_size()

    src_ptr = src_buf.data_ptr()

    with connect_with_retry(("localhost", PORT)) as s:
        info = json.loads(s.recv(1024).decode())
        logging.info("[Sender] Connected to receiver successfully.")
        dst_handle = base64.b64decode(info["dst_handle"])
        dst_ptr = cp.cuda.runtime.ipcOpenMemHandle(dst_handle)

        with cp.cuda.Device(SRC_GPU):
            mem_src = cp.cuda.UnownedMemory(int(src_ptr), int(nbytes), None)
            ptr_src = cp.cuda.MemoryPointer(mem_src, 0)
            cp_src = cp.ndarray((nbytes,), dtype=cp.uint8, memptr=ptr_src)

        with cp.cuda.Device(DST_GPU):
            mem_dst = cp.cuda.UnownedMemory(int(dst_ptr), int(nbytes), None)
            ptr_dst = cp.cuda.MemoryPointer(mem_dst, 0)
            cp_dst = cp.ndarray((nbytes,), dtype=cp.uint8, memptr=ptr_dst)

        logging.info("[Sender] Starting copy...")
        cp.copyto(cp_dst, cp_src)
        cp.cuda.Stream.null.synchronize()
        logging.info("[Sender] Copy done.")

        logging.info("[Sender] Data copied successfully.")
        s.sendall(b"Done\n")
        logging.info("[Sender] Receiver notified.")
        # logging.info("[Sender] data =", src_buf.cpu().numpy())


def receiver():
    dst_buf = torch.zeros(NUM_ELEMS, dtype=torch.float32, device=f"cuda:{DST_GPU}")
    nbytes = dst_buf.numel() * dst_buf.element_size()

    with cp.cuda.Device(DST_GPU):
        mem_dst = cp.cuda.UnownedMemory(int(dst_buf.data_ptr()), int(nbytes), None)
        ptr_dst = cp.cuda.MemoryPointer(mem_dst, 0)
        cp_dst = cp.ndarray((nbytes,), dtype=cp.uint8, memptr=ptr_dst)
        handle = cp.cuda.runtime.ipcGetMemHandle(cp_dst.data.ptr)
        logging.info(f"[Receiver] cupy ptr = {hex(cp_dst.data.ptr)}, pytorch ptr = {hex(dst_buf.data_ptr())}")

    info = {
        "dst_handle": base64.b64encode(handle).decode(),
    }

    with socket.create_server(("localhost", PORT), reuse_port=True) as s:
        conn, _ = s.accept()
        with conn as f:
            f.sendall(json.dumps(info).encode() + b"\n")
            logging.info("[Receiver] IPC info sent to sender.")
            while True:
                data = f.recv(1024)
                if len(data) > 0:
                    break
                time.sleep(INITIAL_RETRY_DELAY)
            logging.info("[Receiver] Data received successfully.")
            # logging.info("[Receiver] data =", dst_buf.cpu().numpy())


if __name__ == "__main__":
    role = sys.argv[1]
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)
    if role == "sender":
        sender()
    elif role == "receiver":
        receiver()
    else:
        raise ValueError(f"Invalid role: {role}")
