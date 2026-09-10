# runtime/weight_streamer.py
"""
Asynchronous Pipelined Weight Streaming Runtime for Jarvis
==========================================================
Enables execution of models larger than physical GPU VRAM by:
1. Staging full layer weights in pinned host memory (high-speed DMA).
2. Double/Triple buffering GPU memory slots to completely overlap PCIe transfer with computation.
3. Fine-grained CUDA event synchronization (zero CPU busy-waiting, zero global cudaSync).
4. Configurable prefetch distance (D in {0, 1, 2, 3}).
5. Automatic buffer reuse and memory footprint bounding.
"""

import time
import math
from typing import List, Dict, Optional
import torch
import torch.nn as nn

class LayerWeightChunk:
    """Represents a discrete weight chunk (one or more transformer blocks)."""
    def __init__(self, chunk_id: int, state_dict: Dict[str, torch.Tensor], pin_memory: bool = True):
        self.chunk_id = chunk_id
        self.param_names = list(state_dict.keys())
        self.shapes = {k: v.shape for k, v in state_dict.items()}
        self.dtypes = {k: v.dtype for k, v in state_dict.items()}
        self.total_bytes = sum(v.numel() * v.element_size() for v in state_dict.values())

        # Host pinned staging
        self.host_params: Dict[str, torch.Tensor] = {}
        for k, v in state_dict.items():
            cpu_tensor = v.detach().cpu().contiguous()
            if pin_memory:
                cpu_tensor = cpu_tensor.pin_memory()
            self.host_params[k] = cpu_tensor

class AsynchronousWeightStreamer:
    """
    Manages asynchronous double-buffering / triple-buffering pipelines
    between pinned host RAM and GPU VRAM slots.
    """
    def __init__(
        self,
        chunks: List[LayerWeightChunk],
        num_buffers: int = 2,  # 2 for double-buffering, 3 for triple-buffering
        prefetch_distance: int = 1,
        device: str = "cuda"
    ):
        self.chunks = chunks
        self.num_chunks = len(chunks)
        self.num_buffers = max(1, num_buffers)
        self.prefetch_distance = prefetch_distance
        self.device = torch.device(device)

        # Dedicated execution streams
        self.transfer_stream = torch.cuda.Stream(device=self.device)
        self.compute_stream = torch.cuda.Stream(device=self.device)

        # Pre-allocate GPU buffer slots
        # Each slot holds pre-allocated GPU tensors matching the chunk geometry
        self.gpu_slots: List[Dict[str, torch.Tensor]] = []
        ref_chunk = chunks[0]
        for slot_idx in range(self.num_buffers):
            slot_tensors = {}
            for k, host_t in ref_chunk.host_params.items():
                slot_tensors[k] = torch.empty(
                    host_t.shape, dtype=host_t.dtype, device=self.device
                )
            self.gpu_slots.append(slot_tensors)

        # CUDA Events for synchronization
        # transfer_ready_events[c]: signaled when chunk c is fully copied to GPU
        # compute_done_events[slot]: signaled when computation on slot is finished
        self.transfer_ready_events = [torch.cuda.Event(enable_timing=True) for _ in range(self.num_chunks)]
        self.compute_done_events = [torch.cuda.Event(enable_timing=True) for _ in range(self.num_buffers)]

        # Slot assignment tracking
        self.chunk_to_slot: Dict[int, int] = {}
        self.active_transfers: Dict[int, int] = {}

    def _get_slot_for_chunk(self, chunk_idx: int) -> int:
        return chunk_idx % self.num_buffers

    def prefetch_chunk(self, chunk_idx: int):
        """Asynchronously dispatches DMA transfer of chunk_idx over PCIe."""
        if chunk_idx >= self.num_chunks or chunk_idx in self.active_transfers:
            return

        slot_idx = self._get_slot_for_chunk(chunk_idx)
        slot_dict = self.gpu_slots[slot_idx]
        chunk = self.chunks[chunk_idx]

        with torch.cuda.stream(self.transfer_stream):
            # Wait until previous compute on this buffer slot has completed
            self.transfer_stream.wait_event(self.compute_done_events[slot_idx])

            # Non-blocking async copy from pinned host memory to GPU buffer slot
            for k, host_tensor in chunk.host_params.items():
                slot_dict[k].copy_(host_tensor, non_blocking=True)

            # Record that transfer of this chunk has finished
            self.transfer_ready_events[chunk_idx].record(self.transfer_stream)

        self.chunk_to_slot[chunk_idx] = slot_idx
        self.active_transfers[chunk_idx] = slot_idx

    def acquire_chunk(self, chunk_idx: int) -> Dict[str, torch.Tensor]:
        """
        Retrieves GPU tensors for chunk_idx on the compute stream.
        Guarantees that compute_stream waits for DMA transfer to complete without CPU synchronization.
        """
        # Ensure transfer was initiated
        if chunk_idx not in self.active_transfers:
            self.prefetch_chunk(chunk_idx)

        slot_idx = self.chunk_to_slot[chunk_idx]

        with torch.cuda.stream(self.compute_stream):
            # Compute stream waits for the transfer event of this chunk
            self.compute_stream.wait_event(self.transfer_ready_events[chunk_idx])

        # Proactively trigger prefetch of upcoming chunks according to prefetch distance
        for dist in range(1, self.prefetch_distance + 1):
            next_idx = chunk_idx + dist
            if next_idx < self.num_chunks:
                self.prefetch_chunk(next_idx)

        return self.gpu_slots[slot_idx]

    def release_chunk(self, chunk_idx: int):
        """Marks chunk compute complete so its buffer slot can be recycled."""
        slot_idx = self.chunk_to_slot[chunk_idx]
        with torch.cuda.stream(self.compute_stream):
            self.compute_done_events[slot_idx].record(self.compute_stream)
        if chunk_idx in self.active_transfers:
            del self.active_transfers[chunk_idx]

    def synchronize(self):
        """Full barrier across compute and transfer streams."""
        self.compute_stream.synchronize()
        self.transfer_stream.synchronize()
