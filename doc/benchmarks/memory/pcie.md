PCIe Memory Benchmark
=====================

Test System:
 - Linux 4.4.0-38-generic (Ubuntu)
 - Intel(R) Xeon(R) CPU E5-2630 v3 @ 2.40GHz (32 cores)
 - 8 x Tesla K80
 - 256GB RAM

Further details on the GPU:
```
CUDA Driver version:  8000
CUDA Runtime version: 8000
=============
Devices:
Tesla K80
    Compute capability: 3.7
    Global memory: 11439.9 MiB
    DMA engines: 2
    Multi processors: 13
    Warp size: 32
    Max concurrent kernels: 1
    Max grid size: 2147483647, 65535, 65535
    Max block size: 1024, 1024, 64
    Max threads per block: 1024
Tesla K80
    Compute capability: 3.7
    Global memory: 11439.9 MiB
    DMA engines: 2
    Multi processors: 13
    Warp size: 32
    Max concurrent kernels: 1
    Max grid size: 2147483647, 65535, 65535
    Max block size: 1024, 1024, 64
    Max threads per block: 1024
Tesla K80
    Compute capability: 3.7
    Global memory: 11439.9 MiB
    DMA engines: 2
    Multi processors: 13
    Warp size: 32
    Max concurrent kernels: 1
    Max grid size: 2147483647, 65535, 65535
    Max block size: 1024, 1024, 64
    Max threads per block: 1024
Tesla K80
    Compute capability: 3.7
    Global memory: 11439.9 MiB
    DMA engines: 2
    Multi processors: 13
    Warp size: 32
    Max concurrent kernels: 1
    Max grid size: 2147483647, 65535, 65535
    Max block size: 1024, 1024, 64
    Max threads per block: 1024
Tesla K80
    Compute capability: 3.7
    Global memory: 11439.9 MiB
    DMA engines: 2
    Multi processors: 13
    Warp size: 32
    Max concurrent kernels: 1
    Max grid size: 2147483647, 65535, 65535
    Max block size: 1024, 1024, 64
    Max threads per block: 1024
Tesla K80
    Compute capability: 3.7
    Global memory: 11439.9 MiB
    DMA engines: 2
    Multi processors: 13
    Warp size: 32
    Max concurrent kernels: 1
    Max grid size: 2147483647, 65535, 65535
    Max block size: 1024, 1024, 64
    Max threads per block: 1024
Tesla K80
    Compute capability: 3.7
    Global memory: 11439.9 MiB
    DMA engines: 2
    Multi processors: 13
    Warp size: 32
    Max concurrent kernels: 1
    Max grid size: 2147483647, 65535, 65535
    Max block size: 1024, 1024, 64
    Max threads per block: 1024
Tesla K80
    Compute capability: 3.7
    Global memory: 11439.9 MiB
    DMA engines: 2
    Multi processors: 13
    Warp size: 32
    Max concurrent kernels: 1
    Max grid size: 2147483647, 65535, 65535
    Max block size: 1024, 1024, 64
    Max threads per block: 1024


```

Pinned memory (physically contiguous)
-------------------------------------

```
[CUDA Bandwidth Test] - Starting...
Running on...

 Device 0: Tesla K80
 Quick Mode

 Host to Device Bandwidth, 1 Device(s)
 PINNED Memory Transfers
   Transfer Size (Bytes)	Bandwidth(MB/s)
   33554432			11340.6

 Device to Host Bandwidth, 1 Device(s)
 PINNED Memory Transfers
   Transfer Size (Bytes)	Bandwidth(MB/s)
   33554432			11328.4

 Device to Device Bandwidth, 1 Device(s)
 PINNED Memory Transfers
   Transfer Size (Bytes)	Bandwidth(MB/s)
   33554432			170697.5

```

Pinned memory (physically contiguous)
-------------------------------------

```
[CUDA Bandwidth Test] - Starting...
Running on...

 Device 0: Tesla K80
 Quick Mode

 Host to Device Bandwidth, 1 Device(s)
 PAGEABLE Memory Transfers
   Transfer Size (Bytes)	Bandwidth(MB/s)
   33554432			9084.3

 Device to Host Bandwidth, 1 Device(s)
 PAGEABLE Memory Transfers
   Transfer Size (Bytes)	Bandwidth(MB/s)
   33554432			10475.3

 Device to Device Bandwidth, 1 Device(s)
 PAGEABLE Memory Transfers
   Transfer Size (Bytes)	Bandwidth(MB/s)
   33554432			170071.6

```

multiple GPUs
-------------

```
[CUDA Bandwidth Test] - Starting...

!!!!!Cumulative Bandwidth to be computed from all the devices !!!!!!

Running on...

 Device 0: Tesla K80
 Device 1: Tesla K80
 Device 2: Tesla K80
 Device 3: Tesla K80
 Device 4: Tesla K80
 Device 5: Tesla K80
 Device 6: Tesla K80
 Device 7: Tesla K80
 Quick Mode

 Host to Device Bandwidth, 8 Device(s)
 PINNED Memory Transfers
   Transfer Size (Bytes)	Bandwidth(MB/s)
   33554432			78952.3

 Device to Host Bandwidth, 8 Device(s)
 PINNED Memory Transfers
   Transfer Size (Bytes)	Bandwidth(MB/s)
   33554432			65986.9

 Device to Device Bandwidth, 8 Device(s)
 PINNED Memory Transfers
   Transfer Size (Bytes)	Bandwidth(MB/s)
   33554432			1362316.4
```
