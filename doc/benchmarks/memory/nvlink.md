NVLink Memory Benchmark
=======================

Test System:
 - Linux 3.10.0
 - POWER8NVL @ 4.023GHz (160 cores)
 - 4 x Tesla P100-SXM2-16GB
 - 256GB RAM

Further details on the GPU:
```
CUDA Driver version:  8000
CUDA Runtime version: 8000
=============
Devices:
Tesla P100-SXM2-16GB
    Compute capability: 6.0
    Global memory: 16276.2 MiB
    DMA engines: 3
    Multi processors: 56
    Warp size: 32
    Max concurrent kernels: 1
    Max grid size: 2147483647, 65535, 65535
    Max block size: 1024, 1024, 64
    Max threads per block: 1024
Tesla P100-SXM2-16GB
    Compute capability: 6.0
    Global memory: 16276.2 MiB
    DMA engines: 3
    Multi processors: 56
    Warp size: 32
    Max concurrent kernels: 1
    Max grid size: 2147483647, 65535, 65535
    Max block size: 1024, 1024, 64
    Max threads per block: 1024
Tesla P100-SXM2-16GB
    Compute capability: 6.0
    Global memory: 16276.2 MiB
    DMA engines: 3
    Multi processors: 56
    Warp size: 32
    Max concurrent kernels: 1
    Max grid size: 2147483647, 65535, 65535
    Max block size: 1024, 1024, 64
    Max threads per block: 1024
Tesla P100-SXM2-16GB
    Compute capability: 6.0
    Global memory: 16276.2 MiB
    DMA engines: 3
    Multi processors: 56
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

 Device 0: Tesla P100-SXM2-16GB
 Quick Mode

 Host to Device Bandwidth, 1 Device(s)
 PINNED Memory Transfers
   Transfer Size (Bytes)	Bandwidth(MB/s)
   33554432			29526.9

 Device to Host Bandwidth, 1 Device(s)
 PINNED Memory Transfers
   Transfer Size (Bytes)	Bandwidth(MB/s)
   33554432			21336.3

 Device to Device Bandwidth, 1 Device(s)
 PINNED Memory Transfers
   Transfer Size (Bytes)	Bandwidth(MB/s)
   33554432			500340.8
```

Pinned memory (physically contiguous)
-------------------------------------

```
[CUDA Bandwidth Test] - Starting...
Running on...

 Device 0: Tesla P100-SXM2-16GB
 Quick Mode

 Host to Device Bandwidth, 1 Device(s)
 PAGEABLE Memory Transfers
   Transfer Size (Bytes)	Bandwidth(MB/s)
   33554432			14134.4

 Device to Host Bandwidth, 1 Device(s)
 PAGEABLE Memory Transfers
   Transfer Size (Bytes)	Bandwidth(MB/s)
   33554432			13948.3

 Device to Device Bandwidth, 1 Device(s)
 PAGEABLE Memory Transfers
   Transfer Size (Bytes)	Bandwidth(MB/s)
   33554432			500596.4
```

multiple GPUs
-------------

```
[CUDA Bandwidth Test] - Starting...

!!!!!Cumulative Bandwidth to be computed from all the devices !!!!!!

Running on...

 Device 0: Tesla P100-SXM2-16GB
 Device 1: Tesla P100-SXM2-16GB
 Device 2: Tesla P100-SXM2-16GB
 Device 3: Tesla P100-SXM2-16GB
 Quick Mode

 Host to Device Bandwidth, 4 Device(s)
 PINNED Memory Transfers
   Transfer Size (Bytes)	Bandwidth(MB/s)
   33554432			127305.9

 Device to Host Bandwidth, 4 Device(s)
 PINNED Memory Transfers
   Transfer Size (Bytes)	Bandwidth(MB/s)
   33554432			108332.4

 Device to Device Bandwidth, 4 Device(s)
 PINNED Memory Transfers
   Transfer Size (Bytes)	Bandwidth(MB/s)
   33554432			1989760.4

```
