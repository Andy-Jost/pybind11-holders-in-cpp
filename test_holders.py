import cuda_core_holders_demo as holders

from cuda.bindings import driver
from cuda.core.experimental import Device, DeviceMemoryResource, Stream
from cuda.core.experimental._utils.cuda_utils import handle_return
device = Device()
device.set_current()

mr = DeviceMemoryResource(device)
buffer = mr.allocate(64)


# # Allocate stream.
# # flags = driver.CUstream_flags.CU_STREAM_NON_BLOCKING
# # priority = 0
# # cu_stream = handle_return(driver.cuStreamCreateWithPriority(flags, priority))
# # h_stream = holders.Stream(cu_stream)
# stream = device.create_stream()
# 
# 
# # Allocate memory pool.
# properties = driver.CUmemPoolProps()
# properties.allocType = driver.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
# properties.handleTypes = driver.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_NONE
# properties.location.id = device.device_id
# properties.location.type = driver.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
# properties.maxSize = 2*1024*1024
# properties.win32SecurityAttributes = 0
# properties.usage = 0
# 
# cu_pool = handle_return(driver.cuMemPoolCreate(properties))
# h_pool = holders.MemPool(cu_pool)
# 
# 
# # Allocate device pointer.
# NBYTES = 64
# cu_devptr = handle_return(driver.cuMemAllocFromPoolAsync(NBYTES, int(h_pool), int(stream.internal_handle)))
# h_devptr = holders.Deviceptr(cu_devptr, h_pool, stream.internal_handle)


import code
code.interact(local=dict(globals(), **locals()))
