# CUDA_Spring_2024
Accompanying code repository for my CUDA course offering in Spring 2024. The first week work is largely getting started with the two most common machine and deeplearning platforms Google Colab and Kaggle. Both are equally good but Kaggle provides more recent GPUs and provides a range of GPUs to pick from. Google Colab is limited to one GPU or a TPU. 

## Week 1

If you examine the Week1 folder your will see 6 jupyter notebooks. The first three do hello world in Google Colab while the later 3 do hello world in Kaggle. Feel free to get comfortable with both of these environments.  

## Week 2

We extend our knowledge of CUDA slightly by getting to know how to initialize data on the device (GPU). We also see how to check for errors in our CUDA code. We also see how to add two numbers on the GPU and return the result from device (GPU) back to the host (CPU).  

## Week 3

We extend our knowledge of CUDA by getting to know how to pass a large array of data on the device (GPU). We also see how to time code on the CPU as well as on the GPU.

## Week 4

We extend our knowledge by starting work in 2D. We first start with creating a simple [image viewer in python](Week4/ImageViewer.ipynb). Next we see how to [lauch threads in 2D](Week4/Simple2D.ipynb). Then, we look at how to do [raw image processing in CUDA](Week4/RawImageProcessing.ipynb) and Finally, we wrap up this week with a small task where students are asked to finish an implementation of [matrix multiplication in CUDA](Week4/MatrixMultiplication_Task.ipynb).

## Week 5

We now see how we can use shared memory to efficienlty optimize writing to device memory across multple threads concurrently. The first example [Shared Memory Bitmap](Week5/SharedMemoryBitmap.ipynb) demonstrates a simple example of creating a bitmap image using shared memory to store data in a bitmap using multiple parallel threads. The examples demonstrates the general dataflow pattern of working with the shared memory. We create the shared memory buffer first. Then we ask each thread to fill up its data from the global memory to shared memory. Next, we put the __syncthreads() call which inserts a synchronization barrier. This barrier ensures that the following execution is halted until all threads in the block have finished writing to their shared memory location. After __syncthreads call, each threads reads data from its own shared memory location for processing. If we comment the __syncthreads() call we see noticable garbage values in the output as there is no guarantee that all threads have written to their shared memory location. 

The second example [WaveformsMemoryBitmap](Week5/WaveformsMemoryBitmap.ipynb) details how you may generate different types of waves in memory using CUDA. The third example [MatrixMultiplicationTiled](Week5/MatrixMultiplicationTiled.ipynb) optimizes the MatrixMultiplication example through tiliing whereby tile of data is copied from device memory into shared memory and then used for matrix multiplication. We do comparison of the naive matrixmultiplication against CPU as well as the optimized matrix multiplication on the GPU.

## Week 6

Week 6 focusess on dynamic parallelism in CUDA that is an ability to launch multiple CUDA kernels from a single CUDA kernel. We give one example of this [CUDA_DynamicParallelism](Week6/CUDA_DynamicParallelism.ipynb).

## Week 7

In Week 7, we focus on concurrency using CUDA streams. We cover the streams in detail using three examples, [CUDA_Streams](Week7/CUDA_Streams.ipynb), [CUDA_Streams](Week7/CUDA_Streams.ipynb) and [CUDA_Streams_SyncEvents](Week7/CUDA_Streams_SyncEvents.ipynb). 

## Week 8

Week 8 starts with understanding of global static memory, unified memory and zero copy memory. We give three examples on these including [globalStaticMemory](Week8/globalStaticMemory.ipynb), [SumArrayZeroCopy](Week8/SumArrayZeroCopy.ipynb) and [DotProductGPU_UnifiedMemory](Week8/DotProductGPU_UnifiedMemory.ipynb). 
