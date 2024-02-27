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

## Week 9

We start Week 9 with understanding parallel reduction which is usually used to parallelize sum calculation on the GPU. The first example [ReductionSumGPU](Week9/ReductionSumGPU.ipynb) demonstrates how to implement naive reduction on the CPU and then on the GPU. Finally, we see an implementation of reduction which is optimized by avoiding branch divergence. The second example [DotProductGPU](Week9/DotProductGPU.ipynb) calculates dot product on the GPU using shared memory and parallel reduction. In the first phase, the vector products are calculated and stored in shared memory. In the second phase, the product pair are summed using parallel reduction. Finally, we cover how to do basic profiling of our code using tools that ship with the NVIDIA CUDA Computing SDK namely nvprof and ncu in [TestProfilers_nvprof_ncu](Week9/TestProfilers_nvprof_ncu.ipynb).

## Week 10

In Week 10, we talk about using CUDA in python code using Numba and PyCUDA and using teh NVIDIA Thrust library which is STL for CUDA. Two examples are given for each in [HelloNumba](Week10/HelloNumba.ipynb) and [HelloPyCUDA](Week10/HelloPyCUDA.ipynb).

## Week 11

With Week 11, we start working on parallel patterns. We talk about two patterns in Week 1. Convolution and Prefix Sum. The first pattern (Convolution) which helps us filter a given set of data with some set of coefficients (kernel or mask). The first example [Conv1D](Week11/Conv1D.ipynb) shows how to carry out 1D convolution in CUDA. The next example [Conv1D_Modified](Week11/Conv1D_Modified.ipynb) shows how to move the filter mask into constant memory to optimize read accessses inside the CUDA kernel. Finally, we wrap the dicussion on convolution with implementing tiling whereby halo elements are moved from global memory into shared memory to increase the memory acces bandwidth. This is shown in [Conv1D_Tiled](Week11/Conv1D_Tiled.ipynb) example. Finally, we give an example of 2D convolution on the host [Conv2D](Week11/Conv2D.ipynb) and request the students to implement the 2D convolution on the GPU and then optimize it using tiling as an exercise. 

## Week 12

In Week 12, we move to prefix sum. Two examples are given: [PrefixSum_Correct](Week12/PrefixSum_Correct.ipynb) and [PrefixSum_WorkEfficient](Week12/PrefixSum_WorkEfficient.ipynb).

## Week 13

Week 13 focuses on the third parallel pattern that is histogram. We talk about four different strategies for computing the histogram. These strategies are given in four examples: [Strategy 1](Week13/Histogram_Strategy_1.ipynb), [Strategy 2](Week13/Histogram_Strategy_2.ipynb), [Strategy 3](Week13/Histogram_Strategy_3.ipynb) and [Strategy 4](Week13/Histogram_Strategy_4.ipynb).

## Week 14

Week 14 will be focusing on SpMV formats. We will look at the stardard data formats for use in Sparse Matrix Vector multiplication. The example implementation is shared in [SpMV_Formats](Week14/SpMV_Formats.ipynb).

## Week 15

The final week is reserved for project presentations therefore there will be no code examples in this week.
