# Parallelization Strategies for Matrix Multiplication

Based on the code in the workspace, there are three different paradigms used for parallelizing General Matrix Multiplication (GEMM): **OpenMP** (Shared Memory Multithreading), **MPI** (Distributed Memory Processing), and **CUDA** (GPU Acceleration).

Here is a breakdown of how each technology is utilized in the project to speed up matrix multiplication.

## 1. OpenMP (`project/omp_gemm.c`)

OpenMP speeds up the computation by dividing the work among multiple CPU cores on a single machine sharing the same memory. The code implements this with increasing levels of optimization:

*   **Parallelization (`omp_gemm`):** It uses the compiler directive `#pragma omp parallel for collapse(2)`. This takes the two outer loops (rows `i` and columns `j`) and fuses them into a single large loop of size $N \times N$, distributing the calculation of each element in the result matrix `C` across available CPU threads.
*   **Memory Optimization - Transposition (`omp_trans_gemm`):** In C, 2D arrays are stored in row-major order. A standard matrix multiplication accesses matrix `B` column-by-column, which causes frequent CPU cache misses (thrashing). To fix this, the code first transposes `B` so that the inner loop reads memory sequentially. Both the transposition and the multiplication are parallelized.
*   **Cache Blocking/Tiling (`omp_trans_tiled_gemm`):** This is the most optimized CPU version. It divides the matrices into smaller `64x64` blocks that perfectly fit into the CPU's fast L1/L2 cache. This drastically reduces the number of times data has to be fetched from main RAM. OpenMP parallelizes the calculation across these blocks (`collapse(3)` on the outer block-iterating loops).

## 2. MPI (`mpi/mpi_matrix_matrix_mult.c`)

MPI speeds up computation by splitting the work across multiple independent processes, which could potentially be running on completely different computers over a network. Since they don't share memory, data must be explicitly sent between them. The code demonstrates two main strategies:

*   **Scatter and Broadcast (`mpi_scatter_broadcast_gemm`):**
    *   **`MPI_Scatter`:** Rank 0 divides matrix `A` into chunks of rows and sends a specific chunk to each process.
    *   **`MPI_Bcast`:** Matrix `B` is broadcasted in its entirety to *every* process.
    *   Each process multiplies its small chunk of `A` with the full matrix `B`, calculating a small chunk of rows for the final matrix `C`.
    *   **`MPI_Gather`:** Rank 0 collects all the calculated chunks of `C` back together into the final matrix.
*   **Cyclic Block Distribution (`mpi_trans_tiled_gemm`):** Instead of using MPI scatter, this function uses a cyclic distribution loop: `for(int ii = BLOCK_SIZE*p_rank; ii < size; ii += BLOCK_SIZE*p_size)`. Each rank skips ahead and calculates specific blocks of the matrix. Finally, `MPI_Reduce` with the `MPI_SUM` operation is used to merge everyone's partially filled local matrices into the final global matrix.

## 3. CUDA (`cuda/cuda_gemm.cu`)

CUDA accelerates the process by offloading the heavy mathematical lifting to the GPU, which has thousands of tiny cores designed specifically for this type of parallel math.

*   **Flattened Memory:** First, matrices are flattened from 2D arrays into 1D arrays of size $N^2$. This makes passing memory between the CPU and GPU much easier. Memory is allocated on the GPU (`cudaMalloc`), and the CPU matrices are copied over (`cudaMemcpy`).
*   **Massive Threading:** The GPU spawns a massive grid of threads. You launch `(N*N / 256)` blocks, each containing `256` threads.
*   **One Thread Per Element:** The parallelization strategy here is extreme: **Each individual GPU thread is responsible for calculating exactly one element of the result matrix `C`.**
*   A thread calculates its global index (`idx = blockIdx.x * blockDim.x + threadIdx.x`), figures out which row of `A` and column of `B` it corresponds to, runs the dot-product loop for that specific coordinate, and saves it to `C`.

---

### Summary of the Scaling Strategies

*   **OpenMP:** "Let's split the loops across our CPU cores and optimize how we read the RAM."
*   **MPI:** "Let's chop the matrices into pieces, mail the pieces to different computers, and merge the results."
*   **CUDA:** "Let's create $1,048,576$ tiny workers (for a $1024 \times 1024$ matrix) and tell each worker to calculate exactly one cell of the final answer."

