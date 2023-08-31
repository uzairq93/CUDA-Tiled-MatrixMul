/*
 * Copyright 1993-2006 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO USER:   
 *
 * This source code is subject to NVIDIA ownership rights under U.S. and 
 * international Copyright laws.  
 *
 * This software and the information contained herein is PROPRIETARY and 
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and 
 * conditions of a Non-Disclosure Agreement.  Any reproduction or 
 * disclosure to any third party without the express written consent of 
 * NVIDIA is prohibited.     
 *
 * NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE 
 * CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR 
 * IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH 
 * REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF 
 * MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.   
 * IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL, 
 * OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS 
 * OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE 
 * OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE 
 * OR PERFORMANCE OF THIS SOURCE CODE.  
 *
 * U.S. Government End Users.  This source code is a "commercial item" as 
 * that term is defined at 48 C.F.R. 2.101 (OCT 1995), consisting  of 
 * "commercial computer software" and "commercial computer software 
 * documentation" as such terms are used in 48 C.F.R. 12.212 (SEPT 1995) 
 * and is provided to the U.S. Government only as a commercial end item.  
 * Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through 
 * 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the 
 * source code with only those rights set forth herein.
 */

/* Matrix multiplication: P = M * N.
 * Device code.
 */

#ifndef _MATRIXMUL_KERNEL_H_
#define _MATRIXMUL_KERNEL_H_

#include <stdio.h>
#include "matrixmul.h"

////////////////////////////////////////////////////////////////////////////////
//! Simple test kernel for device functionality
//! @param g_idata  input data in global memory
//! @param g_odata  output data in global memory
////////////////////////////////////////////////////////////////////////////////
// Matrix multiplication kernel thread specification
__global__ void MatrixMulKernel(Matrix M, Matrix N, Matrix P)
{
    // STEP 1: Allocate two matrices in shared memory that will eventually hold the elements of the current tile of M and N

    const int BLOCK_DIM = 32; // Blocks will always be 32x32
    const int PAD_DIM = 33;
    __shared__ float Mshared[PAD_DIM*BLOCK_DIM]; // Allocate M in shared memory. Add padding to avoid bank conflicts when threads process their own row
    __shared__ float Nshared[BLOCK_DIM*BLOCK_DIM]; // Same with N, but we automatically avoid bank conflicts when processing unique columns for the dot product.

    // STEP 2: Each thread is responsible for calculating one element of P. Find which element this thread will aggregate and initialize the sum.

    int rowP = blockIdx.y * blockDim.y + threadIdx.y; // rowP is the row index, use rowP*P.width when accessing
    int colP = blockIdx.x * blockDim.x + threadIdx.x; // colP is the column index
    float dotProduct = 0.0; // The dot product of M[rowP][:] and N[:][colP] will be accumulated as this thread works

    // STEP 3: Now, loop over all sets of source tiles {M_(i,j), N_(j,k)} that are needed to calculate P[rowP][colP].
    // At each step of this loop, we will bring the sets into shared memory to avoid slow global memory addesses. Then we will
    // compute the partial results of the dotProduct. This will continue until the full dotProduct has been calculated. 

    int totalTiles = (M.width + BLOCK_DIM - 1)/BLOCK_DIM; // Ceiling division, we will need 'totalTiles' source tile sets to get the full dotProduct
    int blockRowOffset = blockIdx.y * blockDim.y; // Necessary to adjust addressing of M when blockIdx.y is nonzero
    int blockColOffset = blockIdx.x * blockDim.x; // Necessary to adjust addressing of N when blockIdx.x in nonzero

    for (int tileIdx = 0; tileIdx < totalTiles; tileIdx++) {
        // STEP 3a: Each thread will copy one element from M and one from N into the current tile

        int readRowM = blockRowOffset + threadIdx.y;
        int readColM = tileIdx * blockDim.x + threadIdx.x;
        int readRowN = tileIdx * blockDim.y + threadIdx.y;
        int readColN = blockColOffset + threadIdx.x;

        // Write the element from M into shared memory. In case we're out of bounds, write a 0 instead.

        if (readRowM < M.height && readColM < M.width) {
            Mshared[threadIdx.y * PAD_DIM + threadIdx.x] = M.elements[readRowM * M.width + readColM];
        } else {
            Mshared[threadIdx.y * PAD_DIM + threadIdx.x] = 0.0;
        }

        // Write the element from M into shared memory. In case we're out of bounds, write a 0 instead.

        if (readRowN < N.height && readColN < N.width) {
            Nshared[threadIdx.y * BLOCK_DIM + threadIdx.x] = N.elements[readRowN * N.width + readColN];
        } else {
            Nshared[threadIdx.y * BLOCK_DIM + threadIdx.x] = 0.0;
        }

        __syncthreads(); // Wait for all threads to finish reading their elements before we move on to calculating the dot products

        // STEP 3b: With shared memory properly filled, each thread will now process the partial dot product between its assigned row of
        //          the M tile and the assigned column of the N tile.

        for (int k = 0; k < BLOCK_DIM; k++) {
            dotProduct += (Mshared[threadIdx.y * PAD_DIM + k] * Nshared[k * BLOCK_DIM + threadIdx.x]);
        }

        __syncthreads(); // Wait for all threads to finish their partial results before contributing them to the total in step 4
    }

    // STEP 4: Store the result in P if the indexes are in bounds (important if P.height or P.width aren't multiples of 32)

    if (rowP < P.height && colP < P.width) {
        P.elements[rowP*P.width + colP] = dotProduct;
    }
}

#endif // #ifndef _MATRIXMUL_KERNEL_H_
