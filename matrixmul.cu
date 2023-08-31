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

/* Matrix multiplication: C = A * B.
 * Host code.
 */

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// includes, project
#include <cutil.h>

// includes, kernels
#include <matrixmul_kernel.cu>

////////////////////////////////////////////////////////////////////////////////
// declarations, forward

extern "C"
void computeGold(float*, const float*, const float*, unsigned int, unsigned int, unsigned int);

Matrix AllocateDeviceMatrix(const Matrix M);
Matrix AllocateMatrix(int height, int width, int init);
void CopyToDeviceMatrix(Matrix Mdevice, const Matrix Mhost);
void CopyFromDeviceMatrix(Matrix Mhost, const Matrix Mdevice);
int ReadFile(Matrix* M, char* file_name);
void WriteFile(Matrix M, char* file_name);
void FreeDeviceMatrix(Matrix* M);
void FreeMatrix(Matrix* M);

void MatrixMulOnDevice(const Matrix M, const Matrix N, Matrix P);

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv) {

	Matrix  M;
	Matrix  N;
	Matrix  P;
	int errorM = 0, errorN = 0;
	
	srand(32);
	
	if(argc != 5 && argc != 4) 
	{
		// Allocate and initialize the matrices
		M = AllocateMatrix(rand() % 1024, rand() % 1024, 1);
		N = AllocateMatrix(M.width, rand() % 1024, 1);
		P = AllocateMatrix(M.height, N.width, 0);

	}
	else
	{
		// Allocate and read in matrices from disk
		int* params = NULL; //(int*)malloc(3 * sizeof(int));
		unsigned int data_read = 3;
		cutReadFilei(argv[1], &params, &data_read, true);
		if(data_read != 3){
			printf("Error reading parameter file\n");
			return 1;
		}

		M  = AllocateMatrix(params[0], params[1], 0);
		N  = AllocateMatrix(params[1], params[2], 0);		
		P  = AllocateMatrix(params[0], params[2], 0);
		errorM = ReadFile(&M, argv[2]);
		errorN = ReadFile(&N, argv[3]);
		if(errorM  || errorN )
		{
			printf("Error reading input files %d, %d\n", errorM, errorN);
			return 1;
		}
	}

	// M * N on the device
	MatrixMulOnDevice(M, N, P);
    
	printf("GPU computation complete\n");
	// compute the matrix multiplication on the CPU for comparison
	Matrix reference = AllocateMatrix(P.height, P.width, 0);
	computeGold(reference.elements, M.elements, N.elements, M.height, M.width, N.width);
        
	printf("CPU computation complete\n");

	// in this case check if the result is equivalent to the expected soluion
	// turn it back to 0.001f for the original test!
	CUTBoolean res = cutComparefe(reference.elements, P.elements, P.height*P.width, 0.001f);
	printf("Test %s\n", (1 == res) ? "PASSED" : "FAILED");

	if(argc == 5)
	{
		WriteFile(P, argv[4]);
	}
	else if(argc == 2)
	{
		WriteFile(P, argv[1]);
	}

	// Free matrices
	FreeMatrix(&M);
	FreeMatrix(&N);
	FreeMatrix(&P);

	return 0;
}

////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for CUDA
////////////////////////////////////////////////////////////////////////////////
void MatrixMulOnDevice(const Matrix M, const Matrix N, Matrix P)
{
	// Step 1: Pad M, N, and P to have heights/widths divisible by 32; only way to completely avoid uncoalesced accesses for boundary tiles!
	// NOTE: I would normally make this a helper function, but Vijay (TA and grader) said that he is only going to copy/paste this function
	//       when testing our kernels for correctness. 
	Matrix padM;
	padM.height = (M.height + 32) - (M.height % 32); // Round up to next multiple of 32
	padM.width = (M.width + 32) - (M.width % 32); // Round up to next multiple of 32
	padM.elements = (float *)malloc(padM.height * padM.width * sizeof(float));
	for (int r=0; r<padM.height; r++) {
		for (int c=0; c<padM.width; c++) {
			if (r < M.height && c < M.width) {
				padM.elements[r * padM.width + c] = M.elements[r * M.width + c];
			} else {
				padM.elements[r*padM.width + c] = 0.0;
			}
		}
	}

	Matrix padN;
	padN.height = (N.height + 32) - (N.height % 32); // Round up to next multiple of 32
	padN.width = (N.width + 32) - (N.width % 32);	 // Round up to next multiple of 32
	padN.elements = (float *)malloc(padN.height * padN.width * sizeof(float));
	for (int r = 0; r < padN.height; r++)
	{
		for (int c = 0; c < padN.width; c++)
		{
			if (r < N.height && c < N.width)
			{
				padN.elements[r * padN.width + c] = N.elements[r * N.width + c];
			}
			else
			{
				padN.elements[r * padN.width + c] = 0.0;
			}
		}
	}

	Matrix padP = AllocateMatrix(padM.height, padN.width, 0);

	// Step 2: Load padded versions of M and N to the device
	Matrix Md = AllocateDeviceMatrix(padM);
	CopyToDeviceMatrix(Md, padM);
	Matrix Nd = AllocateDeviceMatrix(padN);
	CopyToDeviceMatrix(Nd, padN);

	// Step 3: Allocate padded P on the device
	Matrix Pd = AllocateDeviceMatrix(padP);
	CopyToDeviceMatrix(Pd, padP); // Clear memory

	// Step 4: Setup the execution configuration
	int numBlocksX = padP.width / 32; // padP.width is guaranteed to be divisible by 32
	int numBlocksY = padP.height / 32; // padP.height is guaranteed to be divisible by 32

	dim3 gridDims(numBlocksX, numBlocksY); // blocks are 32x32, so our grid must have as many blocks fit the entirety of P
	dim3 blockDims(32, 32); // 32x32=1024, which is the limit of threads per SM. 32 is convenient so threads can coalesce.

	// Step 5: Launch the device computation threads. Wait for computation to complete.
	MatrixMulKernel<<<gridDims, blockDims>>>(Md, Nd, Pd);
	cudaDeviceSynchronize();

	// Step 6: Read padded P from the device
	CopyFromDeviceMatrix(padP, Pd); 

	// Step 7: Copy the computed elements of padded P back into the P matrix that 'main' is expecting to compare with its solution.
	for (int r = 0; r < padP.height; r++)
	{
		for (int c = 0; c < padP.width; c++)
		{
			if (r < P.height && c < P.width)
			{
				P.elements[r * P.width + c] = padP.elements[r * padP.width + c];
			}
		}
	}

	// Step 8: Free device matrices
	FreeDeviceMatrix(&Md);
	FreeDeviceMatrix(&Nd);
	FreeDeviceMatrix(&Pd);

	// Step 9: Free padded copies of host matrices
	FreeMatrix(&padM);
	FreeMatrix(&padN);
	FreeMatrix(&padP);
}

// Allocate a device matrix of same size as M.
Matrix AllocateDeviceMatrix(const Matrix M)
{
    Matrix Mdevice = M;
    int size = M.width * M.height * sizeof(float);
    cudaMalloc((void**)&Mdevice.elements, size);
    return Mdevice;
}

// Allocate a device matrix of dimensions height*width
//	If init == 0, initialize to all zeroes.  
//	If init == 1, perform random initialization.
//  If init == 2, initialize matrix parameters, but do not allocate memory 
Matrix AllocateMatrix(int height, int width, int init)
{
    Matrix M;
    M.width = M.pitch = width;
    M.height = height;
    int size = M.width * M.height;
    M.elements = NULL;
    
    // don't allocate memory on option 2
    if(init == 2)
		return M;
		
	M.elements = (float*) malloc(size*sizeof(float));

	for(unsigned int i = 0; i < M.height * M.width; i++)
	{
		M.elements[i] = (init == 0) ? (0.0f) : (rand()*3 / (float)RAND_MAX);
	}
    return M;
}	

// Copy a host matrix to a device matrix.
void CopyToDeviceMatrix(Matrix Mdevice, const Matrix Mhost)
{
    int size = Mhost.width * Mhost.height * sizeof(float);
    Mdevice.height = Mhost.height;
    Mdevice.width = Mhost.width;
    Mdevice.pitch = Mhost.pitch;
    cudaMemcpy(Mdevice.elements, Mhost.elements, size, 
					cudaMemcpyHostToDevice);
}

// Copy a device matrix to a host matrix.
void CopyFromDeviceMatrix(Matrix Mhost, const Matrix Mdevice)
{
    int size = Mdevice.width * Mdevice.height * sizeof(float);
    cudaMemcpy(Mhost.elements, Mdevice.elements, size, 
					cudaMemcpyDeviceToHost);
}

// Free a device matrix.
void FreeDeviceMatrix(Matrix* M)
{
    cudaFree(M->elements);
    M->elements = NULL;
}

// Free a host Matrix
void FreeMatrix(Matrix* M)
{
    free(M->elements);
    M->elements = NULL;
}

// Read a floating point matrix in from file
// Returns zero if the number of elements read is 
//  equals M.height * M.width, and 1 otherwise
int ReadFile(Matrix* M, char* file_name)
{
	unsigned int data_read = M->height*M->width;
	cutReadFilef(file_name, &(M->elements), &data_read, true);
	return (data_read != (M->height * M->width));
}

// Write a 16x16 floating point matrix to file
void WriteFile(Matrix M, char* file_name)
{
	cutWriteFilef(file_name, M.elements, M.width * M.height,
				  0.0001f);
}
