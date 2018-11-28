#include "utils.h"
#include <stdlib.h>


// use `likwid` on Hamilton in order to measure performance of the routines

// skeleton implementation from other files
// will be able to be seen during compilation
void basic_sparsemm(const COO, const COO, COO *);
void basic_sparsemm_sum(const COO, const COO, const COO,
                        const COO, const COO, const COO,
                        COO *);



// compares 2 COOs, such that we can order by columns, for faster lookup
//int compare_coo_order_cols(const void *v1, const void *v2) {
//    COO coo1 = (COO)v1;
//    COO coo2 = (COO)v2;
//    int col_compare = coo1->coords->j - coo2->coords->i;
//    return col_compare;
//}
//
//
///*
// * Flips the sparse representation to be in the format:
// * jn in vn
// * from:
// * in jn vn
// * this helps to lookup values by column when we are doing the multiplication by another matrix
// * in the current format, only row lookups are efficient due to the way the file is read into memory
// * returns a modifies the contents at the pointer location with the flipped rows and columns
// */
//static void flipped_rows_columns(COO *flip, int num_elems) {
//    qsort(flip, num_elems, sizeof(COO), compare_coo_order_cols);
//}

/*
 * preprocessing to allow for fast lookup of B elements by index
 * returns an array of the offsets of the elements in B, by row.
 * this allows us to easily find a row, because we know it's offset in the COO list
 * this function does the calculation of that.
 * e.g. where i = row num, j = col num and tuples are (i,j,val)
 * [(0,0,999),(0,1,999),(0,2,999),(1,0,999),(1,1,999),(2,0,999),(4,1,999)]
 * we would get:
 * [0,3,5,-1,6]
 * -1 returned if there is no row for that index value
 * here, there is no row `3` in the matrix, index 3 in result is `-1`, indicating no row
 * time complexity: O(n)
 * space complexity: O(n)
 */
static int *first_val_offsets(const COO B, int nzb, int rows_b) {
    
    // where we will store resultant array
    int *result = (int*)malloc(rows_b*sizeof(int));
    
    // keep track of row currently being seen and prior rows
    // used to know when we have already filled in offset for a particular row value
    int curr_row, prev_row;
    curr_row = 0;
    prev_row = -1;
    
    int k;
    for (k = 0; k < nzb; k++) {
        
        // the row number for this coordinate
        curr_row = B->coords[k].i;
        
        // if we have not marked the start of this row already...
        if (curr_row != prev_row) {

            // perform backfill of -1 values if this is not the immediate next index
            // this is because if we have skipped some values in the our result array,
            // they could be filled with garbage data and we want them to be -1 to indicate there is no row for this index
            register int difference = curr_row - prev_row;
            if (difference>1 && prev_row != -1) {
                // update difference with how many mem cells we should backfill with -1s
                difference -= 1;
                int d;
                for (d = 1; d <= difference; d++) {
                    result[curr_row-d] = -1;
                }
            }
            
            // mark the index of where this row starts
            result[curr_row] = k;

            // ensure that we do not update value again, and waste valuable computation
            prev_row = curr_row;
        }
        
    }
    
    // fill any trailing memory cells that we did not reach
    // (because there are no cells in the matrix), with -1s
    int i;
    for (i = curr_row+1; i < rows_b; i++) {
        result[i] = -1;
    }
    
    return result;
    
}


static void perform_sparse_optimised_multi(const COO A, const COO B, double *C) {
    
    // the number of non-zero elements in A and B
    const int nza = A->NZ;
    const int nzb = B->NZ;
    
    const int a_num_rows = A->m; // rows of A
    
    // offsets of row values in the b matrix
    // used to easily locate row values in B
    int *b_row_val_offsets = first_val_offsets(B, nzb, B->m);

    // keep track of current values
    register int a_row, a_col, b_row, b_col;
    register double a_val, b_val;
    
    #pragma acc kernels
    int k;
    for (k = 0; k < nza; k++) {
        
        a_col = A->coords[k].j;
        
        int b_offset = b_row_val_offsets[a_col];
        if (b_offset == -1) {
            // if there is no offset for this row of `b`,
            // there is no row of `b` for this column of `a` to multiply with,
            // so this `a` value is of no use,
            // so skip
            continue;
        }
        
        a_row = A->coords[k].i;
        a_val = A->data[k];
        
        // we will perform up to `the number of rows of B` iterations - likley less unless a certain column of B is totally filled
        // iterate over all b column values while the row of b matches `a`'s column
        // (just ensures that we don't run past the row of b we are interested in or run over the end of the array)
        int p;
        for (p = 0; p < B->m; p++) {
            
            b_row = B->coords[b_offset+p].i;
            
            if (a_col != b_row || b_offset+p >= nzb) {
                // only continue if current `a` column is equal to the current `b` row
                // and we have not gone into the next column of `b`
                // otherwise, there is nothing else we can do for this given `a` entry
                break;
            }
            
            // once we know to continue, get column val of `b` and the value we will use to multiply
            b_col = B->coords[b_offset+p].j;
            b_val = B->data[b_offset+p];
            
            // row = row of a
            // column = col of b
            // use a_num_rows because matrix cells are arranged in a column major format
            // (required for proper conversion back to sparse)
            C[a_num_rows*b_col + a_row] = C[a_num_rows*b_col + a_row] + (a_val * b_val);
            
        }
    }
    
    // free the offsets from memory now we no longer need them!
    free(b_row_val_offsets);
    
}



/* Computes C = A*B.
 * C should be allocated by this routine.
 *
 */
void optimised_sparsemm(const COO A, const COO B, COO *C) {
    
    // idea - keep them all sparse, just try and do it, see how it goes!
    
//    // pointer to the C matrix that we will use to store the result
//    double *c = NULL;
//
//    // m = A rows
//    // n = B columns
//    // k = A columns
//    int m, n, k;
//
//    // ensure there is no value currently stored at C
//    *C = NULL;
//
//    // check that the matrices are compatible sizes
//    m = A->m;
//    k = A->n;
//    n = B->n;
//    if (k != B->m) {
//        fprintf(stderr, "Invalid matrix sizes, got %d x %d and %d x %d\n", A->m, A->n, B->m, B->n);
//        exit(1);
//    }
//
//    // allocate dense, because it could well be the case that every element will be filled after the multiplication
//    alloc_dense(m, n, &c);
//    // zero it out, we don't know if this is guaranteed or not
//    zero_dense(m, n, c);
//
//    // perform the optimised matrix multiplication operation
//    // we pass the
//    perform_sparse_optimised_multi(A, B, c);
//    // as we created C in a dense format, we want to convert the representation back out to the testing suite expects
//    convert_dense_to_sparse(c, m, n, C);
//    free_dense(&c);
    
    return basic_sparsemm(A,B,C);
    
}

/* Computes O = (A + B + C) (D + E + F).
 * O should be allocated by this routine.
 */
void optimised_sparsemm_sum(const COO A, const COO B, const COO C,
                            const COO D, const COO E, const COO F,
                            COO *O) {
    
    
    
    return basic_sparsemm_sum(A, B, C, D, E, F, O);
}
