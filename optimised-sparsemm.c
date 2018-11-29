#include "utils.h"
#include <stdlib.h>

#include <likwid.h>


// use `likwid` on Hamilton in order to measure performance of the routines

// skeleton implementation from other files
// will be able to be seen during compilation
void basic_sparsemm(const COO, const COO, COO *);
void basic_sparsemm_sum(const COO, const COO, const COO,
                        const COO, const COO, const COO,
                        COO *);



static int order_coo(const void *v1, const void *v2) {
    struct coord *c1 = *(struct coord**)v1;
    struct coord *c2 = *(struct coord**)v2;
    int row_comp = c1->i - c2->i;
    if (row_comp != 0) return row_comp;
    // if rows are the same, order by column
    return ( c1->j - c2->j );
}

// compares 2 COOs, such that we can order them for binary search
static int compare_coo_order_cols(const void *v1, const void *v2) {
    struct coord *c1 = (struct coord*)v1;
    struct coord *c2 = (struct coord*)v2;
    // deref the coord and compare the column values
    return ( c1->j - c2->j );
}

/* sorts elements in a COO such that we are ordered by row, then sub-ordered by column
 * pointer array sorted so we can order both the coordinate and data the same way
 * technique thanks to: https://stackoverflow.com/a/32954558/3261161
 */
static void order_coo_matrix(COO M) {
    
    struct coord **pointer_arr = (struct coord**)malloc(M->NZ*sizeof(struct coord*));
    
    /* create array of pointers to coords[] */
    int i;
    for(i = 0; i < M->NZ; i++)
        pointer_arr[i] = &(M->coords[i]);
    
    /* sort array of pointers */
    qsort(pointer_arr, M->NZ, sizeof(struct coord *), order_coo);
    
    /* reorder coords[] and data[] according to the array of pointers */
    struct coord tc;
    double td;
    int k, j;
    for(i = 0; i < M->NZ; i++){
        if(i != pointer_arr[i]-(M->coords)){
            tc = M->coords[i];
            td = M->data[i];
            k = i;
            while(i != (j = pointer_arr[k]-(M->coords))){
                M->coords[k] = M->coords[j];
                M->data[k] = M->data[j];
                pointer_arr[k] = &(M->coords[k]);
                k = j;
            }
            M->coords[k] = tc;
            M->data[k] = td;
            pointer_arr[k] = &(M->coords[k]);
        }
    }

}

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
static int *row_offset_table(const COO B) {
    
    int rows_b = B->m;
    
    // where we will store resultant array
    int *result = (int*)malloc(rows_b*sizeof(int));
    
    // keep track of row currently being seen and prior rows
    // used to know when we have already filled in offset for a particular row value
    int curr_row, prev_row;
    curr_row = 0;
    prev_row = -1;
    
    int k;
    for (k = 0; k < B->NZ; k++) {
        
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
    int *b_row_val_offsets = row_offset_table(B);

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
            C[a_num_rows*b_col + a_row] += a_val * b_val;
            
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
    
    LIKWID_MARKER_INIT;
    
    // pointer to the C matrix that we will use to store the result
    double *c = NULL;

    // m = A rows
    // n = B columns
    // k = A columns
    int m, n, k;

    // ensure there is no value currently stored at C
    *C = NULL;

    // check that the matrices are compatible sizes
    m = A->m;
    k = A->n;
    n = B->n;
    if (k != B->m) {
        fprintf(stderr, "Invalid matrix sizes, got %d x %d and %d x %d\n", A->m, A->n, B->m, B->n);
        exit(1);
    }

    // allocate dense, because it could well be the case that every element will be filled after the multiplication
    alloc_dense(m, n, &c);
    // zero it out, we don't know if this is guaranteed or not
    zero_dense(m, n, c);
    
    // ensure the matrix entries are ordered in the way we expect!
    order_coo_matrix(A);
    order_coo_matrix(B);

    // perform the optimised matrix multiplication operation
    LIKWID_MARKER_START("optimised-multi");
    perform_sparse_optimised_multi(A, B, c);
    LIKWID_MARKER_STOP("optimised-multi");
    // as we created C in a dense format, we want to convert the representation back out to the testing suite expects
    convert_dense_to_sparse(c, m, n, C);
    free_dense(&c);
    
    LIKWID_MARKER_CLOSE;
    
    //return basic_sparsemm(A,B,C);
    
}

// finds a matching entry in the other matrix from the lookup table and then by binary search, returning the value of this element.
// returns 0 if unable to find, therefore no entry exists (no need to add this!)
// if zero_out is set, the value will be zeroed out after access
static double locate_matching_entry(COO M, int *row_offset_table, int row, int col, int zero_out) {
    
    // check that there is a matching row in this matrix
    int row_offset = row_offset_table[row];
    if (row_offset == -1) return 0.0f;
    
    register const int num_rows = M->m;
    register int row_offset_end = -1;
    register int k; // the last offset to check (so we know what range the row takes up)
    for (k = row+1; k < num_rows; k++) {
        row_offset_end = row_offset_table[k];
        // if not -1 we have found an offset where we should step function
        if (row_offset_end != -1) break;
    }
    
    if (row_offset_end == -1) {
        // we should go all the way to the end of the row coordinate list
        // this is because this is the last row
        row_offset_end = M->NZ; // -1 because zero indexed array
    }
    
    // perform a binary search to find the matching column value
    struct coord *col_ptr = (struct coord*)bsearch(&col, &(M->coords[row_offset]), row_offset_end-row_offset, sizeof(struct coord), compare_coo_order_cols);
    if (col_ptr == NULL) return 0.0f;
    
    // find the offset so we can get at the data value now
    int index = (col_ptr-(M->coords))/sizeof(struct coord);
    int result = M->data[index]; // data at the same index coordinates are at, so this is the data value
    if (zero_out) M->data[index] = 0.0f;
    return result;
    
}

/* merge the matrices A and B
 * merging is performed according to the following critereon:
 * - result be A
 * - entries from B we want to remove should have value zeroed out (otherwise duplicate row/cols will appear)
 * - result will not be ordered
 * (called from add matrices)
 */
static void merge_matrices(COO A, COO B, int b_uniques) {
    
    // update A values to reflect the merge
    int old_a_size = A->NZ;
    A->NZ += b_uniques;
    // realloc A so it's large enough to store b's unique entries as well
    A->coords = (struct coord*)realloc(A->coords,A->NZ*sizeof(struct coord));
    A->data = (double*)realloc(A->data,A->NZ*sizeof(double));
    
    // iterate over B and append all the entries to A
    int k,j;
    j = old_a_size;
    for (k = 0; k < B->NZ; k++) {
        // do not append if column is 0 - this value has already been added to A
        if (B->data[k] != 0.0) {
            A->data[j] = B->data[k];
            A->coords[j] = B->coords[k];
            j++; // increment A memory location
        }
    }
    
}

/* add the matrices A and B, storing the result in A.
 * we require messing up entries of B (zeroing some out), so we dealloc B after because it is now useless
 */
static void add_matrices(COO A, COO B) {
    
    // the one with more non-zero values should be A
    // this reduces the amount of binary searching and reallocing we have to do
    // (since this is only an add operation, order does not matter)
    if (B->NZ > A->NZ) {
        COO tmp = A;
        A = B;
        B = tmp;
    }
    
    // to quickly jump to rows in B
    int *b_row_offset_table = row_offset_table(B);
    
    // go through each line of A
    int k, a_row, a_col, b_uniques;
    b_uniques = 0;
    double b_val;
    for (k = 0; k < A->NZ; k++) {
        a_row = A->coords[k].i;
        a_col = A->coords[k].j;
        // find matching entry in B, then add to A.
        b_val = locate_matching_entry(B,b_row_offset_table,a_row,a_col,1);
        if (b_val != 0.0) b_uniques++;
        A->data[k] += b_val;
    }
    
    merge_matrices(A, B, b_uniques);
    
    // B is now useless
    // (it got messed up bad when we were locating matching add values)
    free(b_row_offset_table);
    free(B->data);
    free(B->coords);
    free(B);
    
}

/* Computes O = (A + B + C) (D + E + F).
 * O should be allocated by this routine.
 */
void optimised_sparsemm_sum(const COO A, const COO B, const COO C,
                            const COO D, const COO E, const COO F,
                            COO *O) {
    
    // check that matrices are all compatible sizes
    int i, j, m, n, k;
    
    m = A->m;
    k = A->n;
    n = D->n;
    if (A->m != B->m || A->n != B->n) {
        fprintf(stderr, "A (%d x %d) and B (%d x %d) are not the same shape\n",
                A->m, A->n, B->m, B->n);
        exit(1);
    }
    if (A->m != C->m || A->n != C->n) {
        fprintf(stderr, "A (%d x %d) and C (%d x %d) are not the same shape\n",
                A->m, A->n, C->m, C->n);
        exit(1);
    }
    if (D->m != E->m || D->n != E->n) {
        fprintf(stderr, "D (%d x %d) and E (%d x %d) are not the same shape\n",
                D->m, D->n, E->m, E->n);
        exit(1);
    }
    if (D->m != F->m || D->n != F->n) {
        fprintf(stderr, "D (%d x %d) and F (%d x %d) are not the same shape\n",
                D->m, D->n, F->m, F->n);
        exit(1);
    }
    
    if (A->n != D->m) {
        fprintf(stderr, "Invalid matrix sizes, got %d x %d and %d x %d\n",
                A->m, A->n, D->m, D->n);
        exit(1);
    }
    
    // CREATE MASTER MATRIX A (what we will multiply)
    order_coo_matrix(B);
    add_matrices(A,B);
    order_coo_matrix(C);
    add_matrices(A,C);
    order_coo_matrix(A);
    
    // CREATE MASTER MATRIX D (what we will multiply)
    order_coo_matrix(E);
    add_matrices(D,E);
    order_coo_matrix(F);
    add_matrices(D,F);
    order_coo_matrix(D);
    
    // pointer to the O matrix that we will use to store the result
    double *o = NULL;
    // ensure there is no value currently stored at O
    *O = NULL;
    
    // allocate dense, because it could well be the case that every element will be filled after the multiplication
    alloc_dense(m, n, &o);
    // zero it out, we don't know if this is guaranteed or not
    zero_dense(m, n, o);
    
    // perform the optimised matrix multiplication operation
    LIKWID_MARKER_START("optimised-multi-add");
    perform_sparse_optimised_multi(A, D, o);
    LIKWID_MARKER_STOP("optimised-multi-add");
    // as we created C in a dense format, we want to convert the representation back out to the testing suite expects
    convert_dense_to_sparse(o, m, n, O);
    free_dense(&o);
    
    LIKWID_MARKER_CLOSE;
    
    
//    return basic_sparsemm_sum(A, B, C, D, E, F, O);
}
