#include "utils.h"
#include <stdlib.h>
#include <stddef.h>
#include <string.h>

#define SHOULD_PROFILE 0

#if SHOULD_PROFILE
#include <likwid.h>
#endif


// use `likwid` on Hamilton in order to measure performance of the routines

// skeleton implementation from other files
// will be able to be seen during compilation
void basic_sparsemm(const COO, const COO, COO *);
void basic_sparsemm_sum(const COO, const COO, const COO,
                        const COO, const COO, const COO,
                        COO *);


/* ordering function for sorting COOs
 * works on a list of pointers to the values, because this is used to sort the pointers
 */
static int order_coo(const void *v1, const void *v2) {
    /* pointer indirection, so we must initially dereference */
    struct coord *c1 = *(struct coord**)v1;
    struct coord *c2 = *(struct coord**)v2;
    int row_comp = c1->i - c2->i;
    if (row_comp != 0) return row_comp;
    // if rows are the same, order by column instead
    return ( c1->j - c2->j );
}

/* compare coordinates FOR BINARY SEARCH ONLY */
static int compare_coo_order_cols(const void *v1, const void *v2) {
    struct coord *c1 = (struct coord*)v1;
    struct coord *c2 = (struct coord*)v2;
    // only compare columns
    // because we know they will be in the same row anyway, so don't waste time
    return ( c1->j - c2->j );
}

/* sorts elements in a COO such that we are ordered by row, then sub-ordered by column
 * pointer array sorted so we can order both the coordinate and data the same way
 * (must be done this way because data and coordinates are stored in separate arrays, and we want them BOTH to be sorted according to the ordering of the coordinates)
 * technique thanks to: https://stackoverflow.com/a/32954558/3261161
 * time complexity: O(nlogn)
 * space complexity: O(n)
 */
static void order_coo_matrix(COO M) {
    
    /* an array that contains pointers to coordinates, these are sorted as a layer of indirection */
    struct coord **pointer_arr = (struct coord**)malloc(M->NZ*sizeof(struct coord*));
    
    /* create array of pointers to coords[] */
    int i;
    #pragma acc kernels
    for(i = 0; i < M->NZ; i++)
        pointer_arr[i] = &(M->coords[i]);
    
    /* sort array of pointers */
    qsort(pointer_arr, M->NZ, sizeof(struct coord *), order_coo);
    
    /* reorder coords[] and data[] according to the array of pointers
       - sorting is done in place to save memory */
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
    
    /* pointer array no longer needed */
    free(pointer_arr);

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
    #pragma acc kernels // ---> we need a way to break out once one thread has found it, or this may be slow
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
    
    /* offset table result */
    int *result = (int*)malloc(rows_b*sizeof(int));
    
    // keep track of row currently being seen and prior rows
    // used to know when we have already filled in offset for a particular row value
    int curr_row, prev_row;
    curr_row = 0;
    prev_row = -1;
    
    int k;
    #pragma acc kernels
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
    #pragma acc kernels
    for (i = curr_row+1; i < rows_b; i++) {
        result[i] = -1;
    }
    
    return result;
    
}

/* merges multiple partial row COOs into 1 single COO */
// m is the number of rows result will have n is the number of columns result will have 
static void merge_result_rows(int m, int n, COO *coo_list, COO result) {

    /* result is just the first entry (we will append other items to this) */
    result = coo_list[0];

    /* allocate all the memory that we will need (so we pay less of a performance cost later - we don't want to realloc lots of times) */
    int total_items = 0;
    int k;
    for (k=0; ; k++) {
        COO coo = coo_list[k];
        if (coo == NULL)
            break;
        total_items += coo->NZ;
    }
    /* allocate everything to be the final sizes that we will need */
    result->coords = (struct coord*)realloc(result->coords,total_items*sizeof(struct coord));
    result->data = (double *)realloc(result->data,total_items*sizeof(double));
    result->NZ = total_items;
    result->m = m;
    result->n = n;

    struct coord *coord_ptr;
    double *data_ptr;
    int i, new_size;
    int current_mem_offset = 0;
    /* this needs to be executed serially */
    for (i = 1; ; i++) {
        COO coo = coo_list[i];
        if (coo == NULL) // end of loop
            break;
        /* reallocate enough memory to fit the combined list */
        current_mem_offset += coo->NZ;
        
        /* copy coordinates and data into the result to combine them */
        coord_ptr = (result->coords)+current_mem_offset;
        memcpy(coord_ptr,coo->coords,coo->NZ*sizeof(struct coord));
        data_ptr = (result->data)+current_mem_offset;
        memcpy(data_ptr,coo->data,coo->NZ*sizeof(double));

        /* free the old, unneeded old list (it is now in `result` so we need it no longer) */
        free(coo->coords);
        free(coo->data);
        free(coo_list[i]);
        
    }

}



/* calculates a result row in the resultant matrix 
   - row will allocated by this routine */
static void calculate_result_row(int a_col, COO A, int *a_row_offsets, COO B, int *b_row_offsets, COO row) {

    const int num_rows_a = A->m;
    const int nza = A->NZ;
    const int nzb = B->NZ;

    int b_offset = b_row_offsets[a_col];
    if (b_offset == -1) {
        // if there is no offset for this row of `b`,
        // there is no row of `b` for this column of `a` to multiply with,
        // so this `a` value is of no use,
        // so skip
        // there will be nothing in this row result and *row will remain NULL
        return;
    }
    
    // bear in mind this only represents a single row of the resultant matrix
    // therefore the `m` and `n` for the size will not be accurate
    alloc_sparse(B->m,A->n,num_rows_a,&row);


    int non_zero_elements = 0;

    /* iterate over elements of A in the specified column */
    double a_val, b_val;
    int b_row, b_col;
    int a_row;
    for (a_row = 0; a_row < num_rows_a; a_row++) {

        if (a_row_offsets[a_row] == -1) {
            // this row does not exist in A, so skip this iteration
            continue;
        }

        // find the specific row/column value for A by binary search
        a_val = locate_matching_entry(A,a_row_offsets,a_row,a_col,0);
        if (a_val == 0.0) {
            // there is no value, we cannot use this A
            continue;
        }

        // we will perform up to `the number of columns of B` iterations - likley less unless a certain row of B is totally filled
        // iterate over all b column values while the row of b matches `a`'s column
        // (just ensures that we don't run past the row of b we are interested in or run over the end of the array)
        int p;
        for (p = 0; p < B->n; p++) {
            
            b_row = B->coords[b_offset+p].i;
            
            if (a_col != b_row || b_offset+p >= nzb) {
                // only continue if current `a` column is equal to the current `b` row
                // and we have not gone into the next column of `b`
                // otherwise, there is nothing else we can do for this given `a` entry
                break;
            }
            
            // once we know to continue, get column val of `b` and the value we will use to multiply
            b_val = B->data[b_offset+p];
            
            // row = row of a
            // column = col of b
            // use a_num_rows because matrix cells are arranged in a column major format
            // (required for proper conversion back to sparse)
            row->coords[a_row].i = a_col;
            row->coords[a_row].j = b_row;
            row->data[a_row] += a_val * b_val;
            non_zero_elements++;
            
        }

    }

    /* arrange all elements so they maintain their order but are compressed to the top of the row */
    /* this allows us to free all the empty space taken up by the 0 cells */
    int elem = 0; 
    int itr;
    for (itr = 0; itr < num_rows_a; itr++) {
        if (row->data[itr] != 0.0) {
            row->coords[elem] = row->coords[itr];
            row->data[elem] = row->data[itr];
            elem++;
        }
    }

    /* strip the row down to keep only the memory we need (hopefully much smaller than before if very sparse!) */
    /* this works because all of the values we need have now been pushed to the top of the array, essentially 'squeezing out' the 0s */
    row->NZ = non_zero_elements;
    row->coords = (struct coord*)realloc(row->coords,non_zero_elements*sizeof(struct coord));
    row->data = (double*)realloc(row->data,non_zero_elements*sizeof(double));

}


/* performs the sparse matrix multiplication
 */
static void perform_sparse_optimised_multi(const COO A, const COO B, COO C) {

    const int a_num_cols = A->n;

    /* offsets to easily locate row indices */
    int *a_row_offsets = row_offset_table(A);
    int *b_row_offsets = row_offset_table(B);

    int merge_allocs_performed = 1;
    COO *to_merge = (COO *)malloc((a_num_cols+1)*sizeof(COO *)); // add 1 for the null terminator

    COO row;
    int c;
    for (c = 0; c < a_num_cols; c++) {
        calculate_result_row(c,A,a_row_offsets,B,b_row_offsets,row);
        to_merge[c] = row;
    }

    /* merge the row results, to get the final matrix C! */
    to_merge[c+1] = NULL; // null terminate the list
    merge_result_rows(A->n,B->m,to_merge,C);

    /* we no longer need the offset tables */
    free(a_row_offsets);
    free(b_row_offsets);
    free(to_merge);
}



/* Computes C = A*B.
 * C should be allocated by this routine.
 *
 */
void optimised_sparsemm(const COO A, const COO B, COO *C) {
    
    #if SHOULD_PROFILE
    LIKWID_MARKER_INIT;
    #endif

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
    
    // ensure the matrix entries are ordered in the way we expect!
    #if SHOULD_PROFILE
    LIKWID_MARKER_START("pre-process-multi");
    #endif
    order_coo_matrix(A);
    order_coo_matrix(B);
    #if SHOULD_PROFILE
    LIKWID_MARKER_STOP("pre-process-multi");
    #endif

    // perform the optimised matrix multiplication operation
    #if SHOULD_PROFILE
    LIKWID_MARKER_START("optimised-multi");
    #endif
    perform_sparse_optimised_multi(A, B, *C);
    #if SHOULD_PROFILE
    LIKWID_MARKER_STOP("optimised-multi");
    LIKWID_MARKER_CLOSE;
    #endif
    
    //return basic_sparsemm(A,B,C);
    
}



/* merge the matrices A and B
 * merging is performed according to the following critereon:
 * - result be A
 * - entries from B we want to remove should have value zeroed out (otherwise duplicate row/cols will appear)
 * - result will not be ordered
 * (called from add matrices)
 * time complexity: O(n)
 * space complexity: O(n)
 */
static void merge_matrices(COO A, COO B, int b_uniques) {
    
    // update A values to reflect the merge
    int old_a_size = A->NZ;
    A->NZ += b_uniques;
    // realloc A so it's large enough to store B's unique entries as well
    A->coords = (struct coord*)realloc(A->coords,A->NZ*sizeof(struct coord));
    A->data = (double*)realloc(A->data,A->NZ*sizeof(double));
    
    // iterate over B and append all the entries to A
    int k,j;
    j = old_a_size;
    #pragma acc kernels // ---> not sure if this can be vectorised because of dependcy on `j`?
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
 * DO NOT USE B AFTER DEALLOCATED
 * time complexity: O(n)
 * space complexity: O(n)
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
    #pragma acc kernels
    for (k = 0; k < A->NZ; k++) {
        a_row = A->coords[k].i;
        a_col = A->coords[k].j;
        // find matching entry in B, then add to A.
        b_val = locate_matching_entry(B,b_row_offset_table,a_row,a_col,1);
        if (b_val != 0.0) b_uniques++;
        A->data[k] += b_val;
    }
    
    /* A now contains all common added values, B contains unique values that should be merged */
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
    
    #if SHOULD_PROFILE
    LIKWID_MARKER_INIT;
    #endif
    
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
    

    #if SHOULD_PROFILE
    LIKWID_MARKER_START("optimised-sum-add");
    #endif

    /* CREATE MULT MATRIX A */
    order_coo_matrix(B);
    add_matrices(A,B);
    order_coo_matrix(C);
    add_matrices(A,C);
    order_coo_matrix(A);
    /* CREATE MULT MATRIX D */
    order_coo_matrix(E);
    add_matrices(D,E);
    order_coo_matrix(F);
    add_matrices(D,F);
    order_coo_matrix(D);
    
    // ensure there is no value currently stored at O
    *O = NULL;
    
    // perform the optimised matrix multiplication operation
    perform_sparse_optimised_multi(A, D, *O);
    #if SHOULD_PROFILE
    LIKWID_MARKER_STOP("optimised-sum-add");
    LIKWID_MARKER_CLOSE;
    #endif
    
    
//    return basic_sparsemm_sum(A, B, C, D, E, F, O);
}
