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
// if zero_out is set, the value will be zeroed out after access
// function is able to deal with COO files containing 0 entries
// if entry is not able to be found, the `*not_found` flag will be set to 1, and we return 0
// if we found an entry that is actually 0, we return 0 but the flag is NOT set (it remains 0)
static double locate_matching_entry(COO M, int *row_offset_table, struct coord to_find, char zero_out, char *not_found_flag) {
    
    // check that there is a matching row in this matrix
    int row_offset = row_offset_table[to_find.i];
//    if (row_offset == -1) printf("row of B does not exist\n");
    if (row_offset == -1) {
        *not_found_flag = 1;
        return 0.0f;
    }

//    printf("find matching for %d,%d\n",to_find.i, to_find.j);
    
    const int num_rows = M->m;
    int row_offset_end = -1;
    int k; // the last offset to check (so we know what range the row takes up)
    #pragma acc kernels // ---> we need a way to break out once one thread has found it, or this may be slow
    for (k = (to_find.i+1); k < num_rows; k++) {
        row_offset_end = row_offset_table[k];
        // if not -1 we have found an offset where we should step function
//        printf("row offset: %d, possible offset: %d\n",row_offset, row_offset_end);
        if (row_offset_end != -1) break;
    }
    /* if val is still -1, we should traverse all the way to the end of the list */
    if (row_offset_end == -1)
        row_offset_end = M->NZ;

    /* binary search to find the matching column value */
    struct coord *col_ptr = (struct coord*)bsearch(&to_find, row_offset+(M->coords), row_offset_end-row_offset, sizeof(struct coord), compare_coo_order_cols);
    if (col_ptr == NULL) {
        *not_found_flag = 1;
        return 0.0f;
    }

    // find the offset so we can get at the data value now
    const long index = col_ptr-(M->coords);
    const double result = M->data[index]; // data at the same index coordinates are at, so this is the data value
    if (zero_out) {
        M->data[index] = 0.0f;
    }
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
    
    const int rows_b = B->m;
    
    /* offset table result */
    int *const result = (int*)malloc(rows_b*sizeof(int));
    
    // keep track of row currently being seen and prior rows
    // used to know when we have already filled in offset for a particular row value
    int curr_row, prev_row;
    curr_row = 0;
    prev_row = -1;

//    printf("CREATING ROW OFFSET TABLE\n");
    
    int k;
    #pragma acc kernels
    for (k = 0; k < B->NZ; k++) {
        
        // the row number for this coordinate
        curr_row = B->coords[k].i;

//        printf("k: %d, curr_row: %d, prev_row: %d\n", k, curr_row, prev_row);
        
        // if we have not marked the start of this row already...
        if (curr_row != prev_row) {

            // mark the index of where this row starts
//            printf("SET %d to %d\n",curr_row, k);
            result[curr_row] = k;


            // perform backfill of -1 values if this is not the immediate next index
            // this is because if we have skipped some values in the our result array,
            // they could be filled with garbage data and we want them to be -1 to indicate there is no row for this index
            int backfill = curr_row - prev_row - 1;
            // update difference with how many mem cells we should backfill with -1s
            int d;
            for (d = 1; d <= backfill; d++) {
//                printf("SET %d to -1\n", curr_row - d);
                result[curr_row - d] = -1;
            }

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

//    int itr;
//    for (itr = 0; itr < B->m; itr++) {
//        printf("TO RETURN %d: %d\n", itr, result[itr]);
//    }
    
    return result;
    
}

/* merges multiple partial row COOs into 1 single COO */
// m is the number of rows result will have n is the number of columns result will have 
static void merge_result_rows(int num_rows, int m, int n, COO *coo_list, COO *final) {

    /* final should point to result when we are done */
    *final = NULL;

    /* result is just the first entry (we will append other items to this) */
    COO result = coo_list[0];

    int first_len = result->NZ;

    /* allocate all the memory that we will need (so we pay less of a performance cost later - we don't want to realloc lots of times) */
    int total_items = 0;
    int k;
#pragma acc kernels
    for (k=0; k<num_rows; k++) {
        COO coo = coo_list[k];
        /* null list just means there are no values for this row! */
        if (coo == NULL)
            continue;
        total_items += coo->NZ;
    }

//    printf("merge result rows\n");
    /* allocate everything to be the final sizes that we will need
     * this saves doing a `realloc` for each row we iterate through */
//    printf("total items are: %d\n", total_items);
    result->coords = (struct coord*)realloc(result->coords,total_items*sizeof(struct coord));
    result->data = (double *)realloc(result->data,total_items*sizeof(double));
    result->NZ = total_items;
    result->m = m;
    result->n = n;

    struct coord *coord_ptr;
    double *data_ptr;
    int i;
    int current_mem_offset = 0x0;
    /* this needs to be executed serially */
    for (i = 1; i<num_rows; i++) {
        COO coo = coo_list[i];
        if (coo == NULL) // no list here
            continue;

        /* offset to place value at */
        if (i-1 == 0) {
            current_mem_offset += first_len;
        } else {
            current_mem_offset += coo_list[i-1]->NZ;
        }

//        printf(" ------ MERGING: %d ----- \n", i);
//        printf("NON-ZEROS: %d\n",coo->NZ);
//        printf("OFFSET: %d\n",current_mem_offset);
        
        /* copy coordinates and data into the result to combine them */
        coord_ptr = (result->coords)+current_mem_offset;
//        printf("coord pointer: %p\n",coord_ptr);
        memcpy(coord_ptr,coo->coords,coo->NZ*sizeof(struct coord));
        data_ptr = (result->data)+current_mem_offset;
//        printf("data pointer: %p\n", data_ptr);
        memcpy(data_ptr,coo->data,coo->NZ*sizeof(double));

//        printf("merged so far:\n");
//        print_sparse(result);

        /* free the old, unneeded old list (it is now in `result` so we need it no longer) */
        free(coo->coords);
        free(coo->data);
        
    }

//    printf(" --- RESULT MATRIX ---:\n");
//    print_sparse(result);
    *final = result;

}



/* calculates a result row in the resultant matrix 
   - row_res will allocated by this routine */
static void calculate_result_row(int a_row, COO A, int *a_row_offsets, COO B, int *b_row_offsets, COO *row_res) {

    /* how big this resultant row will be */
    const int num_cols_a = A->n;

    const int nza = A->NZ;
    const int nzb = B->NZ;


    int a_row_offset = a_row_offsets[a_row];
    if (a_row_offset == -1) {
        // if there is no offset for this row of A
        // then there will be no result for this row
        *row_res = NULL;
        return;
    }

    
    // bear in mind this only represents a single row of the resultant matrix
    // this is the values for only one row of this result
    /* FINAL 1 row total (because this is just a result row) */
    /* TEMP columns is num_cols_a (should fix when we remove 0 values) */
    /* TEMP non-zeros num_cols_a (should fix when we remove 0 values) */
    alloc_sparse(1,num_cols_a,num_cols_a,row_res);
    COO row = *row_res;

    int non_zero_elements = 0;

    /* iterate over elements of A in the specified row */
    int b_row, a_col, b_col; // track real row and col positions in the would-be full matrix
    int a_itr, b_itr; // itr variables keep track of positions in the COO files
    /* loop will overshoot, break when needed */
    for (a_itr = 0; a_itr < num_cols_a; a_itr++) {

        // printf("A ITR: %d\n", a_itr);

        /* check we are not shooting past the memory of the A COO */
        if (a_row_offset + a_itr >= A->NZ)
            break;

        /* check we are still in the correct row, otherwise we are done */
        // (do check here so we are more parallelisable)
        if (A->coords[a_row_offset + a_itr].i != a_row)
            break;

        a_col = A->coords[a_row_offset + a_itr].j;

        /* find row of B that corresponds to this column of A */
        int b_row_offset = b_row_offsets[a_col];
        /* if row does not exist, jump to next A column */
        if (b_row_offset == -1)
            continue;


        /* loop will overshoot, break when end of row is reached */
        for (b_itr = 0; b_itr < B->n; b_itr++) {

            /* check we are not shooting past the memory of the B COO */
            if (b_row_offset + b_itr >= B->NZ)
                break;

            b_row = B->coords[b_row_offset + b_itr].i;
            /* check we are still in the correct row of B, otherwise we are done */
            /* (row of b corresponds to the column of A) */
            if (b_row != a_col)
                break;

            /* b_col corresponds to position in output row */
            b_col = B->coords[b_row_offset + b_itr].j;

            /* row and column in context of the full matrix result */
            row->coords[b_col].i = a_row;
            row->coords[b_col].j = b_col;


            double result = A->data[a_row_offset + a_itr] * B->data[b_row_offset + b_itr];

            /* if this is the first value for this column, increment non-zeros! */
            if (result != 0.0 && row->data[b_col] == 0.0) {
                non_zero_elements++;
            }

//            printf("%d,%d adding %.2f\n", a_row, b_col, result);
//            printf("   -> A cell was %.2f\n", A->data[a_row_offset + a_itr]);
//            printf("   -> B cell was %.2f\n", B->data[b_row_offset + b_itr]);
            row->data[b_col] += result;
        }

    }

    /* arrange all elements so they maintain their order but are compressed to the top of the row */
    /* this allows us to free all the empty space taken up by the 0 cells */
    int elem = 0;
    int itr;
    for (itr = 0; itr < num_cols_a; itr++) {
        if (row->data[itr] != 0.0) {
            row->coords[elem] = row->coords[itr];
            row->data[elem] = row->data[itr];
            elem++;
        }
    }

//    printf("    ------->  ROW %d (before) <---------\n", a_row);
//    print_sparse(row);

    /* strip the row down to keep only the memory we need (hopefully much smaller than before if very sparse!) */
    /* this works because all of the values we need have now been pushed to the top of the array, essentially 'squeezing out' the 0s */
    row->NZ = non_zero_elements;
    row->coords = (struct coord*)realloc(row->coords,non_zero_elements*sizeof(struct coord));
    row->data = (double*)realloc(row->data,non_zero_elements*sizeof(double));
    *row_res = row;

    printf("    ------->  ROW %d (after) <---------\n", a_row);
    print_sparse(row);

}


/* performs the sparse matrix multiplication
 */
static void perform_sparse_optimised_multi(const COO A, const COO B, COO *C) {

    const int a_num_rows = A->m;

    /* offsets to easily locate row indices */
    int *a_row_offsets = row_offset_table(A);
    int *b_row_offsets = row_offset_table(B);

    COO *to_merge = (COO *)malloc(a_num_rows*sizeof(COO));

    COO row;
    int k;
    for (k=0;k<a_num_rows;k++) {
//        printf("calculate row: %d\n", k);
        calculate_result_row(k,A,a_row_offsets,B,b_row_offsets,&row);
        to_merge[k] = row;
    }

    /* merge the row results, to get the final matrix C! */
    merge_result_rows(A->m,A->m,B->n,to_merge,C);

//    printf("COO RESULT: %p\n", C);

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
    

    #if SHOULD_PROFILE
    LIKWID_MARKER_START("pre-process-multi");
    #endif

    /* ensure COO files are ordered */
    order_coo_matrix(A);
    order_coo_matrix(B);

    #if SHOULD_PROFILE
    LIKWID_MARKER_STOP("pre-process-multi");
    #endif


    #if SHOULD_PROFILE
    LIKWID_MARKER_START("optimised-multi");
    #endif

    /* now multiply */
    perform_sparse_optimised_multi(A, B, C);


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
static void merge_matrices(COO *A, COO B, int b_uniques) {

    COO added = *A;
    
    // update A values to reflect the merge
    const int old_a_size = added->NZ;
    added->NZ += b_uniques;

    // realloc A so it's large enough to store B's unique entries as well
    added->coords = (struct coord*)realloc(added->coords,added->NZ*sizeof(struct coord));
    added->data = (double*)realloc(added->data,added->NZ*sizeof(double));

    // iterate over B and append all the entries to A
    int k,j;
    j = old_a_size;
    #pragma acc kernels // ---> not sure if this can be vectorised because of dependcy on `j`?
    for (k = 0; k < B->NZ; k++) {
        // do not append if column is 0 - this value has already been added to A
        if (B->data[k] != 0.0) {
            added->data[j] = B->data[k];
            added->coords[j] = B->coords[k];
            j++; // increment A memory location
            printf("added %.2f to (%d,%d)\n", added->data[j-1],added->coords[j-1].i, added->coords[j-1].j);
        }
    }


//    printf("FULLY ADDED MERGED\n");
//    print_sparse(added);

    *A = added;
    
}

/* add the matrices A and B, storing the result in A.
 * we require messing up entries of B (zeroing some out), so we dealloc B after because it is now useless
 * DO NOT USE B AFTER DEALLOCATED
 * time complexity: O(n)
 * space complexity: O(n)
 */
static void add_matrices(COO *A, COO B) {


    COO added = *A;
    
    // the one with more non-zero values should be A
    // this reduces the amount of binary searching and `reallocing` we have to do
    // (since this is only an add operation, order does not matter)
    if (B->NZ > added->NZ) {
        COO tmp = added;
        added = B;
        B = tmp;
    }
    
    /* b row offset table for reference */
    int *b_row_offset_table = row_offset_table(B);
//    printf("TO ADD OFFSET TABLE:\n");
//    int itr;
//    for (itr = 0; itr < B->m; itr++) {
//        printf("READY! %d: %d\n", itr, b_row_offset_table[itr]);
//    }

    
    // go through each line of A
    int k;
    int b_non_uniques = 0; // the number of elements of B for which there is a matching element in A
    char not_found_flag;
    double val;
    #pragma acc kernels
    for (k = 0; k < added->NZ; k++) {
        /* find matching entry in B, then add to A */
//        printf("FINDING %d,%d in B\n",added->coords[k].i,added->coords[k].j);
        not_found_flag = 0;
        val = locate_matching_entry(B,b_row_offset_table,added->coords[k],1,&not_found_flag);
        if (not_found_flag == 0) {
//            printf("ADDING B:%.1f to A:%.1f\n", val, added->data[k]);
            added->data[k] += val;
            b_non_uniques++;
        } else {
            printf("-------> >> >>>> >> NOTHING TO ADD (%d)!!!!!! < ------------\n",k);
        }

    }

    *A = added;

    /* A now contains all common added values, B contains unique values that should be merged */
    const int unique_b_values = B->NZ-b_non_uniques;
    if (unique_b_values != 0) {
        merge_matrices(A, B, unique_b_values);
    }

    printf(" ------> After the merge:");
    print_sparse(added);
    
    // B is now useless
    // (it got messed up bad when we were locating matching add values)
    /* only free the offset table though or the calling function will fail */
    free(b_row_offset_table);
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
    COO a_mut = A;
    COO *abc_added = &a_mut;

//    printf("A to add:\n");
//    print_sparse(A);

    order_coo_matrix(B);
//    printf("B to add:\n");
//    print_sparse(B);
    order_coo_matrix(C);
//    printf("C to add:\n");
//    print_sparse(C);

    printf("Adding A+B\n");
    add_matrices(abc_added,B);
    printf("Adding A+C\n");
    add_matrices(abc_added,C);

    order_coo_matrix(*abc_added);
    printf("ADDED A,B,C :):\n");
//    print_sparse(*abc_added);


    /* CREATE MULT MATRIX D */
    COO d_mut = D;
    COO *def_added = &d_mut;

    order_coo_matrix(E);
    order_coo_matrix(F);
    printf("Adding D+E\n");
    add_matrices(def_added,E);
    printf("Adding D+F\n");
    add_matrices(def_added,F);

    printf("ADDED D,E,F :):\n");
    order_coo_matrix(*def_added);

    // ensure there is no value currently stored at O
    *O = NULL;

    // perform the optimised matrix multiplication operation
    printf("multiplying A*D...\n");
    perform_sparse_optimised_multi(*abc_added, *def_added, O);
    printf("mult done!\n");


    #if SHOULD_PROFILE
    LIKWID_MARKER_STOP("optimised-sum-add");
    LIKWID_MARKER_CLOSE;
    #endif

    
//    return basic_sparsemm_sum(A, B, C, D, E, F, O);
}
