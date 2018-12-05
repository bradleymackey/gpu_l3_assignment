#!/bin/bash

MATRIX_DIR="./matrix/small"

tmp_file_name="OUT_tmp.bm"
matrix_ext=".matrix"
files="$MATRIX_DIR"/*

shuffle() {
   local i tmp size max rand

   # $RANDOM % (i+1) is biased because of the limited range of $RANDOM
   # Compensate by using a range which is a multiple of the array size.
   size=${#files[*]}
   max=$(( 32768 / size * size ))

   for ((i=size-1; i>0; i--)); do
      while (( (rand=$RANDOM) >= max )); do :; done
      rand=$(( rand % (i+1) ))
      tmp=${files[i]} files[i]=${files[rand]} files[rand]=$tmp
   done
}

shuffle

echo the files: $files

echo "---> FULL SAME MULT <---"
for entry1 in $files
do
	if [[ $entry1 != *.bm ]] ;
	then 
		continue
	fi

	# get the head from the associated matrix
	MAT_FILE="$(echo $entry1 | cut -d'.' -f 2)"
	MAT_FILE=".$MAT_FILE$matrix_ext"
	HEAD="$(echo head -1 $MAT_FILE)"
	COLS="$($HEAD | cut -d' ' -f 2)"

	shuffle

	for entry2 in $files
	do

		if [[ $entry2 != *.bm ]] ;
		then 
			continue
		fi

		# get the head from the associated matrix
		MAT_FILE="$(echo $entry2 | cut -d'.' -f 2)"
		MAT_FILE=".$MAT_FILE$matrix_ext"
		HEAD="$(echo head -1 $MAT_FILE)"
		ROWS="$($HEAD | cut -d' ' -f 1)"
		if [[ $ROWS == $COLS ]] ;
		then
			echo "MULT $entry1 $entry2"
		  ./sparsemm --binary $tmp_file_name $entry1 $entry2
		  python3 check.py $entry1 $entry2 $tmp_file_name
		fi

	done
		
	
done

shuffle

echo "---> FULL SAME ADD MULT <---"
for entry1 in $files
do

	if [[ $entry1 != *.bm ]] ;
	then 
		continue
	fi


	# get the head from the associated matrix
	MAT_FILE="$(echo $entry1 | cut -d'.' -f 2)"
	MAT_FILE=".$MAT_FILE$matrix_ext"
	HEAD="$(echo head -1 $MAT_FILE)"
	ROWS_E1="$($HEAD | cut -d' ' -f 1)"
	COLS="$($HEAD | cut -d' ' -f 2)"

	shuffle

	for entry2 in $files
	do

		if [[ $entry2 != *.bm ]] ;
		then 
			continue
		fi

		# get the head from the associated matrix
		MAT_FILE="$(echo $entry2 | cut -d'.' -f 2)"
		MAT_FILE=".$MAT_FILE$matrix_ext"
		HEAD="$(echo head -1 $MAT_FILE)"
		ROWS="$($HEAD | cut -d' ' -f 1)"
		COLS_E2="$($HEAD | cut -d' ' -f 2)"
		if [[ $ROWS == $COLS && $ROWS_E1 == $COLS_E2 ]] ;
		then
			echo "MULT ($entry1 + $entry2 + $entry1) ($entry2 + $entry1 + $entry2)"
		  ./sparsemm --binary $tmp_file_name $entry1 $entry2 $entry1 $entry2 $entry1 $entry2
		  python3 check-sum.py $entry1 $entry2 $entry1 $entry2 $entry1 $entry2 $tmp_file_name
		fi

	done
	
done

rm $tmp_file_name