#!/usr/bin/env bash

NET=$1
k=$2
n=$3
it=$4

c1=0
while [ $c1 -le 2 ]
do
	c2=0
	while [ $c2 -le 2 ]
	do
		c3=0
		while [ $c3 -le 2 ]
		do
			c4=0
			while [ $c4 -le 2 ]
			do	
				c5=0
				while [ $c5 -le 2 ]
				do	
					c6=0
					while [ $c6 -le 2 ]
					do	
						c7=0
						while [ $c7 -le 2 ]
						do	
							python rev2code.py $NET $c1 $c2 $c3 $c4 $c5 $c6 $c7 $it $k $n &
							echo $c1 $c2 $c3 $c4 $c5 $c6 $c7
							(( c7+=1 ))
						done
						(( c6+=1 ))
					done
					(( c5+=1 ))
					# 9
					wait
				done
				(( c4+=1 ))
				#27
				wait
			done
			(( c3+=1 ))
			#81
			#wait;
		done
		(( c2+=1 ))
		#243
		#wait; 
	done
	(( c1+=1 ))
done
