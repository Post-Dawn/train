#!/bin/bash



for m in 10 20 50 100
do
	for k in 1 2 3; do
		echo 
		echo $k attempt. now testing margin $m
		python main.py \
			--input graph/edgelist.csv \
			--output output/emb_d_200_m_${m}_nbr_1_t_${k} \
			--margin $m \
			--dimension 200 \
			--epoch 24 \
			--pre_epoch 4 \
			--label graph/label.csv \
			--cuda 1 --batch_size 100 \
			--struct_order 1 \
			--prob_nbr 1 \
			--prob_r 0.2
	done
done
