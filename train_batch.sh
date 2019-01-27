#!/bin/bash
#datasets=(agent gps)
datasets=(agent)
margin=(1 5 10 30 50 100)
#margin=(200 1000)
for data in ${datasets[@]}; do
	for m in ${margin[@]}; do
		echo dataset: $data
		echo first, semisupervised. margin=$m
		input_file=data/$data/edgelist.csv
		output_file=output/$data/dw_margin_${m}_${data}_dim_20_rneg_walk_10
		label_file=data/$data/label_emb.csv
		echo edgelist path: $input_file
		echo label path: $label_file
		echo output path: $output_file
		python main.py $input_file $output_file $label_file --margin $m
	done
	'''
	output_file=output/$data/dw_${data}_dim_20
	echo then, unsupervised.
	echo edgelist path: $input_file
	echo output path: $output_file
	python main.py $input_file $output_file
	'''
done
