#!/bin/bash
'''
Taking the yt dataset as an example, 
the processing of other datasets is similar; 
simply modify the parameter "--dataset".
'''


echo "start SynRTP yt_dataset..."
python DataPipeline/generate_data_for_SynRTP.py --dataset yt_dataset

echo "start MRGRP yt_dataset..."
python DataPipeline/generate_data_for_MRGRP.py --dataset yt_dataset

echo "start DutyTTE yt_dataset..."
python DataPipeline/generate_data_for_DutyTTE.py --dataset yt_dataset