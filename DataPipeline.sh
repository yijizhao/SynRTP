#!/bin/bash
'''
Taking the cq dataset as an example, 
the processing of other datasets is similar; 
simply modify the parameter "--dataset".
'''


echo "start SynRTP cq_dataset..."
python DataPipeline/generate_data_for_SynRTP.py --dataset cq_dataset

echo "start MRGRP cq_dataset..."
python DataPipeline/generate_data_for_MRGRP.py --dataset cq_dataset

echo "start DutyTTE cq_dataset..."
python DataPipeline/generate_data_for_DutyTTE.py --dataset cq_dataset