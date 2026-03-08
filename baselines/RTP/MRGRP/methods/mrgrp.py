# -*- coding: utf-8 -*-
import os
import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import ExponentialLR
import models 
from utils.functions import EarlyStopping, define_loss 
from utils.eval import Metric 
from data.RPdataset import RPDataset 
import matplotlib.pyplot as plt 
from tqdm import tqdm 
import time

def print_m(*args, **kwargs):
    kwargs.setdefault('flush', True)
    print(*args, **kwargs)

def mrgrp_train(args, flags, T, model_path, file_name, is_print=False, random_seed=729):

    is_test = False
    log_dir = os.path.join(args.log_dir, f'seed{random_seed}') 
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    
    context_c_sizes = [flags.dict_length_day_of_week_c + 1, flags.dict_length_quarter_of_day_c,
                        flags.dict_length_tc_manage_type_c + 1]
    wb_c_sizes = [flags.dict_length_is_new_wb_c, flags.dict_length_is_prebook_c,
                flags.dict_length_delivery_service_c + 1, flags.dict_length_busi_source_c,
                flags.dict_length_time_type_c]
    pickup_c_sizes = [flags.dict_length_poi_familiarity_c + 1, flags.dict_length_poi_type_c,
                    flags.dict_length_pushmeal_before_dispatch_c, flags.dict_length_wifi_arrive_shop_c,
                    flags.dict_length_late_meal_report_c]

    train_dataset = RPDataset(args, flags, mode='train')
    # batch size 1024
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    val_dataset = RPDataset(args, flags, mode='val')
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    

    model = models.MRGRP(context_c_sizes, wb_c_sizes, pickup_c_sizes,flags, args)
    model = model.to(args.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=flags.learning_rate)
    scheduler = ExponentialLR(optimizer, gamma=0.9) 
    graph_size = 25

    train_loss_list = [] 
    val_loss_list = [] 
    val_acc_list = [] 
    
    os.makedirs(os.path.join(log_dir, 'checkpoints'), exist_ok=True) 
    model_save_path = os.path.join(log_dir, 'checkpoints')

    with open(file_name, 'a', encoding='utf-8') as file:
        file.write(f'\n\n{T}\n')

    es = EarlyStopping(patience=11, verbose=True, path=model_path, is_print=is_print)

    for epoch in range(args.epochs):
        model.train() 
        train_loss = 0
        val_loss = 0

        if is_print:
            train_loader = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}") 

        for i, features_all in enumerate(train_loader):
            features_all = features_all.to(args.device) 
            batch_size = features_all.shape[0]
            labels = features_all[:,1300:1425].view(batch_size, graph_size, 5).long() 
            result_final = model(features_all) 
            
            loss = define_loss(labels, result_final, flags, args) 
            optimizer.zero_grad() 
            loss.backward() 
            optimizer.step() 

            if is_print:
                train_loader.set_description(f'Epoch[{epoch}/{args.epochs}] | train-mse={loss:.5f}')
            
            train_loss += loss.item() 
        train_loss_list.append(train_loss/len(train_loader)) 

        scheduler.step() 


        model.eval()
        with torch.no_grad(): 
            graph_size = 25
            evaluators = Metric([1, 25])  
            if is_print:
                val_loader = tqdm(val_loader, desc=f"Validation Epoch {epoch}/{args.epochs}") 
            
            for i, (features_all) in enumerate(val_loader):
                features_all = features_all.to(args.device)
                batch_size = features_all.shape[0]
                labels = features_all[:,1300:1425].view(batch_size, graph_size, 5).long()
                geohash_dist_mat = features_all[:,2157:2886].view(batch_size, graph_size + 2, graph_size + 2)
                result_final = model(features_all)
                
                loss = define_loss(labels, result_final,flags, args) 
                evaluators.update_route_eta(result_final)  
                val_loss += loss.item() 

            metrics = evaluators.route_eta_to_str()  

            val_loss_list.append(val_loss/len(val_loader)) 
            val_acc_list.append( (round(evaluators.to_dict()["krc"], 4))) 

            es((round(evaluators.to_dict()["krc"], 4)), model) 
            if es.early_stop:
                if is_print:
                    print_m("Early stopping") 
                break 

            with open(file_name, "a", encoding="utf-8") as file:
                file.write(
                    f"Epoch {epoch}: {metrics} | Best krc: {es.best_score} | del: {evaluators.del_count}\n"
                )



            
def mrgrp_test(args, flags, T, model_path, file_name, is_print=False, random_seed=729):
    is_test = True

    log_dir = os.path.join(args.log_dir, f'seed{random_seed}')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    model_save_path = os.path.join(log_dir, 'checkpoints')

    context_c_sizes = [flags.dict_length_day_of_week_c + 1, flags.dict_length_quarter_of_day_c,
                        flags.dict_length_tc_manage_type_c + 1]
    wb_c_sizes = [flags.dict_length_is_new_wb_c, flags.dict_length_is_prebook_c,
                flags.dict_length_delivery_service_c + 1, flags.dict_length_busi_source_c,
                flags.dict_length_time_type_c]
    pickup_c_sizes = [flags.dict_length_poi_familiarity_c + 1, flags.dict_length_poi_type_c,
                    flags.dict_length_pushmeal_before_dispatch_c, flags.dict_length_wifi_arrive_shop_c,
                    flags.dict_length_late_meal_report_c]
    
    model = models.MRGRP(context_c_sizes, wb_c_sizes, pickup_c_sizes,flags, args)
    model = model.to(args.device)

    try:
        model.load_state_dict(torch.load(model_path)) 
    except:
        if is_print:
            print_m("Warning: No model checkpoint found, use initialized models......") 
            exit(0)

    test_dataset = RPDataset(args, flags, mode='test')
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model.eval() 
    test_loss = 0

    if is_print:
        test_loader = tqdm(test_loader, desc="Testing") 

    with torch.no_grad(): 
        if is_print:
            print_m("Testing...")
        graph_size = 25
        evaluators = Metric([1, 25])  

        for i, features_all in enumerate(test_loader):
            features_all = features_all.to(args.device)
            batch_size = features_all.shape[0]
            labels = features_all[:,1300:1425].view(batch_size, graph_size, 5).long()
            geohash_dist_mat = features_all[:,2157:2886].view(batch_size, graph_size + 2, graph_size + 2)
            result_final = model(features_all)
            
            loss = define_loss(labels, result_final, flags, args) 
            test_loss += loss.item() 
            evaluators.update_route_eta(result_final)  


        test_loss = test_loss/len(test_loader) 
        metrics = evaluators.route_eta_to_str()  
        print("\nEvaluation in test:", metrics)
        with open(file_name, "a", encoding="utf-8") as file:
            file.write(f"Test Result: {metrics} | del: {evaluators.del_count}\n")



