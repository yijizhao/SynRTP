# -*- coding: utf-8 -*-
import sys
import os
import time
cur_path = os.path.abspath(__file__)
current_ws = os.path.dirname(os.path.dirname(os.path.dirname(cur_path)))
sys.path.append(current_ws)
from sklearn.neighbors import KNeighborsRegressor
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from scipy.sparse import SparseEfficiencyWarning
warnings.simplefilter('ignore', SparseEfficiencyWarning)

from sklearn.multioutput import MultiOutputRegressor
import joblib
from utils.util import *
from utils.eval import Metric
from algorithm.knn.Dataset import KNNDataset

def test(eta_model, test_dataset, params):
    import time
    import numpy as np
    from tqdm import tqdm


    test_eta_label = []
    test_label_len = []
    test_eta_pred = []
    

    input_buffer = []      
    meta_data_buffer = []  

    N = params['max_task_num']
    print('Begin to evaluate (Data Collecting Phase)...')

    iter_num = 1000 if params['is_test'] else test_dataset.max_iter
    print("Test iter_num:", iter_num)
    
    pbar = tqdm(total=iter_num)
    index = 0
    while index < iter_num:
        pbar.update(1)
        V, V_reach_mask, start_fea, route_label, label_len, time_label = test_dataset[index]

        T = V_reach_mask.shape[0]

        for t in range(T):
            label_len_t = int(label_len[t])
            if label_len_t == 0:
                continue
            
            mask = (V_reach_mask[t] == False).astype(int).reshape(-1, 1)
            node_feature = V[t]
            node_fea = (node_feature * mask).reshape(-1)
            start_node_fea = start_fea[t]
            
            input_fea = np.concatenate([start_node_fea, node_fea])
            
            input_buffer.append(input_fea)
            
            meta_data_buffer.append({
                'route_label_t': route_label[t, :],
                'label_len_t': label_len_t,
                'eta_label': time_label[t, :]
            })

        index += 1
    pbar.close()

    num_samples = len(input_buffer)
    print(f'\nCollected {num_samples} samples. Starting Batch Prediction...')
    
    if num_samples > 0:
        batch_input = np.array(input_buffer)
        
        start_time = time.perf_counter()
        
        all_preds = eta_model.predict(batch_input)
        
        end_time = time.perf_counter()
        
        total_inference_time = end_time - start_time
        avg_time = (total_inference_time / num_samples) * 1000 # ms
        print(f"Batch Prediction Finished.")
        print(f"Total Inference Time: {total_inference_time:.4f} s")
        print(f"Average Time per Sample: {avg_time:.4f} ms")
        
        for i, pred_vector in enumerate(all_preds):
            meta = meta_data_buffer[i]
            route_label_t = meta['route_label_t']
            label_len_t = meta['label_len_t']
            eta_label = meta['eta_label']
            
            eta_pred_t = np.zeros([N])
            
            for k in range(label_len_t):
                eta_pred_t[k] = pred_vector[route_label_t[k]]

            test_label_len.append(label_len_t)
            test_eta_label.append(eta_label)
            test_eta_pred.append(eta_pred_t)
    else:
        print("No valid samples found for prediction.")

    labels_len = np.array(test_label_len)
    if len(test_eta_label) > 0:
        eta_label = np.concatenate(test_eta_label).reshape(-1, N)
        eta_pred = np.concatenate(test_eta_pred).reshape(-1, N)
    else:
        eta_label = np.zeros((0, N))
        eta_pred = np.zeros((0, N))

    evaluators = [Metric([1, 25])]
    final_evaluator = None
    
    for e in evaluators:
        e.update_eta(labels_len, eta_pred, eta_label)
        print(e.eta_to_str())
        params_save = dict_merge([e.eta_to_dict(), params])
        params_save['eval_min'], params_save['eval_max'] = e.len_range
        save2file(params_save)
        final_evaluator = e

    return final_evaluator



def main(params):
    params['train_path'], params['val_path'], params['test_path'] = get_dataset_path(params)
    train_dataset = KNNDataset(mode='train', params=params)
    test_dataset = KNNDataset(mode='test', params=params)
    xl_file_name = f'xl_{params["model"]}_{params["dataset"]}.txt'

    if params["inference"]:    
        eta_model = MultiOutputRegressor(KNeighborsRegressor(n_neighbors=3))
        eta_model = joblib.load( f'{params["model_path"]}')
    else:


        # construct training data
        input_lst = []
        label_lst = []
        iter_num = 3000 if params['is_test'] else train_dataset.max_iter
        print("Train iter_num:", iter_num)
        index = 0
        while index < iter_num:
            V, V_reach_mask, start_fea, route_label, label_len, time_label = train_dataset[index]

            T = V_reach_mask.shape[0] # init_mask: (T, N)
            for t in range(T):
                eta_label = time_label[t, :]
                route_label_t = route_label[t,:]
                label_len_t = int(label_len[t])
                if label_len_t == 0:
                    continue

                #construct, target, feature, label for each step
                mask = (V_reach_mask[t] == False).astype(int).reshape(-1, 1)
                node_fea = V[t]

                node_fea = (node_fea * mask).reshape(-1)
                start_node_fea = start_fea[t]
                input = np.concatenate([start_node_fea, node_fea])

                label = np.zeros(params['max_task_num'])
                for i in range(label_len_t):
                    label[route_label_t[i]] = eta_label[i] # 

                input_lst.append(input)
                label_lst.append(label)

            index +=1
        print('Begin to train model...')

        eta_model = KNeighborsRegressor(n_neighbors=3)
        eta_model.fit(np.array(input_lst), np.array(label_lst))

        joblib.dump(eta_model, f'{params["dataset"]}_{params["model"]}.pkl')
        eta_model = joblib.load( f'{params["dataset"]}_{params["model"]}.pkl')
    print('model well trained...')


    result_dict = test(eta_model, test_dataset, params)
    print('\n-------------------------------------------------------------')
    print(f'{params["model"]} Evaluation in test:', result_dict.eta_to_str())

    file_name = f'val_{params["model"]}.txt'

    with open(file_name, 'a') as file:
        file.write("\n\n" + params["dataset"] + "\n" + result_dict.eta_to_str())

    # save the result
    params = dict_merge([result_dict.eta_to_dict(), params])
    return params

def save2file(params):
    from utils.util import save2file_meta, ws
    file_name = ws + f'/output/time_prediction/{params["dataset"]}/{params["model"]}.csv'
    head = [
        # data setting
        'dataset', 'min_task_num', 'max_task_num', 'task', 'eval_min', 'eval_max',
        # model parameters
        'model',
        # training set
        'num_epoch', 'batch_size', 'seed', 'is_test', 'log_time',
        # metric result
        'acc_eta@10', 'acc_eta@20', 'acc_eta@30', 'acc_eta@40', 'acc_eta@50', 'acc_eta@60','mae', 'rmse',
    ]
    save2file_meta(params, file_name, head)

def get_params():
    # Training parameters
    from utils.util import get_common_params
    parser = get_common_params()
    args, _ = parser.parse_known_args()
    return args


if __name__ == "__main__":
    cur_path = os.path.abspath(__file__)
    file = os.path.dirname(os.path.dirname(os.path.dirname(cur_path)))

    params = vars(get_params())
    datasets = ['delivery_yt', 'delivery_sh', 'delivery_cq']  # the name of datasets
    args_lst = []
    params["inference"] = False
    params['is_test'] = False
    for dataset in datasets:
        basic_params = dict_merge([params, {'model': 'knn', 'dataset': dataset}])
        args_lst.append(basic_params)
    # note: here you can use parallel running to accelerate the experiment.
    for p in args_lst:
        main(p)
