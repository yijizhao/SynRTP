import torch
from loader.dataset import TrajFastDataset
from utils.argparser import get_argparser
import os
import argparse
from datetime import datetime
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'
os.environ['CRYPTOGRAPHY_OPENSSL_NO_LEGACY'] = '1'

def dict_merge(dict_list = []):
    dict_ =  {}
    for dic in dict_list:
        assert isinstance(dic, dict), "object is not a dict!"
        dict_ = {**dict_, **dic}
    return dict_


if __name__ == "__main__":
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    parser = get_argparser()

    args, _ = parser.parse_known_args()
    params = vars(args)
    datasets = [args.dataset_name]  # the name of datasets
    args_lst = []

    for dataset in datasets:
        basic_params = dict_merge([params, {'dataset_name': dataset}])
        args_lst.append(basic_params)    

    for p in args_lst:
        args = argparse.Namespace(**p)

        save_time = f'checkpoint_t{datetime.now().strftime("%Y_%m_%d_%H_%M_%S")}'

        if args.device == "default":
            device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(args.device)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(device)

        if args.method == 'MoEUQ':
            from uncertainty_quantification.trainer import Trainer
            from uncertainty_quantification.MoEUQ import MoEUQ_network
            model = MoEUQ_network(args).to(device)
            suffix = "cd"
            trainer = Trainer(model, device, args)
            trainer.generated_path_eta_uq(args)
