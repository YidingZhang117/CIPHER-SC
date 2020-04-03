import argparse
import os
import torch
import pickle
import torch.nn as nn
import time
import threading
from tqdm import tqdm

from torch_geometric.nn import DataParallel

from gcn_emb import GCN_EMB
from distmult import DM
from dataset import read_train_test_dm
from eval import eval_test_dm, eval_all_dm, get_mean_AUC


def parse_args():
    parser = argparse.ArgumentParser(description="GCN")
    # ----- unique part
    parser.add_argument('--type_name_list', nargs='+', type=str, default=["union_SMH_PG"], help="type name list")
    parser.add_argument('--setting_symbol', type=str, default="union_SMH_PG", help="setting symbol for saving")
    parser.add_argument('--device_id', type=int, default=None)
    parser.add_argument('--sc_add', type=int, default=1)
    parser.add_argument('--emb_dim', type=int, default=100)
    # ----- model part
    parser.add_argument('--gcn_dim', nargs='+', type=int, default=[500,500], help="[100,100], [500, 500]")
    parser.add_argument('--conv_type', type=str, default="chebconv", help="gcnconv, chebconv, etc.")
    parser.add_argument('--dropout_ratio', type=float, default=0.6, help="dropout ratio")
    parser.add_argument('--dm_dim', type=int, default=500, help="output dim for gcn and input dim for distmult")
    parser.add_argument('--bs', type=int, default=256, help="batch size")
    parser.add_argument('--checkpoint_folder', type=str, default="SMH_PG_95", help="checkpoint folder")

    return parser.parse_args()


# global variable
args = parse_args()

each_fpr_list = [[],[],[],[],[]]
each_tpr_list = [[],[],[],[],[]]
each_mean_auc = [[],[],[],[],[]]
each_mean_rank = [[],[],[],[],[]]


# training and evaluation thread function
def test(fold_ind, dataset_dir, type_name, setting_symbol, device_id=None):
    if device_id is not None:
        device_ind = device_id
    else:
        device_ind = (fold_ind - 1) % 4
    device = torch.device(f'cuda:{device_ind}' if torch.cuda.is_available() else 'cpu')
    print(device)
    
    # read dataset
    dataset = torch.load(os.path.join(dataset_dir, f'dataset_fold_{fold_ind}.pt'))
    num_node_features = dataset.num_node_features
    dataset = dataset.to(device)

    print(dataset)

    # read train and test
    with open(os.path.join(dataset_dir, f"train_test_split_fold_{fold_ind}.pkl"), "rb") as f:
        positive_train, positive_test = pickle.load(f)
    with open(os.path.join(dataset_dir, f"../10/negative_train_test_split_fold_{fold_ind}.pkl"), "rb") as f:
        negative_train, negative_test = pickle.load(f)
    # generate data loader
    _, test_loader = read_train_test_dm(positive_train, positive_test, negative_train, negative_test, args.bs)

    # read omim and test (use for eval_all)
    with open(f"preprocess/edgelist_result/union_omim_gene_list.pkl", "rb") as f:
        omim_list, gene_list = pickle.load(f)
    with open(os.path.join(dataset_dir, f"id_conversion_fold_{fold_ind}.pkl"), "rb") as f:
        node_name_to_contiguous_id, contiguous_id_to_node_name = pickle.load(f)
    refine_gene_list = []
    for gene in gene_list:
        if gene in node_name_to_contiguous_id:
            refine_gene_list.append(node_name_to_contiguous_id[gene])
        else:
            refine_gene_list.append(-1)
    
    add_links = []
    if "S" in type_name:
        sc_add_name = "sc_add_topall20.txt"
        with open(os.path.join("preprocess/edgelist_result/disease_gene/", sc_add_name),"r") as f:
            for line in f.read().strip().split("\n"):
                d, g = line.strip().split("\t")
                d, g = "omim:"+str(d), "gene:"+str(g)
                if d in omim_list and g in gene_list:
                    add_links.append([node_name_to_contiguous_id[d], node_name_to_contiguous_id[g]])

    # embedding gcn
    gcn = GCN_EMB(num_node_features, args.gcn_dim, args.conv_type, args.dropout_ratio).to(device)
    # classifier
    classifier = DM(args.dm_dim).to(device)

    # eval on best auc
    gcn.load_state_dict(torch.load(f"checkpoint/{args.checkpoint_folder}/best_auc_gcn_{type_name}_{setting_symbol}_chebconv_fold_{fold_ind}.pth"))
    classifier.load_state_dict(torch.load(f"checkpoint/{args.checkpoint_folder}/best_auc_classifier_{type_name}_{setting_symbol}_chebconv_fold_{fold_ind}.pth"))
    mean_auc, mean_rank, fpr_list, tpr_list = eval_all_dm(gcn, classifier, device, dataset, refine_gene_list,
                                                          positive_train+add_links, positive_test, negative_train)
    print(f"====== \n {type_name}_{setting_symbol}, fold {fold_ind}, eval on best auc \n mean auc is: {round(mean_auc, 4)} \n mean rank is: {round(mean_rank, 4)} \n======")

    # save fold result
    each_mean_auc[fold_ind-1].append(mean_auc)
    each_mean_rank[fold_ind-1].append(mean_rank)
    each_fpr_list[fold_ind-1].extend(fpr_list)
    each_tpr_list[fold_ind-1].extend(tpr_list)


if __name__ == "__main__":
    for type_name in args.type_name_list:
        print(type_name)
        print(args)
        dataset_base_dir = f"dataset/{type_name}/"
        dataset_dir = os.path.join(dataset_base_dir, f"dim_{args.emb_dim}")

        # multi thread
        # because we have four gpus, so first run fold 1 to fold 4, then run fold 5 separately.
        threads = []
        for i in range(4):
            t = threading.Thread(target=test, args=(i+1, dataset_dir, type_name, args.setting_symbol))
            threads.append(t)
        for t in threads:
            t.setDaemon(True)
            t.start()
        for t in threads:
            t.join()

        # run fold 5            
        test(5, dataset_dir, type_name, args.setting_symbol)

        # merge 5fold result
        all_mean_auc = [auc for auc in each_mean_auc]
        all_mean_rank = [rank for rank in each_mean_rank]
        all_fpr_list = [fpr for fpr_list in each_fpr_list for fpr in fpr_list]
        all_tpr_list = [tpr for tpr_list in each_tpr_list for tpr in tpr_list]
        
        # all fold result
        print("===================")
        print(type_name, args.setting_symbol)
        print("all fold result on best auc\n")
        final_mean_auc, _, _ = get_mean_AUC(all_fpr_list, all_tpr_list)
        print("mean auc is ", final_mean_auc)
        print("each auc is ", all_mean_auc)
        print("===================")
