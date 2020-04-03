import argparse
import os
import torch
import pickle
import torch.nn as nn
import time
import threading
from tqdm import tqdm

from torch_geometric.nn import DataParallel

from nn import  NN
from inner_product import IP
from gcn_emb import GCN_EMB
from distmult import DM
from dataset import read_train_test_nn, read_train_test_dm
from eval import eval_test_nn, eval_test_dm, eval_all_nn, eval_all_dm, get_mean_AUC


def parse_args():
    parser = argparse.ArgumentParser(description="GCN")
    # ----- unique part
    parser.add_argument('--type_name_list', nargs='+', type=str, default=["union_SMH_PG"], help="type name list")
    parser.add_argument('--setting_symbol', type=str, default="exp", help="setting symbol for saving")
    parser.add_argument('--device_id', type=int, default=None)
    parser.add_argument('--sc_add', type=int, default=1)
    parser.add_argument('--emb_dim', type=int, default=100)
    # ----- model part
    parser.add_argument('--gcn_dim', nargs='+', type=int, default=[500,500], help="[100,100], [500, 500]")
    parser.add_argument('--conv_type', type=str, default="chebconv", help="gcnconv, chebconv, etc.")
    parser.add_argument('--dropout_ratio', type=float, default=0.6, help="dropout ratio")
    parser.add_argument('--classifier', type=str, default='dm', help="nn, dm")
    parser.add_argument('--dm_dim', type=int, default=500, help="output dim for gcn and input dim for distmult")
    parser.add_argument('--nn_dim', nargs='+', type=int, default=[200, 120, 2], help="output dim for gcn and input dim for nn")
    # ----- learning part
    parser.add_argument('--lr', type=float, default=0.01, help="learning rate")
    parser.add_argument('--bs', type=int, default=256, help="batch size")
    parser.add_argument('--epoch_num', type=int, default=100, help="epoch number")
    parser.add_argument('--milestone', nargs='+', type=int, default=[50, 80], help="lr scheduler milestone")

    return parser.parse_args()


# global variable
args = parse_args()

each_fpr_list_on_last = [[],[],[],[],[]]
each_tpr_list_on_last = [[],[],[],[],[]]
each_mean_auc_on_last = [[],[],[],[],[]]
each_mean_rank_on_last = [[],[],[],[],[]]

each_fpr_list = [[],[],[],[],[]]
each_tpr_list = [[],[],[],[],[]]
each_mean_auc = [[],[],[],[],[]]
each_mean_rank = [[],[],[],[],[]]


# training and evaluation thread function
def train(fold_ind, dataset_dir, type_name, setting_symbol, device_id=None):
    if device_id is not None:
        device_ind = device_id
    else:
        device_ind = (fold_ind - 1) % 4
    device = torch.device(f'cuda:{device_ind}' if torch.cuda.is_available() else 'cpu')
    print(device)
    
    # loss
    if args.classifier == "nn":
        crit = nn.NLLLoss()
    elif args.classifier == "dm" or args.classifier == "ip":
        crit = nn.BCELoss()
    
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
    if args.classifier == "nn":
        train_loader, test_loader = read_train_test_nn(positive_train, positive_test, negative_train, negative_test, args.bs)
    elif args.classifier == "dm" or args.classifier == "ip":
        train_loader, test_loader = read_train_test_dm(positive_train, positive_test, negative_train, negative_test, args.bs)

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
    if args.classifier == "nn":
        classifier = NN(args.nn_dim, device).to(device)
    elif args.classifier == "dm":
        classifier = DM(args.dm_dim).to(device)
    elif args.classifier == "ip":
        classifier = IP().to(device)
    
    params = list(gcn.parameters()) + list(classifier.parameters())
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestone, gamma=0.1)

    # train
    best_auc = 0.0
    tqdm_train = tqdm(range(args.epoch_num))
    for epoch in tqdm_train:
        t1 = time.time()
        gcn.train()
        classifier.train()
        epoch_loss = 0.0
        count = 0
        # calculate loss based on train and test split
        delta_time = 0.0
        for batch_ind, batch_target in train_loader:
            optimizer.zero_grad()
            batch_target = batch_target.to(device)
            # forward
            emb = gcn(dataset)
            # distmult
            batch_output = classifier(emb, batch_ind)
            # loss
            loss = crit(batch_output, batch_target)
            epoch_loss += loss.item() * batch_target.size(0)
            count += batch_target.size(0)
            loss.backward()
            optimizer.step()

        epoch_loss /= count

        # eval acc
        if args.classifier == "nn":
            acc, test_loss, roc_auc = eval_test_nn(gcn, classifier, test_loader, device, dataset, crit)
        elif args.classifier == "dm" or args.classifier == "ip":
            acc, test_loss, roc_auc = eval_test_dm(gcn, classifier, test_loader, device, dataset, crit)

        tqdm_train.set_description(f"fold {fold_ind}, epoch {epoch+1}, average loss: {round(epoch_loss, 4)}, test accuracy: {round(acc, 4)} , roc auc: {round(roc_auc, 4)} , test loss: {round(test_loss, 4)}")

        if roc_auc >= best_auc:
            best_auc = roc_auc
            torch.save(gcn.state_dict(), f"best_auc_gcn_{type_name}_{setting_symbol}_fold_{fold_ind}.pth")
            torch.save(classifier.state_dict(), f"best_auc_classifier_{type_name}_{setting_symbol}_fold_{fold_ind}.pth")

        scheduler.step()
    
    # final evaluation
    # eval on last
    if args.classifier == "nn":
        mean_auc, mean_rank, fpr_list, tpr_list = eval_all_nn(gcn, classifier, device, dataset, refine_gene_list,
                                                                positive_train+add_links, positive_test, negative_train)
    elif args.classifier == "dm" or args.classifier == "ip":
        mean_auc, mean_rank, fpr_list, tpr_list = eval_all_dm(gcn, classifier, device, dataset, refine_gene_list,
                                                                positive_train+add_links, positive_test, negative_train)
    print(f"====== \n {type_name}_{setting_symbol}, fold {fold_ind}, eval on last \n mean auc is: {round(mean_auc, 4)} \n mean rank is: {round(mean_rank, 4)} \n======")
    torch.save(gcn.state_dict(), f"last_gcn_{type_name}_{setting_symbol}_fold_{fold_ind}.pth")
    torch.save(classifier.state_dict(), f"last_classifier_{type_name}_{setting_symbol}_fold_{fold_ind}.pth")

    # save fold result
    each_mean_auc_on_last[fold_ind-1].append(mean_auc)
    each_mean_rank_on_last[fold_ind-1].append(mean_rank)
    each_fpr_list_on_last[fold_ind-1].extend(fpr_list)
    each_tpr_list_on_last[fold_ind-1].extend(tpr_list)

    # eval on best auc
    gcn.load_state_dict(torch.load(f"best_auc_gcn_{type_name}_{setting_symbol}_fold_{fold_ind}.pth"))
    classifier.load_state_dict(torch.load(f"best_auc_classifier_{type_name}_{setting_symbol}_fold_{fold_ind}.pth"))
    if args.classifier == "nn":
        mean_auc, mean_rank, fpr_list, tpr_list = eval_all_nn(gcn, classifier, device, dataset, refine_gene_list,
                                                                positive_train+add_links, positive_test, negative_train)
    elif args.classifier == "dm" or args.classifier == "ip":
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
            t = threading.Thread(target=train, args=(i+1, dataset_dir, type_name, args.setting_symbol))
            threads.append(t)
        for t in threads:
            t.setDaemon(True)
            t.start()
        for t in threads:
            t.join()

        # run fold 5            
        train(5, dataset_dir, type_name, args.setting_symbol)

        # merge 5fold result
        all_mean_auc = [auc for auc in each_mean_auc]
        all_mean_rank = [rank for rank in each_mean_rank]
        all_fpr_list = [fpr for fpr_list in each_fpr_list for fpr in fpr_list]
        all_tpr_list = [tpr for tpr_list in each_tpr_list for tpr in tpr_list]

        all_mean_auc_on_last = [auc for auc in each_mean_auc_on_last]
        all_mean_rank_on_last = [rank for rank in each_mean_rank_on_last]
        all_fpr_list_on_last = [fpr for fpr_list in each_fpr_list_on_last for fpr in fpr_list]
        all_tpr_list_on_last = [tpr for tpr_list in each_tpr_list_on_last for tpr in tpr_list]
        
        # all fold result
        print("===================")
        print(type_name, args.setting_symbol)
        print("all fold result on best auc\n")
        final_mean_auc, _, _ = get_mean_AUC(all_fpr_list, all_tpr_list)
        print("mean auc is ", final_mean_auc)
        print("each auc is ", all_mean_auc)

        print()

        print("all fold result on last\n")
        final_mean_auc_on_last, _, _ = get_mean_AUC(all_fpr_list_on_last, all_tpr_list_on_last)
        print("mean auc is ", final_mean_auc_on_last)
        print("each auc is ", all_mean_auc_on_last)

        print("===================")

        each_fpr_list = [[],[],[],[],[]]
        each_tpr_list = [[],[],[],[],[]]
        each_mean_auc = [[],[],[],[],[]]
        each_mean_rank = [[],[],[],[],[]]
        
        each_fpr_list_on_last = [[],[],[],[],[]]
        each_tpr_list_on_last = [[],[],[],[],[]]
        each_mean_auc_on_last = [[],[],[],[],[]]
        each_mean_rank_on_last = [[],[],[],[],[]]
