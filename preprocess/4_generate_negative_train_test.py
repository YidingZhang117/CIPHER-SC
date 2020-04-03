# -*- encoding=utf-8
import os
import numpy as np
import pickle
import time
import random


negative_ratio = 10
type_name_list = ["union_MH_PG", "union_SMH_PG"]


for type_name in type_name_list:
    save_dir = f"../dataset/{type_name}/dim_100/"
    print(type_name)

    for fold_ind in range(1, 6):
        print(f"run {fold_ind}")

        # --------- read preprocessed files ----------
        # read train test split
        with open(os.path.join(save_dir, f"train_test_split_fold_{fold_ind}.pkl"), "rb") as f:
            refine_train, refine_test = pickle.load(f)
        # read omim and test list
        with open(os.path.join(save_dir, f"omim_gene_fold_{fold_ind}.pkl"), "rb") as f:
            omim_list, gene_list = pickle.load(f)
        # read traslate dict
        with open(os.path.join(save_dir, f"id_conversion_fold_{fold_ind}.pkl"), "rb") as f:
            node_name_to_contiguous_id, contiguous_id_to_node_name = pickle.load(f)
        # -------------------------------------------

        # positive
        true_list_dict = {}
        for omim, gene in refine_train:
            omim, gene = str(omim), str(gene)
            true_list_dict[omim+"#"+gene] = 1
        for omim, gene in refine_test:
            omim, gene = str(omim), str(gene)
            true_list_dict[omim+"#"+gene] = 1


        if "S" in type_name:
            with open("edgelist_result/disease_gene/sc_add_topall20.txt","r") as f:
                for line in f.read().split("\n"):
                    d, g = line.strip().split()
                    d, g = "omim:"+str(d), "gene:"+str(g)
                    if d in node_name_to_contiguous_id and g in node_name_to_contiguous_id:
                        d = node_name_to_contiguous_id[d]
                        g = node_name_to_contiguous_id[g]
                        if d in omim_list and g in gene_list:
                            true_list_dict[str(d)+"#"+str(g)] = 1

        
        # shuffle the negative choice
        count = 0
        pair_list = []
        t1 = time.time()
        for omim in omim_list:
            omim = str(omim)
            for gene in gene_list:
                gene = str(gene)
                full_name = omim+"#"+gene
                if full_name in true_list_dict:
                    continue
                pair_list.append(full_name)
                count += 1
                if count % 10000000 == 0:
                    t2 = time.time()
                    print(t2-t1, count)
                    t1 = time.time()
        print(len(pair_list))
        random.shuffle(pair_list)

        # save as train and test
        negative_train_number = negative_ratio * len(refine_train)
        negative_number = negative_train_number + len(refine_test)
        negative_train = []
        negative_test = []
        count = 0
        for pair in pair_list:
            omim, gene = pair.split("#")
            omim, gene = int(omim), int(gene)
            if count < negative_train_number:
                negative_train.append([omim, gene])
            elif count < negative_number:
                negative_test.append([omim, gene])
            count += 1
        print("done")

        # save to file
        if not os.path.exists(os.path.join(save_dir, f"../10")):
            os.mkdir(os.path.join(save_dir, f"../10"))
        with open(os.path.join(save_dir, f"../10/negative_train_test_split_fold_{fold_ind}.pkl"), "wb") as f:
            pickle.dump([negative_train, negative_test], f)

