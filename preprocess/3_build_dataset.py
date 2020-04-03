import os
import pickle
import torch
from tqdm import tqdm
from torch_geometric.data import Data


type_name_list = ["union_MH_PG", "union_SMH_PG"]

for type_name in type_name_list:
    for dim in [100]:
        save_dir = f"../dataset/{type_name}/dim_{dim}"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        for fold_ind in range(1, 6):
            print("==========")
            print("process fold", fold_ind)
            # -------- read edge ---------
            print("read edge")
            node_list = []
            edge_list = []
            omim_list = []
            gene_list = []
            with open(f"edgelist_result/{type_name}/raw_link_edge_list_result_fold_{fold_ind}_morethan2.txt", "r") as f:
                content = f.read().split("\n")
                for l in tqdm(content):
                    name1, name2 = l.split("\t")
                    # name1
                    if name1.startswith("omim:"):
                        omim_list.append(name1)
                    elif name1.startswith("gene:"):
                        gene_list.append(name1)
                    # name2
                    if name2.startswith("omim:"):
                        omim_list.append(name2)
                    elif name2.startswith("gene:"):
                        gene_list.append(name2)

                    node_list.append(name1)
                    node_list.append(name2)
                    edge_list.append([name1, name2])
                    edge_list.append([name2, name1])
            # all unique sorted node list
            node_list = list(set(node_list))
            node_list = sorted(node_list)
            # omim_list and gene list
            omim_list = list(set(omim_list))
            gene_list = list(set(gene_list))
            # id-id conversion
            node_name_to_contiguous_id = { v: i for i, v in enumerate(node_list) }
            contiguous_id_to_node_name = { v: k for k, v in node_name_to_contiguous_id.items()}
            # ----------------------------

            # ---- read node feature -----
            print("read node features")
            node_features = {}
            with open(f"name_embedding/{type_name}/{type_name}_name_embedding_fold_{fold_ind}_morethan2_{dim}.pkl", "rb") as f:
                node_features = pickle.load(f)
            
            x = []
            for k in node_list:
                x.append(node_features[k])
            x = torch.tensor(x, dtype=torch.float)
            # ----------------------------

            # ---- read train test link -----
            with open(f"edgelist_result/disease_gene/train_test_association_morethan2_union.pkl", "rb") as f:
                _, test = pickle.load(f)[fold_ind-1]
            test_set_dict = {str(omim)+"#"+str(gene): 1 for omim, gene in test}
            train = []
            with open(f"edgelist_result/disease_gene/union_links.pkl", "rb") as f:
                content = pickle.load(f)
                for omim, gene in content:
                    if str(omim) + "#" + str(gene) in test_set_dict:
                        continue
                    train.append([omim, gene])


            if "S" in type_name:
                with open("edgelist_result/disease_gene/sc_add_topall20.txt","r") as f:
                    for line in f.read().split("\n"):
                        d, g = line.strip().split()
                        if "omim:"+str(d) in omim_list and "gene:"+str(g) in gene_list:
                            train.append([d, g])
                            

            # refine train and test if any node not appeared before
            refine_train = []
            refine_test = []
            for omim, gene in train:
                omim, gene = "omim:"+str(omim), "gene:"+str(gene)
                if omim in node_list and gene in node_list:
                    refine_train.append([omim, gene])
            for omim, gene in test:
                omim, gene = "omim:"+str(omim), "gene:"+str(gene)
                if omim in node_list and gene in node_list:
                    refine_test.append([omim, gene])
            # -------------------------------

            # ------ translate to contiguous id -------
            print("translate to contiguous id")
            # translate edge list
            for i in range(len(edge_list)):
                a, b = edge_list[i]
                edge_list[i][0] = node_name_to_contiguous_id[a]
                edge_list[i][1] = node_name_to_contiguous_id[b]
            # translate train and test
            for i in range(len(refine_train)):
                id1, id2 = refine_train[i]
                refine_train[i][0] = node_name_to_contiguous_id[id1]
                refine_train[i][1] = node_name_to_contiguous_id[id2]
            for i in range(len(refine_test)):
                id1, id2 = refine_test[i]
                refine_test[i][0] = node_name_to_contiguous_id[id1]
                refine_test[i][1] = node_name_to_contiguous_id[id2]
            # translate omim and gene list
            for i in range(len(omim_list)):
                omim = omim_list[i]
                omim_list[i] = node_name_to_contiguous_id[omim]
            for i in range(len(gene_list)):
                gene = gene_list[i]
                gene_list[i] = node_name_to_contiguous_id[gene]
            # ------------------------------------------

            # ------- construct final dataset --------------
            edge_index = torch.tensor(edge_list, dtype=torch.long)
            print(torch.max(edge_index))
            print(len(x))
            data = Data(x=x, edge_index=edge_index.t().contiguous())
            print(data)
            # ----------------------------------------------

            # ------------- save useful information --------------
            # save dataset
            torch.save(data, os.path.join(save_dir, f'dataset_fold_{fold_ind}.pt'))
            # save train test split based on contiguous id
            with open(os.path.join(save_dir, f"train_test_split_fold_{fold_ind}.pkl"), "wb") as f:
                pickle.dump([refine_train, refine_test], f)
            # save contiguous id
            with open(os.path.join(save_dir, f"id_conversion_fold_{fold_ind}.pkl"), "wb") as f:
                pickle.dump([node_name_to_contiguous_id, contiguous_id_to_node_name], f)
            # save omim and test list
            with open(os.path.join(save_dir, f"omim_gene_fold_{fold_ind}.pkl"), "wb") as f:
                pickle.dump([omim_list, gene_list], f)
            # ----------------------------------------------------
            print(f"finish fold {fold_ind} \n==========")