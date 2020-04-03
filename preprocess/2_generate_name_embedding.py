# -*- encoding=utf-8
import os
import numpy as np
import pickle


type_name_list = ["union_MH_PG", "union_SMH_PG"]

for type_name in type_name_list:
    for fold_ind in range(1, 6):
        for dim in [100]:
            fold_name_embedding = {}
            with open(f"emb_result/{type_name}/{type_name}_emb_fold_{fold_ind}_morethan2_{dim}.emb", "r") as f:
                content = f.read().split("\n")[1:]
                for l in content:
                    if len(l) == 0:
                        continue
                    items = l.split(" ")
                    node_id = items[0]
                    vec = [float(i) for i in items[1:]]
                    fold_name_embedding[node_id] = vec
            # save
            if not os.path.exists(f"name_embedding/{type_name}"):
                os.mkdir(f"name_embedding/{type_name}")
            with open(f"name_embedding/{type_name}/{type_name}_name_embedding_fold_{fold_ind}_morethan2_{dim}.pkl", "wb") as f:
                pickle.dump(fold_name_embedding, f)
            print(f"{type_name}, dim {dim} finished")
