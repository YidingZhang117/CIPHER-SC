import os
import pickle


USE_H = True  # use hierachical network
type_name_list = ["union_MH_PG", "union_SMH_PG"]

for type_name in type_name_list:
    print("===================")
    if not os.path.exists(type_name):
        os.mkdir(type_name)

    omim_list = []
    gene_list = []
    mesh_list = []
    hpo_list = []
    go_list = []
    
    mesh_mesh_link = []
    hpo_hpo_link = []
    go_go_link = []

    edge_result = []
    edge_result_add = []

    # -------------
    # read disease
    # -------------
    dis_type = type_name.split("_")[1]

    # ------- MeSH --------
    if "M" in dis_type:
        omim_mesh_link = {}
        disease_folder = "disease/MESH/"
        disease_file_list = os.listdir(os.path.join(disease_folder, "mesh_AC"))
        for name in disease_file_list:
            full_name = os.path.join(os.path.join(disease_folder, "mesh_AC", name))
            omim = "omim:"+name.split(".")[0]
            omim_list.append(omim)
            temp_list = []
            with open(full_name, "r") as f:
                content = f.read()
                if len(content) == 0:
                    continue
                content = content.split("\n")
                for l in content:
                    mesh_id = l.split("|")[0]
                    temp_list.append(mesh_id)
            temp_list = list(set(temp_list))
            omim_mesh_link[omim] = temp_list

        # edge for omim and mesh concept
        for omim, mesh in omim_mesh_link.items():
            for m in mesh:
                edge_result.append(omim+"\t"+m)

        if USE_H:
            # edge for mesh concept tree structures
            with open(os.path.join(disease_folder, "sub_network_edge_list.txt"), "r") as f:
                mesh_mesh_link = f.read().split("\n")
            edge_result.extend(mesh_mesh_link)


    # --------- hpo ---------
    if "H" in dis_type:
        disease_folder = "disease/HPO/"
        with open(os.path.join(disease_folder, "omim_hpo_pair.pkl"), "rb") as f:
            edge_result.extend(pickle.load(f))
            
        if USE_H:
            # edge for mesh concept tree structures
            with open(os.path.join(disease_folder, "sub_network_edge_list.txt"), "r") as f:
                hpo_hpo_link = f.read().split("\n")
            edge_result.extend(hpo_hpo_link)


    print("disease done")


    # -------------
    #  read gene
    # -------------
    gene_type = type_name.split("_")[-1]
    # ----- PPI -----
    if "P" in gene_type:
        with open("gene/PPI/PPI_use.txt", "r") as f:
            content = f.read().split("\n")
            for l in content:
                g1, g2 = l.split("\t")
                edge_result.append("gene:"+g1+"\tgene:"+g2)

    # ----- GO -------
    if "G" in gene_type:
        remain_go_list = []
        with open(f"gene/GO/GO_use_remain10.pkl", "rb") as f:
            remain_go_list = pickle.load(f)

        remain_go_list = remain_go_list[:,0].tolist()
        remain_go_list = set(remain_go_list)

        with open(f"gene/GO/entrez_GO_dic_bp.pkl", "rb") as f:
            entrez_GO_dic = pickle.load(f)
            for gene, go in entrez_GO_dic.items():
                for g in go:
                    if g in remain_go_list:
                        edge_result.append("gene:"+gene+"\t"+g)
        
    print("gene done")


    # ------------------------
    # node number
    # ------------------------
    for l in edge_result:
        n1, n2 = l.split("\t")
        if n1.startswith("omim:"):
            omim_list.append(n1)
        elif n1.startswith("gene:"):
            gene_list.append(n1)
        elif n1.startswith("D"):
            mesh_list.append(n1)
        elif n1.startswith("HP:"):
            hpo_list.append(n1)
        elif n1.startswith("GO:"):
            go_list.append(n1)

        if n2.startswith("omim:"):
            omim_list.append(n2)
        elif n2.startswith("gene:"):
            gene_list.append(n2)
        elif n2.startswith("D"):
            mesh_list.append(n2)
        elif n1.startswith("HP:"):
            hpo_list.append(n1)
        elif n2.startswith("GO:"):
            go_list.append(n2)

    omim_list = list(set(omim_list))
    gene_list = list(set(gene_list))
    mesh_list = list(set(mesh_list))
    go_list = list(set(go_list))
    hpo_list = list(set(hpo_list))

    print("type: ", type_name)
    print("omim:", len(omim_list))
    print("gene:", len(gene_list))
    print("mesh:", len(mesh_list))
    print("hpo:", len(hpo_list))
    print("GO:", len(go_list))

    # ------------------------
    # read Single cell link
    # ------------------------
    with open("union_omim_gene_list.pkl", "rb") as f:
        union_omim_list, union_gene_list = pickle.load(f)
    edge_result_add = []
    if "S" in dis_type:
        with open("disease_gene/sc_add_topall20.txt", "r") as f:
            for line in f.read().strip().split("\n")[1:]:
                d, g = line.strip().split("\t")
                d = "omim:"+str(d)
                g = "gene:"+str(g)
                if d in union_omim_list and g in union_gene_list:
                    edge_result_add.append(d+"\t"+g)
        edge_result.extend(edge_result_add)
    
    print(len(edge_result))

    # ------------------------
    # read omim and gene link
    # ------------------------
    for fold_ind in range(1, 6):
        print(f"run fold {fold_ind}")
        
        with open("disease_gene/train_test_association_morethan2_union.pkl", "rb") as f:
            train, test = pickle.load(f)[fold_ind-1]
        test_set_dict = {str(omim)+"#"+str(gene): 1 for omim, gene in test}
        refine_train = []
        with open("disease_gene/union_links.pkl", "rb") as f:
            content = pickle.load(f)
            for omim, gene in content:
                if str(omim) + "#" + str(gene) in test_set_dict:
                    continue
                refine_train.append("omim:"+str(omim)+"\tgene:"+str(gene))

        print(len(refine_train), len(test))
        # full edge result, save to file
        fold_edge_result = []
        fold_edge_result.extend(edge_result)
        fold_edge_result.extend(refine_train)

        with open(f"{type_name}/raw_link_edge_list_result_fold_{fold_ind}_morethan2.txt", "w") as f:
            f.write("\n".join(fold_edge_result))

