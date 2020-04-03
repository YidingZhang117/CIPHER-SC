import os
import pickle
import numpy as np


omim_list = []
gene_list = []
edge_result = []

# -------------
# read disease
# -------------
# mesh
omim_mesh_link = {}
disease_folder = "disease/MESH/"
disease_file_list = os.listdir(os.path.join(disease_folder, "mesh_AC"))
for name in disease_file_list:
    full_name = os.path.join(os.path.join(disease_folder, "mesh_AC", name))
    omim = "omim:"+name.split(".")[0]
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


# hpo disease
disease_folder = "disease/HPO/"
with open(os.path.join(disease_folder, "omim_hpo_pair.pkl"), "rb") as f:
    edge_result.extend(pickle.load(f))


print("disease done")


# -------------
#  read gene
# -------------
# ----- PPI -----
with open("gene/PPI/PPI_use.txt", "r") as f:
    content = f.read().split("\n")
    for l in content:
        g1, g2 = l.split("\t")
        g1, g2 = "gene:"+g1, "gene:"+g2
        edge_result.append(g1+"\t"+g2)

# ----- GO -------
with open("gene/GO2/entrez_GO_dic_bp.pkl", "rb") as f:
    entrez_GO_dic = pickle.load(f)
    for gene, go in entrez_GO_dic.items():
        for g in go:
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

    if n2.startswith("omim:"):
        omim_list.append(n2)
    elif n2.startswith("gene:"):
        gene_list.append(n2)

omim_list = list(set(omim_list))
gene_list = list(set(gene_list))

print("omim:", len(omim_list))
print("gene:", len(gene_list))

# ------------------------
# read omim and gene link
# ------------------------
saved_links = []
with open(f"disease_gene/pheno_entrez_all_link.pkl", "rb") as f:
    content = pickle.load(f)
    for omim, gene in content:
        if "omim:"+str(omim) in omim_list and "gene:"+str(gene) in gene_list:
            saved_links.append([omim, gene])
print(len(saved_links))
saved_links = np.array(saved_links, dtype=np.int32)
print(saved_links)

with open(f"disease_gene/union_links.pkl", "wb") as f:
    pickle.dump(saved_links, f)
with open(f"union_omim_gene_list.pkl", "wb") as f:
    pickle.dump([omim_list, gene_list], f)
