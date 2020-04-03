import argparse
import numpy as np
import os
import time
import pickle
import networkx as nx
import util.node2vec as node2vec
from gensim.models import Word2Vec


def parse_args():
    '''
    Parses the node2vec arguments.
    '''
    parser = argparse.ArgumentParser(description="Run node2vec.")

    parser.add_argument('--input', nargs='?', default='',
                        help='Input graph path')

    parser.add_argument('--output', nargs='?', default='',
                        help='Embeddings path')

    parser.add_argument('--dimensions', type=int, default=128,
                        help='Number of dimensions. Default is 128.')

    parser.add_argument('--walk-length', type=int, default=80,
                        help='Length of walk per source. Default is 80.')

    parser.add_argument('--num-walks', type=int, default=10,
                        help='Number of walks per source. Default is 10.')

    parser.add_argument('--window-size', type=int, default=10,
                        help='Context size for optimization. Default is 10.')

    parser.add_argument('--iter', default=5, type=int,
                      help='Number of epochs in SGD')

    parser.add_argument('--workers', type=int, default=8,
                        help='Number of parallel workers. Default is 8.')

    parser.add_argument('--p', type=float, default=1,
                        help='Return hyperparameter. Default is 1.')

    parser.add_argument('--q', type=float, default=1,
                        help='Inout hyperparameter. Default is 1.')

    parser.add_argument('--weighted', dest='weighted', action='store_true',
                        help='Boolean specifying (un)weighted. Default is unweighted.')
    parser.add_argument('--unweighted', dest='unweighted', action='store_false')
    parser.set_defaults(weighted=False)

    parser.add_argument('--directed', dest='directed', action='store_true',
                        help='Graph is (un)directed. Default is undirected.')
    parser.add_argument('--undirected', dest='undirected', action='store_false')
    parser.set_defaults(directed=False)

    return parser.parse_args()

def read_graph(edgelist_name):
    '''
    Reads the input network in networkx.
    '''
    if args.weighted:
        G = nx.read_edgelist(edgelist_name, nodetype=str, data=(('weight',float),), create_using=nx.DiGraph())
    else:
        G = nx.read_edgelist(edgelist_name, nodetype=str, create_using=nx.DiGraph())
        for edge in G.edges():
            G[edge[0]][edge[1]]['weight'] = 1

    if not args.directed:
        G = G.to_undirected()

    return G

def learn_embeddings(walks, dim, output_name):
    '''
    Learn embeddings by optimizing the Skipgram objective using SGD.
    '''
    walks = [list(map(str, walk)) for walk in walks]
    model = Word2Vec(walks, size=dim, window=args.window_size, min_count=0, sg=1, workers=args.workers, iter=args.iter)
    model.wv.save_word2vec_format(output_name)
    return

def main(args, type_name):
    print(type_name)
    for fold_ind in range(1, 6):
        edgelist_name = f"edgelist_result/{type_name}/raw_link_edge_list_result_fold_{fold_ind}_morethan2.txt"
        print("===========\n")
        print("start to node2vec for ", edgelist_name)
        print("\n===========")
        t1 = time.time()
        nx_G = read_graph(edgelist_name)
        t2 = time.time()
        print("===========\n")
        print(t2-t1, "s:  read done")
        print("\n===========")
        G = node2vec.Graph(nx_G, args.directed, args.p, args.q)
        G.preprocess_transition_probs()
        t3 = time.time()
        print("==================\n")
        print(t3-t2, "s:  preprocess done")
        print("\n==================")
        t3 = time.time()
        walks = G.simulate_walks(args.num_walks, args.walk_length)
        t4 = time.time()
        print("======================\n")
        print(t4-t3, "s:  simulate walk done")
        print("\n======================")
        
        if not os.path.exists(f"emb_result/{type_name}"):
            os.mkdir(f"emb_result/{type_name}")
        for dim in [100]:
            output_name = f"emb_result/{type_name}/{type_name}_emb_fold_{fold_ind}_morethan2_{dim}.emb"
            print(output_name)
            t4 = time.time()
            learn_embeddings(walks, dim, output_name)
            t5 = time.time()
            print("======================\n")
            print(t5-t4, "s:  Word2Vec done")
            print("\n======================")


if __name__ == "__main__":
    edgelist_name_type = ["union_MH_PG", "union_SMH_PG"]
    args = parse_args()
    for type_name in edgelist_name_type:
        main(args, type_name)
