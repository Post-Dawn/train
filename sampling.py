
def get_labeled_edges(edges, label_dict):
    new_edges = []
    for e in edges:
        if e[0] in label_dict and e[1] in label_dict:
            new_edges.append(e)

    return new_edges


def get_inter_edges(edges, label_dict, label):
    new_edges = []
    for e in edges:
        src = e[0]
        tar = e[1]
        if (src in label_dict and tar not in label_dict
            and label_dict[src] == label):
            new_edges.append(e)
        if (tar in label_dict and src not in label_dict
            and label_dict[tar] == label):
            new_edges.append(e)

    return new_edges


def get_unsup_edges(edges, label_dict):
    new_edges = []
    for e in edges:
        src = e[0]
        tar = e[1]
        if src not in label_dict and tar not in label_dict:
            new_edges.append(e)
    return new_edges