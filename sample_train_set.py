
import numpy as np
import sys
import re 

def sample(data_name):
    data_folder = 'data/' + data_name + '/'
    with open(data_folder + 'label.csv', 'r') as f:
        label_dict = {x[0]: int(x[1]) for x in [re.split('[ ,]', x) for x in f]}

    nodes_list = list(label_dict.keys())
    np.random.shuffle(nodes_list)

    train_set_ratio = 0.7
    train_set_emb_ratio = 0.5

    train_set_knot = int(train_set_ratio * len(nodes_list))
    nodes_train = nodes_list[: train_set_knot]
    nodes_test = nodes_list[train_set_knot+1:]

    train_set_emb_knot = int(train_set_emb_ratio * len(nodes_train))
    nodes_train_emb = nodes_train[:train_set_emb_knot]
    nodes_train_reg = nodes_train[train_set_emb_knot+1:]

    assert len(set(nodes_train_emb).intersection(nodes_train_reg)) == 0
    assert len(set(nodes_train).intersection(nodes_test)) == 0
    assert len(set(nodes_train_emb) - set(nodes_train)) == 0
    assert len(set(nodes_train_reg) - set(nodes_train)) == 0

    print('data: ' + data_name)
    print('train set ratio: ' + str(train_set_ratio))
    print('train set num: ' + str(len(nodes_train)))
    print('test set num: ' + str(len(nodes_test)))
    print('train set emb num: ' + str(len(nodes_train_emb)))
    print('train set reg num: ' + str(len(nodes_train_reg)))

    with open(data_folder + 'label_is.csv', 'w+') as f:
        for n in nodes_train:
            f.write(n + ' ' + str(label_dict[n]) + '\n')

    with open(data_folder + 'label_os.csv', 'w+') as f:
        for n in nodes_test:
            f.write(n + ' ' + str(label_dict[n]) + '\n')

    with open(data_folder + 'label_emb.csv', 'w+') as f:
        for n in nodes_train_emb:
            f.write(n + ' ' + str(label_dict[n]) + '\n')

    with open(data_folder + 'label_reg.csv', 'w+') as f:
        for n in nodes_train_reg:
            f.write(n + ' ' + str(label_dict[n]) + '\n')


for data_name in sys.argv[1:]:
    sample(data_name)
