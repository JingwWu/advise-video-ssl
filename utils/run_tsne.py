from MulticoreTSNE import MulticoreTSNE as TSNE
import torch
import numpy as np
import json

data_0 = torch.load("/data/visual/tsne/AdViSe/AdViSe_feats_mlp_rk0.pt")
data_1 = torch.load("/data/visual/tsne/AdViSe/AdViSe_feats_mlp_rk1.pt")

X, Y, take_off, som, twist, fpos = [], [], [], [], [], []
for data in data_0:
    X.append(data[0].numpy())
    Y.append(data[1])
for data in data_1:
    X.append(data[0].numpy())
    Y.append(data[1])

X = np.array(X)
print(X.shape, len(Y))

with open('/home/wujingwei/Workspace/projects/data_list/diving/Diving48_vocab.json', 'r') as file:
    d48_label = json.load(file)

for g_idx in range(6):
    Y = np.array(Y)
    X_sub, Y_sub = [], []
    for idx, label in enumerate(Y):
        if label % 6 == g_idx:
            X_sub.append(X[idx])
            Y_sub.append(label // 6)
    X_sub = np.array(X_sub)
    Y_sub = np.array(Y_sub)

    np.random.seed(0)
    init = np.random.rand(len(Y_sub), 2)

    embeddings = TSNE(
        n_components=2,
        perplexity=24,
        early_exaggeration=96.0,
        n_iter=5000,
        init=init,
        verbose=1,
        n_jobs=8
    ).fit_transform(X_sub)

    vis_x = embeddings[:, 0]
    vis_y = embeddings[:, 1]

    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from matplotlib.colors import ListedColormap

    cmap = plt.cm.Accent
    size=3.0
    fig = plt.figure(figsize=(2, 2))
    plt.scatter(vis_x, vis_y, s=size, c=Y_sub, cmap=cmap, marker='.')
    plt.clim(vmin=-0.5, vmax=7.5)
    plt.xticks([])
    plt.yticks([])
    plt.show()

    plt.savefig(f'AdViSe_mlp_pp-24_ee-96_nit-5000_tsne_group-{g_idx}.png', dpi=300, format='png')
