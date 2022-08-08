import os
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from matplotlib.colors import ListedColormap


def visualize(h, color, model, dataset, epoch):
    z = TSNE(n_components=2).fit_transform(h.detach().cpu().numpy())
    plt.figure(figsize=(10,10))
    plt.axis('off')
    plt.xticks([])
    plt.yticks([])

    plt.scatter(z[:, 0], z[:, 1], s=70, c=color.detach().cpu().numpy(), cmap=ListedColormap(['green', 'blue', 'orange', 'purple', 'red']))
    if not os.path.exists('./figure'):
        os.makedirs('./figure')
    plt.savefig(f'./figure/{model}_{dataset}_{epoch}.png', dpi=300)