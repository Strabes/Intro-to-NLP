from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

def plot_top_words(model, feature_names, n_top_words, title):
    rows = int(np.ceil(len(model.components_)/5))
    fig, axes = plt.subplots(rows, 5, figsize=(15, 8*rows), sharex=True)
    axes = axes.flatten()
    for topic_idx, topic in enumerate(model.components_): #[:10,:]
        top_features_ind = topic.argsort()[: -n_top_words - 1 : -1]
        top_features = [feature_names[i] for i in top_features_ind]
        weights = topic[top_features_ind]

        ax = axes[topic_idx]
        ax.barh(top_features, weights, height=0.7)
        ax.set_title(f"Topic {topic_idx +1}", fontdict={"fontsize": 30})
        ax.invert_yaxis()
        ax.tick_params(axis="both", which="major", labelsize=20)
        for i in "top right left".split():
            ax.spines[i].set_visible(False)
        fig.suptitle(title, fontsize=40)

    plt.subplots_adjust(top=0.95, bottom=0.05, wspace=0.95, hspace=0.05)
    return fig


def plot_heatmap(lda_preds, targets,target_labels):
    z = (pd.DataFrame(lda_preds)
           .assign(target = targets)
           .groupby("target").mean()
           .applymap(lambda x: round(x,2)))
    fig, ax = plt.subplots(1,1,figsize=(10,10))
    ax.imshow(z);
    ax.axes.set_yticks(range(len(target_labels)));
    ax.axes.set_xticks(range(z.shape[1]));
    ax.axes.set_yticklabels(target_labels);
    ax.axes.set_xticklabels(range(1,z.shape[1]+1));
    ax.set_ylabel("20 Newsgroup Category");
    ax.set_xlabel("Latent Dirichlet Allocation Topic");
    ax.set_title("Relationship between 20 Newsgroup Category and LDA Topics");
    for i in range(len(target_labels)):
        for j in range(z.shape[1]):
            text = ax.text(j, i, z.iloc[i, j],
                       ha="center", va="center", color="w")
    return fig