import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from const import LETTERS

import numpy as np

colors = ['r', 'g', 'b', 'k', 'y']

model_scatter_style = {
    'Phi-3': 's',
    'LLaMA-3': 'o',
    'LLaMA-3-3B': '*',
    'Mistral-2': 'D',
}

model_to_nice_model = {
    'Phi-3': 'Phi-3',
    'LLaMA-3': 'LLaMA-3-8B',
    'LLaMA-3-3B': 'LLaMA-3-3B',
    'Mistral-2': 'Mistral-2',
}

dataset_to_nice_dataset = {
    'arc-challenge': 'ARC-Challenge',
    'openbook': 'OpenBookQA',
    'sqa': 'StrategyQA',
    'sports': 'Sports'
}

model_color = {
    'Phi-3': 'tab:blue',
    'LLaMA-3': 'tab:red',
    'LLaMA-3-3B': 'tab:orange',
    'Mistral-2': 'tab:green',
}

model_color_simple = {
    'Phi-3': 'b',
    'LLaMA-3': 'r',
    'LLaMA-3-3B': 'k',
    'Mistral-2': 'g',
}

method_color = {
    'npo_grad_diff': 'tab:orange',
    'npo_KL': 'tab:red',
}

method_shape = {
    'npo_grad_diff': 'o',
    'npo_KL': '*',
    # 'rmu': 'o',
}


THICK = 150

# Create custom legend handles for models
method_handles = [
    Line2D([0], [0], marker=style, color='black', linestyle='None', markersize=8, label=model)
    for model, style in method_shape.items()
]

# Create custom legend handles for methods
model_handles = [
    Line2D([0], [0], color=color, linestyle='None', marker='s', markersize=8, label=model_to_nice_model[model])
    for model, color in model_color.items()
]
custom_legend = model_handles # + method_handles


def scatter_results(dataset_results, savefig=False, fmt='pdf'):
    D = len(dataset_results)
    assert D == 2*2
    fig, axs = plt.subplots(2, 2, figsize=(8,6))
    # fig.supxlabel('Efficacy')
    # fig.supylabel('Specificity')
    major_ticks = np.arange(0, 101, 20)
    
    for idx, (dataset, model_results) in enumerate(sorted(dataset_results.items())):
        # One scatter for each dataset
        row = idx // 2
        col = idx % 2
        # Leave some space for point annotations
        axs[row][col].set_ylim(-5, 105)
        axs[row][col].set_xlim(-5,105)
        axs[row][col].set_xticks(major_ticks)
        axs[row][col].set_yticks(major_ticks)
        axs[row][col].grid()
        for model, model_result in model_results.items():
            for method, method_result in model_result.items():
                # if method == 'npo_KL': continue

                color = model_color[model]
                data = []
                # color = model_color[model]
                for lr, res in method_result.items():
                    data.append(
                        (res['efficacy'],
                        res['specificity'],
                        res['faithfulness'],
                        res['n_instances'],
                        lr,
                        res['faithfulness']))

                xs = [d[0] for d in data]
                ys = [d[1] for d in data]
                thickness = [50 + d[5]/100.*THICK for d in data]
                # print(xs, ys)
                lbls = [f"lr={d[4]}" for d in data]
                axs[row][col].scatter(xs, ys, label=model, marker=method_shape[method], facecolors='none', edgecolors=color, s=thickness) # 
            #for i, lbl in enumerate(lbls):
            #    axs[idx].annotate(lbl, (xs[i], ys[i]))
        axs[row][col].set_title(f"{dataset_to_nice_dataset[dataset]}", fontweight="bold")
        if col == 0:
            axs[row][col].set_ylabel('Specificity', fontsize=12)
        if row == 1:
            axs[row][col].set_xlabel('Efficacy', fontsize=12)
    
    plt.tight_layout()
    # Add the custom legends to the plot
    lgd = plt.legend(handles=custom_legend, loc='lower left', ncol=2, fancybox=True, bbox_to_anchor=(-0.45, 0.02)) # , 
    # plt.legend(loc='upper center', bbox_to_anchor=(-1.7, -0.08),
                # ncol=4, fancybox=True, shadow=True)
    if savefig:
        fig.savefig(f'figures/lr_ablation.{fmt}', dpi=fig.dpi, bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.show()

def probs_barplot(probs, agree=None, flips=None, spec=None, eff=None, renorm=False, savefig=False, fname='', plt_title=''):
    E = len(probs)
    A = len(probs[0])
    
    fig, axs = plt.subplots(1, E, sharey=True, sharex=True, figsize=(8,2))
    # fig.suptitle(plt_title, fontsize=14)
    for idx, (ax, p_a) in enumerate(zip(axs, probs)):
        if renorm:
            p_a = [p/sum(p_a) for p in p_a]
            # print(p_a)
        ax.set_ylim(0, 1)
        ax.grid()
        ax.bar(LETTERS[:A], p_a, width=0.3, color=colors[:A])
        if spec is not None:
            ax.set_title(f"A={agree[idx]}, F={flips[idx]}\ns={spec[idx]:.2f}, e={float(np.exp(eff[idx])):.2f}", fontsize=10)
        
    
    plt.tight_layout()
    if savefig:
        plt.savefig(f"{fname}.pdf", dpi=fig.dpi)
    plt.show()