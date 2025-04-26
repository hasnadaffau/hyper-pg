import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def viz_query_output(hyper, hypermax, labels,
                                 output_path=None,
                                 table_path=None,
                                 figsize=(7, 7.5),
                                 fontsize=20):
    x = np.arange(len(labels))
    width = 0.15

    gap = [round(maxi - mini, 3) for mini, maxi in zip(hyper, hypermax)]

    fig, ax = plt.subplots(figsize=figsize)
    rects1 = ax.barh(x - 0.15, hyper, width, label='Minimum', color='lightcoral', edgecolor='black', hatch="//")
    rects2 = ax.barh(x, hypermax, width, label='Maximum', color='gainsboro', edgecolor='black', hatch="\\\\")

    for i in range(len(gap)):
        xpos = max(hyper[i], hypermax[i]) + 0.05
        ypos = x[i] - 0.075
        ax.text(xpos, ypos, f'+{gap[i]}', fontsize=fontsize * 0.7, color='black')

    ax.set_xlabel('Query Output', fontsize=fontsize)
    ax.set_yticks(x)
    ax.set_yticklabels(labels, fontsize=fontsize)
    ax.xaxis.set_ticks_position('top')
    ax.xaxis.set_label_position('top')
    plt.xticks(np.arange(0, 2, 0.5))
    ax.legend(loc='lower right')
    ax.invert_yaxis()
    ax.margins(0.1, 0.05)
    plt.xlim([0, 1.5])
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path)
    else:
        plt.show()

    df_gap = pd.DataFrame({
        'Feature': labels,
        'Min Query Output': hyper,
        'Max Query Output': hypermax,
        'Gap': gap
    })

    if table_path:
        df_gap.to_csv(table_path, index=False)

    return df_gap
