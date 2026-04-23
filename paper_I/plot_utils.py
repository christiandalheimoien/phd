import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import seaborn as sns

def model_eval_plot(df, true_col, pred_col, color_col, x_label, y_label, ax, legend_order=[]):

    order = ['6060', '6063', '6005 v1', '6005 v2', '6082 v1', '6082 v2', '6082 v3']
    hue_style_order = ['6005 v1', '6082 v1', '6063', '6005 v2', '6060', '6082 v3', '6082 v2']

    plt.sca(ax)
    sns.scatterplot(x=true_col, y=pred_col, data=df, hue=color_col,
                    hue_order=hue_style_order, style=color_col,
                    style_order=hue_style_order, palette='muted')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.plot([0, 400], [0, 400], linestyle='dotted', color='black', alpha=0.5)
    plt.xlim(200, 400)
    plt.ylim(200, 400)
    plt.grid()

    rmse = np.sqrt(mean_squared_error(df[true_col], df[pred_col]))
    r2 = r2_score(df[true_col], df[pred_col])
    xy1 = (255,388)
    xy2 = (255,378)
    _ = plt.annotate(f'$R^2={r2:0.2f}$', xy=xy1, xytext=xy1, backgroundcolor="w")
    _ = plt.annotate(f'RMSE$={rmse:0.1f}$ MPa', xy=xy2, xytext=xy2, backgroundcolor="w")

        # Reorder legend if legend order specified
    if len(legend_order)==0:
        #legend_order = [4, 2, 0, 3, 1, 6, 5]
        legend_order = [0, 1, 2, 3, 4, 5, 6]
    handles, labels = ax.get_legend_handles_labels() #get handles and labels
    plt.legend([handles[idx] for idx in legend_order],[labels[idx] for idx in legend_order])

    axins = inset_axes(ax, width="100%", height="100%",
                        bbox_to_anchor=(.65, .1, .3, .3),
                        bbox_transform=ax.transAxes)
    error = df[pred_col] - df[true_col]
    error.hist(ec='black', linewidth=1, ax=axins, color='gray', bins=20)
    # y_test_df['Rp02_reg_error'].plot.kde(ax=axins)#secondary_y=True)
    axins.grid(False)
    axins.set_ylim(0)
    axins.set_xlim(-70, 70)
    axins.set_ylabel('')
    axins.set_yticks([])
    axins.set_title('Prediction errors', fontsize=10, backgroundcolor="w", pad=7.5)
    