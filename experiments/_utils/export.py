import os

def export_figure(fig, dir_path, file_name, hide_titles=True):
    if hide_titles:
        fig.suptitle('')
        for ax in fig.axes:
            ax.set_title('')

    save_path = os.path.join(dir_path, file_name)
    fig.savefig(save_path, bbox_inches='tight')

    print(f"Figure saved to `_exports` folder as `{file_name}`")
