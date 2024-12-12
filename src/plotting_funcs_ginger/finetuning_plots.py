###########PLOTTING WITH BOTTOM#############################
# Plotting was done mostly in Jupiter Notebooks
import matplotlib.pyplot as plt
import seaborn as sns

def plot(df, target_col='AUC', use_lora=True, share_y=True, save=False):
    
    dataset_mapping = {'MidJourney': 'MidJourney', 'starganv2': 'StarGAN v2', 'heygen': 'HeyGen'}
    baseline_scores = {'MidJourney': 0.4343, 'starganv2': 0.78, 'heygen': 0.701}
    title_mapping = { 'CollabDiff': 'CollabDiff', 'JDB_random': 'Random JDB & Coco', 
                     'JDB_train': 'Random JDB & Coco', 'mixed': 'Mixed Dataset'}
        
    df_filtered = df[df['LoRA'] == use_lora]
    training_set = df_filtered['Train Dataset'].iloc[0]
    finetune_type = "PEFT-LoRA" if use_lora else "Full Model Finetuning"
    datasets = ['MidJourney', 'starganv2', 'heygen']
    filtered_dfs = {dataset: df_filtered[df_filtered['Validation Dataset'] == dataset] for dataset in datasets}
    # *********************************************************************************************************************
    # The different colors, Dark4 is best
    seaborn_palettes = [ "Set1", "Dark2", "tab10", "Paired", "Spectral", "muted" ]
    
    color_palette = sns.color_palette("Dark2", 4)
    size_color_map = {size: color for size, color in zip(sorted(df_filtered['Size of Dataset'].unique()), color_palette)}
    
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=share_y)
    
    sns.set_style("whitegrid")
    handles, labels = [], []
    baseline_handles, baseline_labels = [], []

    for ax, dataset in zip(axes, datasets):
        data = filtered_dfs[dataset]
        ax.grid(True, alpha=0.5, linestyle='--')
        baseline = baseline_scores[dataset]
        baseline_line = ax.axhline(y=baseline, color='gray', linestyle='--', linewidth=1.5)
        
        if dataset_mapping[dataset] not in baseline_labels:
            baseline_handles.append(baseline_line)
            baseline_labels.append(f"{dataset_mapping[dataset]} Baseline: {baseline:.2f}")

        for size, group in data.groupby('Size of Dataset'):
            epochs = group['Epoch'].values + 1
            performance = group[target_col].tolist()
            lr = group['Learning Rate'].iloc[0]
            label = f"Size {size} (LR={lr:.0e})"
            if use_lora:
                lora_alpha = group['LoRA Alpha'].iloc[0]
                lora_rank = group['LoRA Rank (r)'].iloc[0]
                label += f", α={lora_alpha}, r={lora_rank}"
            line, = ax.plot(
                epochs,
                performance,
                marker='o',
                color=size_color_map[size]
            )
            if label not in labels:
                handles.append(line)
                labels.append(label)

        ax.set_title(f"Performance on {dataset_mapping[dataset]} Validation Set", fontsize=14)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel(target_col if ax == axes[0] else '', fontsize=12)
        ax.set_xticks([1, 2, 3, 4, 5])

    all_handles = handles + baseline_handles
    all_labels = labels + baseline_labels

    fig.legend(all_handles,
        all_labels,
        loc='lower center',
        ncol=2,
        fontsize=10,
        title="Dataset Size, LR, & LoRA" if use_lora else "Dataset Size & LR",
        title_fontsize=12,
        bbox_to_anchor=(0.5, -0.2))

    suptitle = f"Cross-Dataset Results of CLIP Finetuned with {finetune_type} on {title_mapping[training_set]}"
    plt.suptitle(suptitle, fontsize=18, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save:
        root_fp = '/home/ginger/code/gderiddershanghai/deep-learning/visualization/finetuning_plots/'
        plt.savefig(f"{root_fp}_{target_col}_{training_set}_{use_lora}_performance_plot.png", 
            dpi=300, bbox_inches='tight')

    plt.show()
