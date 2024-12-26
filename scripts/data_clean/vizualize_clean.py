import matplotlib.pyplot as plt
import seaborn as sns

def plot_outliers(y, y_truncated):
    """
    Visualiza distribuição dos dados antes e depois da truncagem.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sns.set(style="whitegrid")
    sns.histplot(y, bins=50, kde=True, ax=axes[0], color="red")
    axes[0].set_title("Histograma Antes do Tratamento")
    axes[0].set_xlabel("Valores do Target")
    sns.histplot(y_truncated, bins=50, kde=True, ax=axes[1], color="green")
    axes[1].set_title("Histograma Depois do Tratamento")
    axes[1].set_xlabel("Valores do Target")
    plt.tight_layout()
    plt.savefig("./outliers_truncados.png")
    plt.show()