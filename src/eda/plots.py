import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_target_distribution(df: pd.DataFrame, target_col: str) -> None:
    """Raw vs log1p distribution of the target — skewness check."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    sns.histplot(data=df, x=target_col, kde=True, ax=axes[0])
    axes[0].set_title(f"{target_col} — raw distribution")

    sns.histplot(np.log1p(df[target_col]), kde=True, ax=axes[1])
    axes[1].set_title(f"{target_col} — after log1p  (skew ≈ {np.log1p(df[target_col]).skew():.3f})")

    fig.suptitle("Target distribution", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.show()
    return fig


def plot_missing_values(df: pd.DataFrame, top_n: int = 30) -> None:
    """Bar chart of columns with the most missing values."""
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False).head(top_n)
    missing_pct = (missing / len(df) * 100).round(1)

    report = pd.DataFrame({"count": missing, "pct": missing_pct})
    print("─── Missing values ────────────────────")
    print(report.to_string())
    print()

    fig, ax = plt.subplots(figsize=(12, 5))
    missing_pct.plot(kind="bar", ax=ax, color="steelblue", edgecolor="white")
    ax.set_title(f"Missing values by column (top {top_n}), %")
    ax.set_ylabel("%")
    ax.set_xlabel("")
    for patch in ax.patches:
        ax.annotate(
            f"{patch.get_height():.1f}",
            (patch.get_x() + patch.get_width() / 2, patch.get_height()),
            ha="center", va="bottom", fontsize=8,
        )
    plt.tight_layout()
    plt.show()
    return fig


def plot_top_correlations(df: pd.DataFrame, target_col: str, top: int = 15) -> None:
    """Bar chart of features most correlated with the target."""
    corr = df.corr(numeric_only=True)[target_col]
    corr = corr.drop(target_col, errors="ignore")
    corr = corr.reindex(corr.abs().sort_values(ascending=False).index).head(top)

    print("─── Top correlations ─────────────────")
    print(corr.to_frame("corr").to_string())
    print()

    colors = ["teal" if v > 0 else "salmon" for v in corr.values]
    fig, ax = plt.subplots(figsize=(10, 5))
    corr.plot(kind="bar", ax=ax, color=colors, edgecolor="white")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title(f"Top-{top} correlations with {target_col}")
    ax.set_ylabel("Pearson r")
    plt.tight_layout()
    plt.show()


def plot_outlier_check(df: pd.DataFrame, x_col: str, y_col: str) -> None:
    """Scatter to spot outliers — large area with suspiciously low price etc."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(df[x_col], df[y_col], alpha=0.5, edgecolors="none", color="steelblue")
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(f"{x_col} vs {y_col}")
    plt.tight_layout()
    plt.show()


def plot_numeric_distributions(df: pd.DataFrame, cols: list, ncols: int = 4) -> None:
    """Grid of histograms for a list of numeric columns."""
    nrows = (len(cols) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3))
    axes = axes.flatten()

    for i, col in enumerate(cols):
        sns.histplot(df[col].dropna(), kde=True, ax=axes[i], color="steelblue")
        axes[i].set_title(col, fontsize=9)
        axes[i].set_xlabel("")

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.show()


def plot_correlation_heatmap(df: pd.DataFrame, top_n: int = 20, target_col: str = "SalePrice") -> None:
    """Heatmap of the top-N most correlated features (easier to read than the full 80×80)."""
    corr_matrix = df.corr(numeric_only=True)

    top_features = (
        corr_matrix[target_col]
        .abs()
        .sort_values(ascending=False)
        .head(top_n)
        .index
    )
    sub_corr = corr_matrix.loc[top_features, top_features]

    fig, ax = plt.subplots(figsize=(12, 10))
    mask = np.triu(np.ones_like(sub_corr, dtype=bool), k=1)
    sns.heatmap(
        sub_corr, mask=mask, annot=True, fmt=".2f",
        cmap="coolwarm", center=0, linewidths=0.5,
        annot_kws={"size": 7}, ax=ax,
    )
    ax.set_title(f"Correlation heatmap — top {top_n} features", fontsize=13)
    plt.tight_layout()
    plt.show()
