import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


def plot_variable_pairs(df):
    sns.set(style="ticks")

    # Created a pairplot with regression lines
    sns.pairplot(df, kind="reg", diag_kind="kde", corner=True)
    plt.tight_layout()
    plt.show()


def plot_categorical_and_continuous_vars(df):

    # Bin 'year_built' column
    year_bins = list(range(1900, 2021, 10))
    # Define custom labels for the bins
    bin_labels = [f"{start}-{start + 9}s" for start in year_bins[:-1]]
    
    df['year_built_bins'] = pd.cut(df['year_built'], bins=year_bins, labels=bin_labels)
    
    # define categories
    cont_cols = ['area', 'tax_value', 'taxamount']
    cat_cols = ['bedrooms', 'bathrooms', 'fips', 'year_built_bins']

    for cat_col in cat_cols:
        for cont_col in cont_cols:
            # Create a figure with three subplots
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))

            # Create a Box Plot
            sns.boxplot(x=cat_col, y=cont_col, data=df, palette='Set2', ax=axes[0])
            axes[0].set_xlabel(cat_col)
            axes[0].set_ylabel(cont_col)
            axes[0].set_title(f'Box Plot: {cat_col} vs. {cont_col}')

            # Create a Scatter Plot
            sns.scatterplot(x=cat_col, y=cont_col, data=df, alpha=0.5, color='b', ax=axes[1])
            axes[1].set_xlabel(cat_col)
            axes[1].set_ylabel(cont_col)
            axes[1].set_title(f'Scatter Plot: {cat_col} vs. {cont_col}')

            # Create a Point Plot
            sns.pointplot(x=cat_col, y=cont_col, data=df, errorbar='sd', capsize=0.2, color='b', ax=axes[2])
            axes[2].set_xlabel(cat_col)
            axes[2].set_ylabel(cont_col)
            axes[2].set_title(f'Point Plot: {cat_col} vs. {cont_col}')

            plt.tight_layout()
            plt.show()
            
    # Remove the 'year_built_bins' column
    df.drop(columns=['year_built_bins'], inplace=True)
