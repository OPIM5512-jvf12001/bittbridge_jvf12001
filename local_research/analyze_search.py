import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

folder = "local_research/db_files_csv"
trials_df = pd.read_csv(f'{folder}/trials.csv')
values_df = pd.read_csv(f'{folder}/trial_values.csv')
params_df = pd.read_csv(f'{folder}/trial_params.csv')

# Pivot the parameters so each hyperparameter is a column
params_pivoted = params_df.pivot(index='trial_id', columns='param_name', values='param_value').reset_index()

# Merge the objective values
data = trials_df.merge(values_df[['trial_id', 'value']], on='trial_id')
data = data.rename(columns={'value': 'MAPE'})

# Merge the hyperparameters into the main dataframe
df = data.merge(params_pivoted, on='trial_id')

# Filter for successful trials and ensure numeric types
df = df[df['state'] == 'COMPLETE']
numeric_cols = ['MAPE', 'lr', 'units', 'dense_units', 'dropout', 'batch_size']
df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

plt.figure(figsize=(16, 12))

# MAPE over time
plt.subplot(2, 2, 1)
plt.scatter(df['number'], df['MAPE'], alpha=0.6, edgecolors='w', s=80)
plt.plot(df['number'], df['MAPE'].cummin(), color='red', label='Best Value', linewidth=2)
plt.title('Optimization History (Trial vs. MAPE)', fontsize=14)
plt.xlabel('Trial Number')
plt.ylabel('MAPE (%)')
plt.legend()
plt.grid(True, alpha=0.3)

# Shows the relationship between Learning Rate and performance
plt.subplot(2, 2, 2)
sns.scatterplot(data=df, x='lr', y='MAPE', hue='units', size='dropout', sizes=(40, 200), palette='viridis')
plt.xscale('log')
plt.title('Learning Rate vs. MAPE (Hue: Units, Size: Dropout)', fontsize=14)
plt.xlabel('Learning Rate (Log Scale)')
plt.ylabel('MAPE (%)')

# Units vs. Dense Units
plt.subplot(2, 2, 3)
sns.scatterplot(data=df, x='units', y='dense_units', hue='MAPE', palette='magma', s=100)
plt.title('LSTM Units vs. Dense Units (Hue: MAPE)', fontsize=14)
plt.xlabel('LSTM Units')
plt.ylabel('Dense Units')

# Dropout Distribution
plt.subplot(2, 2, 4)
sns.regplot(data=df, x='dropout', y='MAPE', scatter_kws={'alpha':0.5}, line_kws={'color':'red'})
plt.title('Dropout vs. MAPE (Trendline)', fontsize=14)
plt.xlabel('Dropout Rate')
plt.ylabel('MAPE (%)')

plt.tight_layout()
plt.savefig('local_research/figs/optuna_dashboard_summary.png')
plt.show()

print(f"Best MAPE found: {df['MAPE'].min():.4f}")