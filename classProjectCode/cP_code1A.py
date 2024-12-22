#Anova Test for Dengue Fever Prognosis Dataset
import pandas as pd
from scipy.stats import f_oneway
from sklearn.model_selection import train_test_split

# Load the dataset
file_path = './Dengue_Fever_Prognosis_Dataset.csv'
data = pd.read_csv(file_path)

# Filter out ND columns and separate DHF and DF
dhf_columns = [col for col in data.columns if "DHF" in col]
df_columns = [col for col in data.columns if "DF" in col]

# Transpose for easier indexing
dhf_data = data[dhf_columns].T
df_data = data[df_columns].T

# Add a column to identify the category
dhf_data['Category'] = 'DHF'
df_data['Category'] = 'DF'

# Combine DHF and DF data
combined_data = pd.concat([dhf_data, df_data])

# Split into training and testing for each category
dhf_train, dhf_test = train_test_split(dhf_data, test_size=0.2, random_state=42)
df_train, df_test = train_test_split(df_data, test_size=0.2, random_state=42)

# Perform ANOVA test for each Probe_Set_ID
data.set_index('Probe_Set_ID', inplace=True)

anova_results = []
for probe in data.index:
    # Extract gene expression levels for DHF and DF in the training set
    dhf_values = data.loc[probe, dhf_train.index].values
    df_values = data.loc[probe, df_train.index].values
    
    # Perform ANOVA test
    f_stat, p_value = f_oneway(dhf_values, df_values)
    
    # Store the results
    anova_results.append({'Probe_Set_ID': probe, 'F-statistic': f_stat, 'p-value': p_value})

# Convert the results to a DataFrame
anova_results_df = pd.DataFrame(anova_results)

# Sort by p-value (ascending order)
anova_results_df.sort_values(by='p-value', inplace=True)

# Extract top 10 genes based on p-value
top_genes = anova_results_df.head(10)['Probe_Set_ID']

# Extract the values for training and testing DHF and DF patients for these genes
train_dhf_values = data.loc[top_genes, dhf_train.index]
train_df_values = data.loc[top_genes, df_train.index]

test_dhf_values = data.loc[top_genes, dhf_test.index]
test_df_values = data.loc[top_genes, df_test.index]

# Combine them into separate DataFrames for easier interpretation
train_values = pd.concat([train_dhf_values, train_df_values], axis=1, keys=['DHF', 'DF'])
test_values = pd.concat([test_dhf_values, test_df_values], axis=1, keys=['DHF', 'DF'])

# Display the top genes and their training/testing values
print("Top Genes from ANOVA Test")
print(anova_results_df.head(10))

print("\nTraining Values for Top Genes")
print(train_values)

print("\nTesting Values for Top Genes")
print(test_values)
