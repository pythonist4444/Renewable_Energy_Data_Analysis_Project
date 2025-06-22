import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import squarify
import json

# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the full path to the CSV file
csv_path = os.path.join(current_dir, 'Training_set_augmented.csv')

df = pd.read_csv(csv_path)

# Initial Data Exploration
# Display the first few rows of the dataset
print(df.head())
# Display the information about the dataset
print(df.info())
# Display the shape of the dataset (number of rows and columns)
print(f"Shape of dataset: {df.shape}")
# Check for missing values
print(df.isnull().sum())
# Check for duplicates
print(df.duplicated().sum())
# Check the statistics of the dataset    
print(df.describe())


#  EDA on Renewable Energy Data

# ========== Phase 1: General Overview ==============================================================================================================================
print(f"Countries in the dataset: {df['Country'].unique()}")
print(f"Energy types in the dataset: {df['Energy Type'].unique()}")
print(f"Years in the dataset: {df['Year'].unique()}")
print(f"Years covered in the dataset: {sorted(df['Year'].unique())}")
print(f"Total Production (GWh): {df['Production (GWh)'].sum()}")
print(f"Total Investments (USD): {df['Investments (USD)'].sum()}")
print(f"Total Installed Capacity (MW): {df['Installed Capacity (MW)'].sum()}")
print(f"Average Proportion of Energy from Renewables: {df['Proportion of Energy from Renewables'].mean()}")

print("country_counts: ", df['Country'].value_counts())
print("energy_counts: ", df['Energy Type'].value_counts())

# Distribution of renewable energy sources by country
country_energy_counts = df.groupby(['Country', 'Energy Type']).size().reset_index(name='Count')
print(country_energy_counts)
# Visualizing the distribution of renewable energy sources by country
plt.figure(figsize=(12, 6))
sns.countplot(data=df, x='Country', hue='Energy Type', palette='viridis', order=df['Country'].value_counts().index)
plt.title('Distribution of Renewable Energy Sources by Country')
plt.xlabel('Country')
plt.ylabel('Count of Renewable Energy Sources')
plt.xticks(rotation=45)
plt.legend(title='Energy Type')
plt.tight_layout()
# plt.show()



# ========== Phase 2: Country-Level Analysis ==============================================================================================================================
# Total production by country
production_by_country = df.groupby('Country')['Production (GWh)'].sum().sort_values(ascending=False)
print("Total energy production by country:\n", production_by_country)
# Visualizing total production by country
plt.figure(figsize=(12, 6))
sns.barplot(x=production_by_country.index, y=production_by_country.values, palette='coolwarm')
plt.title('Total Renewable Energy Production by Country')
plt.xlabel('Country')
plt.ylabel('Total Production (GWh)')
plt.xticks(rotation=45)
plt.tight_layout()
# plt.show()

# Total investments by country
investments_by_country = df.groupby('Country')['Investments (USD)'].sum().sort_values(ascending=False)
print("Total investments by country:\n", investments_by_country)
# Visualizing total investments by country
plt.figure(figsize=(12, 6))
sns.barplot(x=investments_by_country.index, y=investments_by_country.values, palette='magma')
plt.title('Total Investments in Renewable Energy by Country')
plt.xlabel('Country')
plt.ylabel('Total Investments (USD)')
plt.xticks(rotation=45)
plt.tight_layout()
# plt.show()

# Total installed capacity by country
capacity_by_country = df.groupby('Country')['Installed Capacity (MW)'].sum().sort_values(ascending=False)
print("Total installed capacity by country:\n", capacity_by_country)
# Visualizing total installed capacity by country
plt.figure(figsize=(12, 6))
sns.barplot(x=capacity_by_country.index, y=capacity_by_country.values, palette='coolwarm')
plt.title('Total Installed Capacity for Renewable Energy by Country')
plt.xlabel('Country')
plt.ylabel('Total Installed Capacity (MW)')
plt.xticks(rotation=45)
plt.tight_layout()
# plt.show()

# Proportion of Energy from Renewables Analysis by Country
# Average proportion of energy from renewables by country
proportion_by_country = df.groupby('Country')['Proportion of Energy from Renewables'].mean().sort_values(ascending=False)
print("Average proportion of energy from renewables by country:\n", proportion_by_country)
# Visualizing average proportion of energy from renewables by country
plt.figure(figsize=(12, 6))
sns.barplot(x=proportion_by_country.index, y=proportion_by_country.values, palette='Set2')
plt.title('Average Proportion of Energy from Renewables by Country')
plt.xlabel('Country')
plt.ylabel('Average Proportion of Energy from Renewables (%)')
plt.xticks(rotation=45)
plt.tight_layout()
# plt.show()


# ========== Phase 3: Energy Type Analysis ==============================================================================================================================
# Total production by energy type
production_by_energy = df.groupby('Energy Type')['Production (GWh)'].sum().sort_values(ascending=False)
print("Total energy production by energy type:\n", production_by_energy)
# Visualizing the distribution of renewable energy production by type using a pie chart
plt.figure(figsize=(12, 6))
plt.pie(
    production_by_energy, 
    labels=production_by_energy.index, 
    autopct='%1.1f%%', 
    startangle=140, 
    colors=sns.color_palette('Set2', n_colors=len(production_by_energy)))
plt.title('Distribution of Renewable Energy Production by Type')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
# plt.show()

# Total investments by energy type
investments_by_energy = df.groupby('Energy Type')['Investments (USD)'].sum().sort_values(ascending=False)
print("Total investments by energy type:\n", investments_by_energy)
# Visualizing total investments by energy type using a donut chart
plt.figure(figsize=(12, 6))
plt.pie(
    investments_by_energy, 
    labels=investments_by_energy.index, 
    autopct='%1.1f%%', 
    startangle=140, 
    colors=sns.color_palette('magma', n_colors=len(investments_by_energy)))
plt.title('Distribution of Investments in Renewable Energy by Type')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.gca().add_artist(plt.Circle((0, 0), 0.70, color='white'))  # Create a white circle at the center to make it a donut chart
plt.tight_layout()
# plt.show()

# Total installed capacity by energy type
capacity_by_energy = df.groupby('Energy Type')['Installed Capacity (MW)'].sum().sort_values(ascending=False)
print("Total installed capacity by energy type:\n", capacity_by_energy)
# Visualizing total installed capacity by energy type using a tree map
plt.figure(figsize=(12, 6))
squarify.plot(
    sizes=capacity_by_energy.values, 
    label=[f"{energy}: {value:.2f} MW ({value/capacity_by_energy.sum()*100:.2f}%)" for energy, value in capacity_by_energy.items()], alpha=0.7)
plt.title('Total Installed Capacity for Renewable Energy by Type')
plt.axis('off')
# plt.show()

# Average proportion of energy from renewables by energy type
proportion_by_energy = df.groupby('Energy Type')['Proportion of Energy from Renewables'].mean().sort_values(ascending=False)
print("Average proportion of energy from renewables by energy type:\n", proportion_by_energy)
# Visualizing average proportion of energy from renewables by energy type using a donut chart
plt.figure(figsize=(12, 6))
plt.pie(
    proportion_by_energy, 
    labels=proportion_by_energy.index, 
    autopct='%1.1f%%', 
    startangle=140, 
    colors=sns.color_palette('Set2', n_colors=len(proportion_by_energy)))
plt.title('Average Proportion of Energy from Renewables by Type')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.gca().add_artist(plt.Circle((0, 0), 0.70, color='white'))  # Create a white circle at the center to make it a donut chart
plt.tight_layout()
# plt.show()


# ========== Phase 4: Efficiency Analysis ==============================================================================================================================
# Calculate efficiency metrics
df['Production_per_Investment'] = df['Production (GWh)'] / (df['Investments (USD)']/ 1e9)  # Convert investments to billions USD for better readability
df['Production_per_Capacity'] = df['Production (GWh)'] / df['Installed Capacity (MW)']

# Efficiency by country
efficiency_by_country = df.groupby('Country').agg({
    'Production_per_Investment': 'mean', # Average production per billion USD invested
    'Production_per_Capacity': 'mean', # Average production per MW installed
    'Proportion of Energy from Renewables': 'mean', # Average proportion of energy from renewables
    'Investments (USD)': 'sum', # Total investments in USD
    'Production (GWh)': 'sum', # Total production in GWh
    'Installed Capacity (MW)': 'sum' # Total installed capacity in MW
}).sort_values(by='Production_per_Investment', ascending=False)

print("Efficiency by country:\n", efficiency_by_country)

# Visualizing investment efficiency by country
plt.figure(figsize=(12, 6))
sns.barplot(x=efficiency_by_country.index, y=efficiency_by_country['Production_per_Investment'], palette='viridis')
plt.title('Investment Efficiency by Country (Production per Billion USD Invested)')
plt.xlabel('Country')
plt.ylabel('Production per Billion USD Invested (GWh)')
plt.xticks(rotation=45)
plt.tight_layout()
# plt.show()    

# Visualizing production efficiency by country
plt.figure(figsize=(12, 6))
sns.barplot(x=efficiency_by_country.index, 
            y=efficiency_by_country['Production_per_Capacity'], 
            palette='coolwarm', 
            order=efficiency_by_country.sort_values('Production_per_Capacity', ascending=False).index)
plt.title('Production Efficiency by Country (Production per MW Installed)')
plt.xlabel('Country')
plt.ylabel('Production per MW Installed (GWh)')
plt.xticks(rotation=45)
plt.tight_layout()
# plt.show()


# ========== Phase 5: Correlation Matrix Analysis ==============================================================================================================================
# Select numerical columns for correlation analysis
excluded_columns = ['Production_per_Investment', 'Production_per_Capacity'] # excluding columns with 'Production_per_Investment' and 'Production_per_Capacity' because they are derived metrics from 'Production (GWh)'
numerical_cols = [col for col in df.columns if df[col].dtype in [np.float64, np.int64] and col not in excluded_columns] 
print("Numerical columns for correlation analysis:\n", numerical_cols)

# Calculate correlation matrix
correlation_matrix = df[numerical_cols].corr()

# Focus on the correlations with 'Production (GWh)'
production_corralation = correlation_matrix['Production (GWh)'].sort_values(ascending=False)
print('Factors most correlated with Production (GWh):\n', production_corralation)
# Visualizing factors correlations with Production (GWh) as a bar plot
plt.figure(figsize=(10, 6))
production_correlation = correlation_matrix['Production (GWh)'].sort_values(ascending=False)
sns.barplot(x=production_correlation.values, y=production_correlation.index, palette="viridis")
plt.title('Correlation of Factors with Production (GWh)')
plt.xlabel('Correlation Coefficient')
plt.ylabel('Factors')
plt.tight_layout()
# plt.show()

# Total Investments vs. Production 
plt.figure(figsize=(10, 6))
sns.scatterplot(data=efficiency_by_country, 
                x='Investments (USD)', 
                y='Production (GWh)', 
                hue=efficiency_by_country.index, 
                palette='viridis', 
                s=100)
plt.title('Investments vs. Production by Country')
plt.xlabel('Total Investments (USD)')
plt.ylabel('Total Production (GWh)')
plt.legend(title='Country', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
# plt.show()

# Total Installed Capacity vs. Production
plt.figure(figsize=(10, 6))
sns.scatterplot(data=efficiency_by_country, 
                x='Installed Capacity (MW)', 
                y='Production (GWh)', 
                hue=efficiency_by_country.index, 
                palette='coolwarm', 
                s=100)
plt.title('Installed Capacity vs. Production by Country')
plt.xlabel('Installed Capacity (MW)')
plt.ylabel('Total Production (GWh)')
plt.legend(title='Country', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
# plt.show() 

# Production Efficiency vs. Investment Efficiency. # How efficiently countries use their investments (Production_per_Investment) versus their installed capacity (Production_per_Capacity)
plt.figure(figsize=(10, 6))
sns.scatterplot(data=efficiency_by_country, 
                x='Production_per_Investment', 
                y='Production_per_Capacity', 
                hue=efficiency_by_country.index, 
                palette='Set2', 
                s=100)
plt.title('Production Efficiency vs. Investment Efficiency by Country')
plt.xlabel('Production per Billion USD Invested (GWh)')
plt.ylabel('Production per MW Installed (GWh)')
plt.legend(title='Country', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
# plt.show()



# ========== Phase 6: Policy Impact Analysis ==============================================================================================================================
# Compare countries with and without renewable energy targets
policy_impact = df.groupby('Renewable Energy Targets').agg({
    'Production (GWh)': 'sum',
    'Proportion of Energy from Renewables': 'mean',
    'Investments (USD)': 'sum',
    'Installed Capacity (MW)': 'sum'
})
print("Policy Impact Analysis:\n", policy_impact) # where 0 = No Targets, 1 = Targets
# Visualizing the impact of renewable energy targets on production
plt.figure(figsize=(8, 6))
sns.barplot(x=policy_impact.index, y=policy_impact['Production (GWh)'], palette='Set1')
plt.title('Impact of Renewable Energy Targets on Production (GWh)')
plt.xlabel('Renewable Energy Targets')
plt.ylabel('Total Production (GWh)')
plt.xticks(rotation=0)
plt.tight_layout()
# plt.show()

# Government policies and their impact on renewable energy production
policy_correlation = df.groupby('Government Policies').agg({
    'Production (GWh)': 'sum',
    'Proportion of Energy from Renewables': 'mean',
    'Investments (USD)': 'sum',
    'Installed Capacity (MW)': 'sum'
})
print("Government Policies Impact Analysis:\n", policy_correlation) # where 0 = No Policy, 1 = Policy
# Visualizing the impact of government policies on renewable energy production
plt.figure(figsize=(8, 6))
sns.barplot(x=policy_correlation.index, y=policy_correlation['Production (GWh)'], palette='Set2')
plt.title('Impact of Government Policies on Renewable Energy Production (GWh)')
plt.xlabel('Government Policies')
plt.ylabel('Total Production (GWh)')
plt.xticks(rotation=0)
plt.tight_layout()
# plt.show()


# ========== Phase 7: Time Series Analysis ==============================================================================================================================
# Year-over-year production trends
yearly_trends = df.groupby('Year').agg({
    'Production (GWh)': 'sum',
    'Investments (USD)': 'sum',
    'Installed Capacity (MW)': 'sum',
    'Proportion of Energy from Renewables': 'mean'
})

# Growth rates
yearly_trends['Production Growth Rate'] = yearly_trends['Production (GWh)'].pct_change() * 100 # Calculate percentage change for each year over the previous year
yearly_trends['Investment Growth Rate'] = yearly_trends['Investments (USD)'].pct_change() * 100
yearly_trends['Capacity Growth Rate'] = yearly_trends['Installed Capacity (MW)'].pct_change() * 100
yearly_trends['Renewable Energy Growth Rate'] = yearly_trends['Proportion of Energy from Renewables'].pct_change() * 100

print("Yearly Trends:\n", yearly_trends)

# Normalize the data for visualization
normalized_trends = yearly_trends.copy()
for col in ['Production (GWh)', 'Investments (USD)', 'Installed Capacity (MW)', 'Proportion of Energy from Renewables']:
    normalized_trends[col] = (yearly_trends[col] - yearly_trends[col].min()) / (yearly_trends[col].max() - yearly_trends[col].min()) # Normalize each column using min-max scaling where the minimum value is subtracted from each value and divided by the range (max - min)

# Plot normalized trends
plt.figure(figsize=(12, 6))
sns.lineplot(data=normalized_trends, x=normalized_trends.index, y='Production (GWh)', marker='o', label='Production (GWh)')
sns.lineplot(data=normalized_trends, x=normalized_trends.index, y='Investments (USD)', marker='o', label='Investments (USD)')
sns.lineplot(data=normalized_trends, x=normalized_trends.index, y='Installed Capacity (MW)', marker='o', label='Installed Capacity (MW)')
sns.lineplot(data=normalized_trends, x=normalized_trends.index, y='Proportion of Energy from Renewables', marker='o', label='Proportion of Renewables (%)')
plt.title('Yearly Trends in Renewable Energy Production and Investments (Normalized)')
plt.xlabel('Year')
plt.ylabel('Normalized Values')
plt.legend()
plt.xticks(yearly_trends.index, rotation=45)
plt.tight_layout()
# plt.show()

# ========== Phase 8: Conclusion and Recommendations ==============================================================================================================================
# Summary of findings
summary = {
    'Total Countries': df['Country'].nunique(),
    'Total Energy Types': df['Energy Type'].nunique(),
    'Total Years Covered': df['Year'].nunique(),
    'Total Production (GWh)': df['Production (GWh)'].sum(),
    'Total Investments (USD)': df['Investments (USD)'].sum(),
    'Total Installed Capacity (MW)': df['Installed Capacity (MW)'].sum(),
    'Average Proportion of Energy from Renewables': df['Proportion of Energy from Renewables'].mean(),
    
    'Top 3 Countries by Production': df.groupby('Country')['Production (GWh)'].sum().nlargest(3).to_dict(),
    'Bottom 3 Countries by Production': df.groupby('Country')['Production (GWh)'].sum().nsmallest(3).to_dict(),
    
    'Top 3 Countries by Investments': df.groupby('Country')['Investments (USD)'].sum().nlargest(3).to_dict(),
    'Bottom 3 Countries by Investments': df.groupby('Country')['Investments (USD)'].sum().nsmallest(3).to_dict(),
    
    'Top 3 Countries by Installed Capacity': df.groupby('Country')['Installed Capacity (MW)'].sum().nlargest(3).to_dict(),
    'Bottom 3 Countries by Installed Capacity': df.groupby('Country')['Installed Capacity (MW)'].sum().nsmallest(3).to_dict(),
    
    'Top 3 Countries by Proportion of Renewables': df.groupby('Country')['Proportion of Energy from Renewables'].mean().nlargest(3).to_dict(),
    'Bottom 3 Countries by Proportion of Renewables': df.groupby('Country')['Proportion of Energy from Renewables'].mean().nsmallest(3).to_dict(),
    
    'Distribution of Energy Types by Production': df.groupby('Energy Type')['Production (GWh)'].sum().nlargest(5).to_dict(),
    'Distribution of Energy Types by Investments': df.groupby('Energy Type')['Investments (USD)'].sum().nlargest(5).to_dict(),
    'Distribution of Energy Types by Installed Capacity': df.groupby('Energy Type')['Installed Capacity (MW)'].sum().nlargest(5).to_dict(),
    'Distribution of Energy Types by Proportion of Renewables': df.groupby('Energy Type')['Proportion of Energy from Renewables'].mean().nlargest(5).to_dict(),
    
    'Investment Efficiency (Top 3 Countries)': efficiency_by_country['Production_per_Investment'].nlargest(3).to_dict(),
    'Investment Efficiency (Bottom 3 Countries)': efficiency_by_country['Production_per_Investment'].nsmallest(3).to_dict(),
    
    'Production Efficiency (Top 3 Countries)': efficiency_by_country['Production_per_Capacity'].nlargest(3).to_dict(),
    'Production Efficiency (Bottom 3 Countries)': efficiency_by_country['Production_per_Capacity'].nsmallest(3).to_dict(),
    
    'Yearly Trends': yearly_trends[['Production (GWh)', 'Investments (USD)', 'Installed Capacity (MW)', 'Proportion of Energy from Renewables']].to_dict(orient='index'),
    'Policy Impact': policy_impact.to_dict(orient='index'),
    'Government Policies Impact': policy_correlation.to_dict(orient='index')
    
}

# Pretty print the summary
print("Summary of Findings:\n")
print(json.dumps(summary, indent=5))

yearly_trends_df = pd.DataFrame.from_dict(summary['Yearly Trends'], orient='index')
print("\nYearly Trends DataFrame:\n", yearly_trends_df)

# Save the summary to a JSON file
# summary_json_path = os.path.join(current_dir, 'renewable_energy_summary.json')
# with open(summary_json_path, 'w') as f:
#     json.dump(summary, f, indent=5)
# print(f"\nSummary saved to {summary_json_path}")

# # Save the suummary to a CSV file
# summary_csv_path = os.path.join(current_dir, 'renewable_energy_summary.csv')
# with open(summary_csv_path, 'w') as file:
#     pd.DataFrame.from_dict(summary, orient='index').to_csv(file) # option a
#     json.dump(summary, file, indent=5) # option b
# print(f"Summary saved to {summary_csv_path}")


# # Save the summary to a text file
# summary_text_path = os.path.join(current_dir, 'renewable_energy_summary.txt')
# with open(summary_text_path, 'w') as f:
#     f.write(json.dumps(summary, indent=5))
# print(f"Summary saved to {summary_text_path}")
