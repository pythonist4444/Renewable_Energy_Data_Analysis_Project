# 🌍 Renewable Energy Data Analysis Project

[![Python](https://img.shields.io/badge/Language-Python-green)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/Library-NumPy-blue)](https://numpy.org/)
[![Pandas](https://img.shields.io/badge/Library-Pandas-lightgrey)](https://pandas.pydata.org/)
[![Matplotlib](https://img.shields.io/badge/Library-Matplotlib-orange)](https://matplotlib.org/)
[![Seaborn](https://img.shields.io/badge/Library-Seaborn-cyan)](https://seaborn.pydata.org/)
[![Squarify](https://img.shields.io/badge/Library-Squarify-purple)](https://github.com/laserson/squarify)
[![MySQL](https://img.shields.io/badge/Database-MySQL-blue)](https://www.mysql.com/)



This project performs an in-depth exploratory data analysis (EDA) on a dataset containing information about renewable energy production, investments, capacity, and policy impacts across multiple countries and energy types over several years.

## 📁 Dataset

The dataset used is `Training_set_augmented.csv`, which contains the following key columns:

- `Country`
- `Energy Type`
- `Year`
- `Production (GWh)`
- `Investments (USD)`
- `Installed Capacity (MW)`
- `Proportion of Energy from Renewables`
- `Renewable Energy Targets` (0 = No, 1 = Yes)
- `Government Policies` (0 = No, 1 = Yes)

## 📊 Project Objectives

The main objectives of this project are:

1. Conduct initial data exploration.
2. Analyze renewable energy by:
   - Country
   - Energy type
   - Investment and production efficiency
3. Evaluate the impact of government policies and targets.
4. Identify trends across time.
5. Present visualizations for deeper insight.

---

## 📌 Key Features

### ✅ **Phase 1: Initial Exploration**

- Dataset structure, missing values, duplicates
- Summary statistics
- Unique countries, energy types, and years

### 🌐 **Phase 2: Country-Level Analysis**

- Total production, investment, and capacity per country
- Proportion of renewables in each country
- Visual comparisons using bar charts

### ⚡ **Phase 3: Energy Type Analysis**

- Total values grouped by energy type
- Pie, donut, and treemap visualizations for:
  - Production
  - Investments
  - Installed capacity
  - Renewable proportion

### 📈 **Phase 4: Efficiency Analysis**

- `Production_per_Investment` (GWh per billion USD)
- `Production_per_Capacity` (GWh per MW)
- Ranked efficiency by country

### 🧠 **Phase 5: Correlation Analysis**

- Correlation matrix of key numerical features
- Focused analysis on factors influencing `Production (GWh)`
- Scatter plots showing:
  - Investments vs. Production
  - Capacity vs. Production
  - Investment Efficiency vs. Capacity Efficiency

### 🏛️ **Phase 6: Policy Impact Analysis**

- Comparing countries with/without renewable energy targets
- Evaluating effect of government policies on:
  - Production
  - Investments
  - Capacity
  - Renewable share

### ⏳ **Phase 7: Time Series Analysis**

- Annual trends of key metrics:
  - Production
  - Investments
  - Capacity
  - Renewable share
- Growth rate calculations
- Normalized trend visualization

### 📌 **Phase 8: Summary & Recommendations**

- Highlights:
  - Top & bottom countries by key metrics
  - Most/least efficient countries
  - Best-performing energy types
- JSON summary printed and optionally saved

---

## 📂 Project Structure

Renewable Energy Project/

├── Visuals folder                     
├── Training_set_augmented.csv                   
├── renewable.py  
├── insights.txt                  
├── renewable_energy_summary.json (optional output)                
├── renewable_energy_summary.txt (optional output)                  
├── renewable_energy_summary.csv (optional output)               


---

## 🛠️ Libraries Used

- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `squarify`
- `json`
- `os`

---

## 📷 Visualizations

All visualizations are generated using `matplotlib` and `seaborn`. These include:

- Bar plots
- Pie charts
- Donut charts
- Tree maps
- Line charts
- Scatter plots

> **Note:** All `plt.show()` calls are currently commented out to avoid blocking execution in automated or non-interactive environments. Uncomment as needed for visualization.

---

## 📌 How to Run

Ensure you have Python 3 installed, along with the required libraries:

```bash
pip install pandas numpy matplotlib seaborn squarify
python renewable_eda.py
