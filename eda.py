"""
EDA Script — Real Estate Investment Advisor
Run: python eda.py
"""
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
import warnings, os
warnings.filterwarnings('ignore')
sns.set_theme(style='whitegrid', palette='husl')
os.makedirs('plots', exist_ok=True)

df = pd.read_csv('india_housing_prices.csv')
print(f"Shape: {df.shape}")
print(f"Missing values: {df.isnull().sum().sum()}")
print(f"Duplicates: {df.duplicated().sum()}")
df.drop_duplicates(inplace=True)

# ── Feature Engineering ─────────────────────────────────────────────────────
df['Infra_Score'] = df['Nearby_Schools'] + df['Nearby_Hospitals']
df['Floor_Ratio'] = df['Floor_No'] / (df['Total_Floors'] + 1)
city_growth = df.groupby('City')['Price_in_Lakhs'].median()
city_growth_rate = ((city_growth - city_growth.min()) /
                    (city_growth.max() - city_growth.min()) * 0.04) + 0.06
df['City_Growth_Rate'] = df['City'].map(city_growth_rate)
df['Future_Price_5yr'] = df['Price_in_Lakhs'] * (1 + df['City_Growth_Rate']) ** 5
median_ppsf = df['Price_per_SqFt'].median()
df['Good_Investment'] = (
    (df['Price_per_SqFt'] <= median_ppsf) &
    (df['BHK'] >= 2) &
    (df['Infra_Score'] >= 8)
).astype(int)
print(f"\nGood Investment ratio: {df['Good_Investment'].mean():.1%}")

# ── 1–5: Price & Size Analysis ───────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Price & Size Analysis (Q1–Q5)', fontsize=15, fontweight='bold')

axes[0,0].hist(df['Price_in_Lakhs'], bins=60, color='#4C72B0', edgecolor='white')
axes[0,0].set(title='Q1: Distribution of Property Prices', xlabel='Price (Lakhs)', ylabel='Count')

axes[0,1].hist(df['Size_in_SqFt'], bins=60, color='#55A868', edgecolor='white')
axes[0,1].set(title='Q2: Distribution of Property Sizes', xlabel='Size (SqFt)')

pt_ppsf = df.groupby('Property_Type')['Price_per_SqFt'].median().sort_values()
axes[0,2].barh(pt_ppsf.index, pt_ppsf.values, color='#C44E52')
axes[0,2].set(title='Q3: Price/SqFt by Property Type', xlabel='Median Price/SqFt')

samp = df.sample(5000, random_state=42)
axes[1,0].scatter(samp['Size_in_SqFt'], samp['Price_in_Lakhs'], alpha=0.3, s=5, color='#8172B2')
axes[1,0].set(title='Q4: Size vs Price', xlabel='Size (SqFt)', ylabel='Price (Lakhs)')

axes[1,1].boxplot(df['Price_per_SqFt'])
axes[1,1].set(title='Q5: Price/SqFt Outliers', ylabel='Price/SqFt')

gi = df['Good_Investment'].value_counts()
axes[1,2].bar(['Not Good','Good Investment'], gi.values, color=['#C44E52','#55A868'])
axes[1,2].set(title='Target: Good Investment Split', ylabel='Count')

plt.tight_layout()
plt.savefig('plots/01_price_size_analysis.png', dpi=120, bbox_inches='tight')
plt.close()
print("Saved: plots/01_price_size_analysis.png")

# ── 6–10: Location-Based Analysis ────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Location-Based Analysis (Q6–Q10)', fontsize=14, fontweight='bold')

top_states = df.groupby('State')['Price_per_SqFt'].mean().nlargest(10)
axes[0,0].barh(top_states.index, top_states.values, color='#4C72B0')
axes[0,0].set(title='Q6: Top 10 States by Avg Price/SqFt', xlabel='Avg Price/SqFt')

top_cities = df.groupby('City')['Price_in_Lakhs'].mean().nlargest(10)
axes[0,1].barh(top_cities.index, top_cities.values, color='#DD8452')
axes[0,1].set(title='Q7: Top 10 Cities by Avg Price', xlabel='Avg Price (Lakhs)')

med_age = df.groupby('Locality')['Age_of_Property'].median().nlargest(10)
axes[1,0].barh(med_age.index, med_age.values, color='#55A868')
axes[1,0].set(title='Q8: Top 10 Localities by Median Property Age', xlabel='Median Age (Years)')

bhk_city = df.groupby('City')['BHK'].mean().nlargest(10)
axes[1,1].barh(bhk_city.index, bhk_city.values, color='#C44E52')
axes[1,1].set(title='Q9: Avg BHK by City (Top 10)', xlabel='Avg BHK')

plt.tight_layout()
plt.savefig('plots/02_location_analysis.png', dpi=120, bbox_inches='tight')
plt.close()
print("Saved: plots/02_location_analysis.png")

# ── 11–15: Feature Relationships ─────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Feature Relationships (Q11–Q15)', fontsize=14, fontweight='bold')

num_cols = ['BHK','Size_in_SqFt','Price_in_Lakhs','Price_per_SqFt',
            'Age_of_Property','Nearby_Schools','Nearby_Hospitals','Infra_Score']
sns.heatmap(df[num_cols].corr(), annot=True, fmt='.2f', cmap='coolwarm', ax=axes[0,0])
axes[0,0].set_title('Q11: Numeric Feature Correlations')

axes[0,1].scatter(df['Nearby_Schools'], df['Price_per_SqFt'], alpha=0.05, s=2)
axes[0,1].set(title='Q12: Schools vs Price/SqFt', xlabel='Nearby Schools', ylabel='Price/SqFt')

axes[0,2].scatter(df['Nearby_Hospitals'], df['Price_per_SqFt'], alpha=0.05, s=2, color='#C44E52')
axes[0,2].set(title='Q13: Hospitals vs Price/SqFt', xlabel='Nearby Hospitals', ylabel='Price/SqFt')

furn_price = df.groupby('Furnished_Status')['Price_in_Lakhs'].median()
axes[1,0].bar(furn_price.index, furn_price.values, color=['#4C72B0','#55A868','#C44E52'])
axes[1,0].set(title='Q14: Median Price by Furnished Status', ylabel='Price (Lakhs)')

facing_ppsf = df.groupby('Facing')['Price_per_SqFt'].median().sort_values()
axes[1,1].barh(facing_ppsf.index, facing_ppsf.values, color='#8172B2')
axes[1,1].set(title='Q15: Price/SqFt by Facing Direction', xlabel='Median Price/SqFt')

axes[1,2].axis('off')
plt.tight_layout()
plt.savefig('plots/03_feature_relationships.png', dpi=120, bbox_inches='tight')
plt.close()
print("Saved: plots/03_feature_relationships.png")

# ── 16–20: Investment / Amenities / Ownership ─────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Investment, Amenities & Ownership (Q16–Q20)', fontsize=14, fontweight='bold')

owner_cnt = df['Owner_Type'].value_counts()
axes[0,0].bar(owner_cnt.index, owner_cnt.values, color='#4C72B0')
axes[0,0].set(title='Q16: Properties by Owner Type', ylabel='Count')

avail_cnt = df['Availability_Status'].value_counts()
axes[0,1].bar(avail_cnt.index, avail_cnt.values, color='#DD8452')
axes[0,1].set(title='Q17: Availability Status Distribution', ylabel='Count')
axes[0,1].tick_params(axis='x', rotation=15)

park_price = df.groupby('Parking_Space')['Price_in_Lakhs'].median()
axes[0,2].bar(park_price.index, park_price.values, color='#55A868')
axes[0,2].set(title='Q18: Median Price by Parking Space', ylabel='Price (Lakhs)')

amen_ppsf = df.groupby('Amenities')['Price_per_SqFt'].median().nlargest(8)
axes[1,0].barh(amen_ppsf.index, amen_ppsf.values, color='#C44E52')
axes[1,0].set(title='Q19: Top Amenities by Price/SqFt', xlabel='Median Price/SqFt')

trans_ppsf = df.groupby('Public_Transport_Accessibility')['Price_per_SqFt'].median().sort_values()
axes[1,1].barh(trans_ppsf.index, trans_ppsf.values, color='#8172B2')
axes[1,1].set(title='Q20: Transport Access vs Price/SqFt', xlabel='Median Price/SqFt')

axes[1,2].axis('off')
plt.tight_layout()
plt.savefig('plots/04_investment_analysis.png', dpi=120, bbox_inches='tight')
plt.close()
print("Saved: plots/04_investment_analysis.png")

print("\n✅ EDA Complete! All plots saved to ./plots/")
