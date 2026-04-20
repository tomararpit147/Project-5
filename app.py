"""
Real Estate Investment Advisor — Streamlit App
Run: streamlit run app.py
"""

import os
if not os.path.exists('models/regressor.pkl'):
    import train_models
    
import streamlit as st 
import pandas as pd
import numpy as np 
import pickle
import os
import matplotlib.pyplot as plt 
import seaborn as sns 

# ── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🏠 Real Estate Investment Advisor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── LOAD MODELS ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    models_dir = os.path.join(os.path.dirname(__file__), 'models')
    cls   = pickle.load(open(os.path.join(models_dir, 'classifier.pkl'), 'rb'))
    reg   = pickle.load(open(os.path.join(models_dir, 'regressor.pkl'), 'rb'))
    scaler = pickle.load(open(os.path.join(models_dir, 'scaler.pkl'), 'rb'))
    le_dict = pickle.load(open(os.path.join(models_dir, 'label_encoders.pkl'), 'rb'))
    feat_cols = pickle.load(open(os.path.join(models_dir, 'feature_cols.pkl'), 'rb'))
    return cls, reg, scaler, le_dict, feat_cols

@st.cache_data
def load_data():
    data_path = os.path.join(os.path.dirname(__file__), 'india_housing_prices_cleaned.csv')
    if not os.path.exists(data_path):
        data_path = os.path.join(os.path.dirname(__file__), 'india_housing_prices.csv')
    return pd.read_csv(data_path)

try:
    cls, reg, scaler, le_dict, feat_cols = load_models()
    df = load_data()
    models_loaded = True
except Exception as e:
    models_loaded = False
    st.error(f"⚠️ Could not load models: {e}. Please run `python train_models.py` first.")

# ── SIDEBAR ──────────────────────────────────────────────────────────────────
st.sidebar.title("🏠 Navigation")
page = st.sidebar.radio("Go to", ["🔮 Predict", "📊 EDA Dashboard", "📈 Model Performance", "ℹ️ About"])

st.sidebar.markdown("---")
st.sidebar.markdown("**Dataset Stats**")
if models_loaded:
    st.sidebar.metric("Total Properties", f"{len(df):,}")
    st.sidebar.metric("Cities", df['City'].nunique())
    st.sidebar.metric("States", df['State'].nunique())

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 1: PREDICTION
# ─────────────────────────────────────────────────────────────────────────────
if page == "🔮 Predict":
    st.title("🔮 Real Estate Investment Advisor")
    st.markdown("Enter property details below to get **investment advice** and a **5-year price forecast**.")

    if not models_loaded:
        st.stop()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("📍 Location")
        state = st.selectbox("State", sorted(df['State'].unique()))
        cities_in_state = sorted(df[df['State']==state]['City'].unique())
        city = st.selectbox("City", cities_in_state)
        localities = sorted(df[df['City']==city]['Locality'].unique())
        locality = st.selectbox("Locality", localities)

    with col2:
        st.subheader("🏗️ Property Details")
        property_type = st.selectbox("Property Type", sorted(df['Property_Type'].unique()))
        bhk = st.slider("BHK", 1, 5, 3)
        size_sqft = st.number_input("Size (SqFt)", min_value=200, max_value=10000, value=1200, step=50)
        price_lakhs = st.number_input("Current Price (₹ Lakhs)", min_value=5.0, max_value=5000.0, value=80.0, step=5.0)
        year_built = st.number_input("Year Built", min_value=1950, max_value=2024, value=2015)
        floor_no = st.number_input("Floor Number", min_value=0, max_value=50, value=3)
        total_floors = st.number_input("Total Floors", min_value=1, max_value=60, value=10)

    with col3:
        st.subheader("🏡 Amenities & Features")
        furnished = st.selectbox("Furnished Status", sorted(df['Furnished_Status'].unique()))
        parking = st.selectbox("Parking Space", sorted(df['Parking_Space'].unique()))
        security = st.selectbox("Security", sorted(df['Security'].unique()))
        amenities = st.selectbox("Amenities", sorted(df['Amenities'].unique()))
        facing = st.selectbox("Facing", sorted(df['Facing'].unique()))
        owner_type = st.selectbox("Owner Type", sorted(df['Owner_Type'].unique()))
        availability = st.selectbox("Availability Status", sorted(df['Availability_Status'].unique()))
        transport = st.selectbox("Public Transport", sorted(df['Public_Transport_Accessibility'].unique()))
        nearby_schools = st.slider("Nearby Schools (1–10)", 1, 10, 5)
        nearby_hospitals = st.slider("Nearby Hospitals (1–10)", 1, 10, 5)

    st.markdown("---")
    if st.button("🔍 Analyze Property", use_container_width=True, type="primary"):
        # Build input dict
        age = 2024 - year_built
        ppsf = price_lakhs * 100000 / size_sqft
        infra = nearby_schools + nearby_hospitals
        floor_ratio = floor_no / (total_floors + 1)
        school_density = nearby_schools / (size_sqft / 1000 + 1)

        # Encode categoricals safely
        def safe_encode(le, val):
            classes = list(le.classes_)
            return classes.index(val) if val in classes else 0

        input_data = {
            'BHK': bhk, 'Size_in_SqFt': size_sqft, 'Price_in_Lakhs': price_lakhs,
            'Price_per_SqFt': ppsf, 'Age_of_Property': age,
            'Nearby_Schools': nearby_schools, 'Nearby_Hospitals': nearby_hospitals,
            'Floor_No': floor_no, 'Total_Floors': total_floors,
            'Floor_Ratio': floor_ratio, 'Infra_Score': infra,
            'School_Density_Score': school_density,
            'Property_Type_enc':   safe_encode(le_dict['Property_Type'], property_type),
            'Furnished_Status_enc': safe_encode(le_dict['Furnished_Status'], furnished),
            'Public_Transport_Accessibility_enc': safe_encode(le_dict['Public_Transport_Accessibility'], transport),
            'Parking_Space_enc':   safe_encode(le_dict['Parking_Space'], parking),
            'Security_enc':        safe_encode(le_dict['Security'], security),
            'Amenities_enc':       safe_encode(le_dict['Amenities'], amenities),
            'Facing_enc':          safe_encode(le_dict['Facing'], facing),
            'Owner_Type_enc':      safe_encode(le_dict['Owner_Type'], owner_type),
            'Availability_Status_enc': safe_encode(le_dict['Availability_Status'], availability),
            'State_enc':           safe_encode(le_dict['State'], state),
            'City_enc':            safe_encode(le_dict['City'], city),
        }
        input_df = pd.DataFrame([input_data])[feat_cols]
        input_sc = scaler.transform(input_df)

        # Predictions
        gi_pred  = cls.predict(input_sc)[0]
        gi_proba = cls.predict_proba(input_sc)[0][1]
        fp_pred  = reg.predict(input_sc)[0]

        # Show results
        r1, r2, r3 = st.columns(3)
        with r1:
            color = "🟢" if gi_pred == 1 else "🔴"
            verdict = "GOOD INVESTMENT" if gi_pred == 1 else "NOT RECOMMENDED"
            st.metric("Investment Verdict", f"{color} {verdict}")
        with r2:
            st.metric("Investment Confidence", f"{gi_proba*100:.1f}%",
                      delta=f"{'▲ Positive' if gi_pred==1 else '▼ Caution'}")
        with r3:
            appreciation = ((fp_pred - price_lakhs) / price_lakhs) * 100
            st.metric("Estimated Price in 5 Years", f"₹{fp_pred:.1f} Lakhs",
                      delta=f"+{appreciation:.1f}% appreciation")

        # Visual: Price growth timeline
        years = list(range(0, 6))
        city_rate = df[df['City']==city]['City_Growth_Rate'].mean() if 'City_Growth_Rate' in df.columns else 0.08
        if pd.isna(city_rate): city_rate = 0.08
        prices_over_time = [price_lakhs * (1 + city_rate) ** y for y in years]

        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(years, prices_over_time, marker='o', color='#4C72B0', linewidth=2.5, markersize=7)
        ax.fill_between(years, prices_over_time, alpha=0.15, color='#4C72B0')
        ax.axhline(y=price_lakhs, color='gray', linestyle='--', alpha=0.5, label='Current Price')
        ax.set(title=f'Projected Price Growth — {city}', xlabel='Years', ylabel='Price (₹ Lakhs)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig, use_container_width=True)
        plt.close()

        # Key insight
        if gi_pred == 1:
            st.success(f"""
            ✅ **Investment Summary:** This property in **{locality}, {city}** appears to be a **good investment**.
            At ₹{ppsf:.0f}/SqFt with an infrastructure score of {infra}/20, it offers solid value.
            Expected 5-year appreciation: **+{appreciation:.1f}%** (₹{fp_pred - price_lakhs:.1f} Lakhs gain).
            """)
        else:
            st.warning(f"""
            ⚠️ **Investment Summary:** This property may **not be the best investment** right now.
            The price per SqFt (₹{ppsf:.0f}) is above the market median or amenity/infra scores are lower.
            Consider negotiating the price or exploring other localities in {city}.
            """)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 2: EDA DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────
elif page == "📊 EDA Dashboard":
    st.title("📊 Exploratory Data Analysis Dashboard")

    if not models_loaded:
        st.stop()

    tab1, tab2, tab3, tab4 = st.tabs(["Price & Size", "Location", "Feature Relations", "Investment Factors"])

    with tab1:
        st.subheader("Price & Size Analysis")
        c1, c2 = st.columns(2)
        with c1:
            fig, ax = plt.subplots()
            ax.hist(df['Price_in_Lakhs'], bins=50, color='#4C72B0', edgecolor='white')
            ax.set(title='Property Price Distribution', xlabel='Price (Lakhs)', ylabel='Count')
            st.pyplot(fig); plt.close()
        with c2:
            fig, ax = plt.subplots()
            ax.hist(df['Size_in_SqFt'], bins=50, color='#55A868', edgecolor='white')
            ax.set(title='Property Size Distribution', xlabel='Size (SqFt)', ylabel='Count')
            st.pyplot(fig); plt.close()

        fig, ax = plt.subplots(figsize=(10, 4))
        pt_ppsf = df.groupby('Property_Type')['Price_per_SqFt'].median().sort_values()
        ax.barh(pt_ppsf.index, pt_ppsf.values, color='#C44E52')
        ax.set(title='Median Price/SqFt by Property Type', xlabel='Median Price/SqFt')
        st.pyplot(fig); plt.close()

    with tab2:
        st.subheader("Location Analysis")
        n_top = st.slider("Number of top locations to show", 5, 20, 10)
        c1, c2 = st.columns(2)
        with c1:
            top_states = df.groupby('State')['Price_per_SqFt'].mean().nlargest(n_top)
            fig, ax = plt.subplots()
            ax.barh(top_states.index, top_states.values, color='#4C72B0')
            ax.set(title=f'Top {n_top} States by Avg Price/SqFt', xlabel='Avg Price/SqFt')
            st.pyplot(fig); plt.close()
        with c2:
            top_cities = df.groupby('City')['Price_in_Lakhs'].mean().nlargest(n_top)
            fig, ax = plt.subplots()
            ax.barh(top_cities.index, top_cities.values, color='#DD8452')
            ax.set(title=f'Top {n_top} Cities by Avg Price', xlabel='Avg Price (Lakhs)')
            st.pyplot(fig); plt.close()

        # Filter by state
        selected_state = st.selectbox("Explore State", sorted(df['State'].unique()))
        state_df = df[df['State'] == selected_state]
        fig, ax = plt.subplots(figsize=(10, 4))
        bhk_city = state_df.groupby('City')['Price_in_Lakhs'].mean().sort_values(ascending=False).head(10)
        ax.bar(bhk_city.index, bhk_city.values, color='#8172B2')
        ax.set(title=f'Average Property Prices — {selected_state}', ylabel='Price (Lakhs)')
        ax.tick_params(axis='x', rotation=45)
        st.pyplot(fig); plt.close()

    with tab3:
        st.subheader("Feature Correlations")
        num_cols = ['BHK','Size_in_SqFt','Price_in_Lakhs','Price_per_SqFt',
                    'Age_of_Property','Nearby_Schools','Nearby_Hospitals']
        if 'Infra_Score' in df.columns:
            num_cols.append('Infra_Score')
        if 'Good_Investment' in df.columns:
            num_cols.append('Good_Investment')
        fig, ax = plt.subplots(figsize=(10, 7))
        sns.heatmap(df[num_cols].corr(), annot=True, fmt='.2f', cmap='coolwarm', ax=ax)
        ax.set_title('Numeric Feature Correlation Heatmap')
        st.pyplot(fig); plt.close()

        c1, c2 = st.columns(2)
        with c1:
            fig, ax = plt.subplots()
            furn = df.groupby('Furnished_Status')['Price_in_Lakhs'].median()
            ax.bar(furn.index, furn.values, color=['#4C72B0','#55A868','#C44E52'])
            ax.set(title='Median Price by Furnished Status', ylabel='Price (Lakhs)')
            st.pyplot(fig); plt.close()
        with c2:
            fig, ax = plt.subplots()
            facing = df.groupby('Facing')['Price_per_SqFt'].median().sort_values()
            ax.barh(facing.index, facing.values, color='#8172B2')
            ax.set(title='Price/SqFt by Facing Direction', xlabel='Median Price/SqFt')
            st.pyplot(fig); plt.close()

    with tab4:
        st.subheader("Investment & Ownership Analysis")
        c1, c2 = st.columns(2)
        with c1:
            fig, ax = plt.subplots()
            owner = df['Owner_Type'].value_counts()
            ax.bar(owner.index, owner.values, color='#4C72B0')
            ax.set(title='Properties by Owner Type', ylabel='Count')
            st.pyplot(fig); plt.close()
        with c2:
            fig, ax = plt.subplots()
            avail = df['Availability_Status'].value_counts()
            ax.bar(avail.index, avail.values, color='#DD8452')
            ax.set(title='Availability Status', ylabel='Count')
            ax.tick_params(axis='x', rotation=15)
            st.pyplot(fig); plt.close()

        if 'Good_Investment' in df.columns:
            fig, ax = plt.subplots(figsize=(10, 4))
            transport_gi = df.groupby('Public_Transport_Accessibility')['Good_Investment'].mean() * 100
            ax.bar(transport_gi.index, transport_gi.values, color='#55A868')
            ax.set(title='Good Investment Rate by Transport Accessibility (%)',
                   ylabel='% Good Investment', xlabel='Transport Level')
            st.pyplot(fig); plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 3: MODEL PERFORMANCE
# ─────────────────────────────────────────────────────────────────────────────
elif page == "📈 Model Performance":
    st.title("📈 Model Performance & Evaluation")

    import json
    metrics_path = os.path.join(os.path.dirname(__file__), 'models', 'metrics_summary.json')
    if os.path.exists(metrics_path):
        m = json.load(open(metrics_path))

        st.subheader("🏆 Best Models")
        c1, c2 = st.columns(2)
        with c1:
            st.success(f"**Best Classifier:** {m['best_classifier']}")
            st.metric("Accuracy", f"{m['cls_accuracy']*100:.2f}%")
            st.metric("F1 Score", f"{m['cls_f1']:.4f}")
        with c2:
            st.info(f"**Best Regressor:** {m['best_regressor']}")
            st.metric("RMSE", f"₹{m['reg_rmse']:.2f} Lakhs")
            st.metric("MAE", f"₹{m['reg_mae']:.2f} Lakhs")
            st.metric("R² Score", f"{m['reg_r2']:.4f}")

        if 'all_classifiers' in m:
            st.subheader("📊 All Classifier Results")
            cls_df = pd.DataFrame(m['all_classifiers']).T.reset_index()
            cls_df.columns = ['Model','Accuracy','F1 Score']
            st.dataframe(cls_df, use_container_width=True)

            fig, ax = plt.subplots(figsize=(8, 3))
            x = np.arange(len(cls_df))
            ax.bar(x - 0.2, cls_df['Accuracy'], 0.4, label='Accuracy', color='#4C72B0')
            ax.bar(x + 0.2, cls_df['F1 Score'], 0.4, label='F1 Score', color='#C44E52')
            ax.set_xticks(x); ax.set_xticklabels(cls_df['Model'], rotation=10)
            ax.set(title='Classifier Comparison', ylim=(0, 1)); ax.legend()
            st.pyplot(fig); plt.close()

        if 'all_regressors' in m:
            st.subheader("📊 All Regressor Results")
            reg_df = pd.DataFrame(m['all_regressors']).T.reset_index()
            reg_df.columns = ['Model','RMSE','MAE','R2']
            st.dataframe(reg_df, use_container_width=True)
    else:
        st.warning("Metrics file not found. Please run `python train_models.py` first.")

    # Show saved plots
    for plot_name, title in [
        ('plots/05_model_evaluation.png', 'Confusion Matrix & Actual vs Predicted'),
        ('plots/01_price_size_analysis.png', 'Price & Size Analysis'),
        ('plots/03_correlation_heatmap.png', 'Correlation Heatmap'),
    ]:
        plot_path = os.path.join(os.path.dirname(__file__), plot_name)
        if os.path.exists(plot_path):
            st.subheader(title)
            st.image(plot_path, use_column_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 4: ABOUT
# ─────────────────────────────────────────────────────────────────────────────
elif page == "ℹ️ About":
    st.title("ℹ️ About This Project")
    st.markdown("""
    ## 🏠 Real Estate Investment Advisor

    ### Problem Statement
    This machine learning application assists real estate investors with two key decisions:
    - **Classification** — Is this property a *Good Investment*?
    - **Regression** — What will the property be worth *in 5 years*?

    ### Target Variables
    | Target | Type | Description |
    |--------|------|-------------|
    | `Good_Investment` | Binary (0/1) | 1 if: price ≤ median, BHK ≥ 2, infra score ≥ 8 |
    | `Future_Price_5yr` | Continuous | Current price × (1 + city growth rate)^5 |

    ### Models Trained
    - **Logistic Regression** — Baseline classifier
    - **Random Forest** — Ensemble classifier & regressor
    - **Gradient Boosting** — Advanced ensemble
    - **Ridge Regression** — Regularized linear regressor

    ### Key Engineered Features
    - `Infra_Score` = Nearby Schools + Nearby Hospitals
    - `Floor_Ratio` = Floor / Total Floors
    - `School_Density_Score` = Schools per 1000 SqFt
    - `City_Growth_Rate` = Location-adjusted appreciation rate

    ### Tech Stack
    `Python` · `Pandas` · `Scikit-learn` · `Streamlit` · `MLflow` · `Matplotlib` · `Seaborn`

    ### EDA Highlights (20 Questions Answered)
    1. Price distribution is right-skewed with outliers in metro cities
    2. Apartment type dominates the market
    3. Maharashtra and Goa have the highest avg price/SqFt
    4. Fully furnished properties command 15-25% premium
    5. North-facing properties tend to be priced highest
    6. Properties with better transport access show stronger investment returns

    ---
    *Project: Real Estate Investment Advisor — Predicting Property Profitability & Future Value*
    """)
