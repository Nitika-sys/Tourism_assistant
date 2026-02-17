import streamlit as st
import pandas as pd
import pickle
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="AI Tourism Assistant", layout="wide", page_icon="🌍")

# --- 2. MODEL LOADING WITH SAFETY CHECK ---
required_files = ['reg_model.pkl', 'clf_model.pkl', 'le_vmode.pkl', 'user_item_matrix.pkl']
missing_files = [f for f in required_files if not os.path.exists(f)]

if missing_files:
    st.error(f"🚨 Missing Model Files: {missing_files}")
    st.info("Please run 'python model_train.py' first to generate these files.")
    st.stop()

@st.cache_resource
def load_models():
    reg = pickle.load(open('reg_model.pkl', 'rb'))
    clf = pickle.load(open('clf_model.pkl', 'rb'))
    le = pickle.load(open('le_vmode.pkl', 'rb'))
    matrix = pickle.load(open('user_item_matrix.pkl', 'rb'))
    return reg, clf, le, matrix

reg_model, clf_model, le_vmode, ui_matrix = load_models()

@st.cache_data
def load_data():
    df_trans = pd.read_csv('master_tourism_data.csv')
    if os.path.exists('master_items.csv'):
        df_items = pd.read_csv('master_items.csv')
    else:
        st.error("master_items.csv not found. Please run data_pipeline.py.")
        st.stop()

    df_country = pd.read_excel('Country.xlsx')
    df_region = pd.read_excel('Region.xlsx')
    df_cont = pd.read_excel('Continent.xlsx')

    for d in [df_country, df_region, df_cont]:
        d.columns = d.columns.str.strip()

    country_geo = pd.merge(df_country, df_region[['RegionId', 'ContinentId']], on='RegionId', how='left')
    country_geo = pd.merge(country_geo, df_cont, on='ContinentId', how='left')

    id_cols = ['ContinentId', 'CountryId', 'RegionId', 'CityId', 'VisitMonth', 'AttractionTypeId', 'UserId', 'AttractionId']
    for col in id_cols:
        if col in df_trans.columns:
            df_trans[col] = pd.to_numeric(df_trans[col], errors='coerce').fillna(0).astype(int)

    for col in ['AttractionId', 'CityId', 'CountryId', 'ContinentId', 'AttractionTypeId']:
        if col in df_items.columns:
            df_items[col] = pd.to_numeric(df_items[col], errors='coerce').fillna(0).astype(int)

    for col in ['CountryId', 'RegionId', 'ContinentId']:
        if col in country_geo.columns:
            country_geo[col] = pd.to_numeric(country_geo[col], errors='coerce').fillna(0).astype(int)

    return df_trans, df_items, country_geo, df_cont

df_trans, df_items, country_geo, df_cont = load_data()

# --- 3. SIDEBAR INPUTS ---
st.sidebar.header("🗺️ User Trip Details")

# User selection (from transactions)
unique_users = sorted(df_trans['UserId'].unique())
selected_user = st.sidebar.selectbox("Identify User (by ID)", unique_users)

# Location filters based on dimension tables
valid_continents = sorted([c for c in country_geo['Continent'].astype(str).unique() if c not in ['-', 'Unknown']])
selected_continent = st.sidebar.selectbox("Target Continent", valid_continents)

filtered_countries_df = country_geo[country_geo['Continent'] == selected_continent]
filtered_countries = sorted([c for c in filtered_countries_df['Country'].astype(str).unique() if c != '-'])
selected_country = st.sidebar.selectbox("Target Country", filtered_countries)

# Trip Context
selected_month = st.sidebar.slider("Month of Travel", 1, 12, 6)

# Interest Category (from ALL ITEMS)
valid_types = sorted([t for t in df_items['AttractionType'].unique().astype(str) if t != 'Other'])
selected_type = st.sidebar.selectbox("Interest Category", valid_types)

try:
    cont_row = filtered_countries_df.iloc[0]
    cont_id = int(cont_row['ContinentId'])

    country_row = filtered_countries_df[filtered_countries_df['Country'] == selected_country].iloc[0]
    count_id = int(country_row['CountryId'])

    type_row = df_items[df_items['AttractionType'] == selected_type].iloc[0]
    attr_type_id = int(type_row['AttractionTypeId'])
except:
    cont_id, count_id, attr_type_id = 0, 0, 0

# --- 4. MAIN INTERFACE ---
st.title("🌍 Smart Tourism Recommendation System")
st.markdown(f"**Welcome, User {selected_user}!** Plan your next adventure with AI-driven insights.")

tab1, tab2, tab3 = st.tabs(["🔮 Trip Prediction", "🎯 Recommendations", "📊 Market Trends"])

# --- TAB 1: REGRESSION & CLASSIFICATION ---
with tab1:
    st.header("Predictive Insights")
    st.info("Our AI models analyze the profile to predict travel behavior and satisfaction.")
    
    if st.button("🚀 Run Prediction Analysis"):
        # Features: [ContinentId, CountryId, RegionId, CityId, VisitMonth, AttractionTypeId]
        # We use 0 for RegionId and CityId as generic prediction for Country
        input_data = np.array([[cont_id, count_id, 0, 0, selected_month, attr_type_id]])
        
        # 1. Classification (Visit Mode)
        mode_idx = clf_model.predict(input_data)[0]
        try:
            mode_name = le_vmode.inverse_transform([mode_idx])[0]
        except:
            mode_name = "Standard Visit"
        
        # 2. Regression (Rating)
        predicted_rating = reg_model.predict(input_data)[0]
        
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Predicted Visit Mode", mode_name)
        with c2:
            st.metric("Estimated Satisfaction Rating", f"{predicted_rating:.2f} / 5.0")
            
        if predicted_rating > 4:
            st.success("This trip aligns perfectly with the historical preferences!")
        else:
            st.warning("This attraction might not fully meet expectations based on the profile.")

# --- TAB 2: RECOMMENDATION ENGINE ---
with tab2:
    st.header("Personalized Attraction Suggestions")
    st.write(f"Finding unique matches in **{selected_country}** for **{selected_type}**")
    
    if st.button("🔍 Find Top Suggestions"):
        candidates = df_items[
            (df_items['Country'] == selected_country)
            & (df_items['AttractionType'] == selected_type)
        ].copy()
        fallback_note = None

        if candidates.empty:
            fallback = df_items[
                (df_items['Continent'] == selected_continent)
                & (df_items['AttractionType'] == selected_type)
            ].copy()
            if not fallback.empty:
                fallback_note = f"Not enough data for {selected_type} in {selected_country}. Showing similar places in {selected_continent}."
                candidates = fallback
            else:
                fallback = df_items[df_items['AttractionType'] == selected_type].copy()
                if not fallback.empty:
                    fallback_note = f"Not enough data for {selected_type} in {selected_country}. Showing similar places worldwide."
                    candidates = fallback
                else:
                    fallback = df_items.copy()
                    fallback_note = "Not enough data for this interest. Showing top attractions worldwide."
                    candidates = fallback

        if candidates.empty:
            st.warning("No attractions available to recommend.")
        else:
            if fallback_note:
                st.caption(fallback_note)

            scored_recommendations = []
            user_weights = None
            if selected_user in ui_matrix.index:
                user_vector = ui_matrix.loc[[selected_user]]
                similarities = cosine_similarity(user_vector, ui_matrix)[0]
                user_weights = pd.Series(similarities, index=ui_matrix.index)

            for _, item in candidates.iterrows():
                attr_id = item['AttractionId']
                attr_name = item['Attraction']
                city_name = item['CityName']

                score = 0.0
                source = "New Gem 💎"

                if attr_id in ui_matrix.columns and user_weights is not None:
                    users_who_rated = ui_matrix[ui_matrix[attr_id] > 0][attr_id]

                    if not users_who_rated.empty:
                        relevant_weights = user_weights[users_who_rated.index]

                        if relevant_weights.sum() > 0:
                            weighted_rating = (users_who_rated * relevant_weights).sum() / relevant_weights.sum()
                            score = weighted_rating
                            source = "Personalized Match 🎯"

                scored_recommendations.append(
                    {
                        'Name': attr_name,
                        'City': city_name,
                        'Score': score,
                        'Type': source,
                    }
                )

            scored_recommendations.sort(key=lambda x: x['Score'], reverse=True)

            for rec in scored_recommendations[:10]:
                col_r1, col_r2 = st.columns([3, 1])
                with col_r1:
                    st.markdown(f"### {rec['Name']}")
                    st.caption(f"📍 {rec['City']}")
                with col_r2:
                    if rec['Score'] > 0:
                        st.metric("Match Score", f"{rec['Score']:.2f}")
                    else:
                        st.badge(rec['Type'])
                st.divider()

# --- TAB 3: ANALYTICS (EDA) ---
with tab3:
    st.header("Tourism Trends Visualization")
    # Only analyze real visit data (where rating > 0)
    real_data = df_trans[df_trans['Rating'] > 0]
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Top 10 Popular Attraction Types")
        st.bar_chart(real_data['AttractionType'].value_counts().head(10))
    with col_b:
        st.subheader("Average Rating by Continent")
        st.line_chart(real_data.groupby('Continent')['Rating'].mean())

    st.subheader("Master Data Preview")
    st.dataframe(df_items.head(10))
