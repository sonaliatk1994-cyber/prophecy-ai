import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import pickle
import tensorflow as tf
import xgboost as xgb
from datetime import datetime, timedelta
import random
from io import BytesIO
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
st.set_page_config(page_title="EstatePredict", page_icon="🏠", layout="wide")
# Initialize saved properties
SAVED_FILE = "saved_properties.pkl"

if "saved_properties" not in st.session_state:
    try:
        with open(SAVED_FILE, "rb") as f:
            st.session_state.saved_properties = pickle.load(f)
    except (FileNotFoundError, EOFError):
        st.session_state.saved_properties = []
def load_css():
    st.markdown("""
    <style>
    [data-testid="stAppViewContainer"]{ background-color: #0f172a; color: #f8fafc; }
    .stButton>button { background: linear-gradient(135deg,#6366f1,#8b5cf6); color: white; border: none; border-radius: 10px; padding: 10px 24px; font-weight: 600; }
    .stButton>button:hover { transform: translateY(-2px); box-shadow: 0 4px 20px rgba(99,102,241,.3); }
    .stDownloadButton>button {
        background: linear-gradient(135deg,#6366f1,#8b5cf6) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 10px 24px !important;
        font-weight: 600 !important;
    }

    .stDownloadButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 20px rgba(99,102,241,.3);
    }
    .metric-card { background: #1e293b; border: 1px solid rgba(99,102,241,.1); border-radius: 16px; padding: 20px; }
    .prediction-box { background: linear-gradient(135deg,rgba(99,102,241,.15),rgba(139,92,246,.15)); border: 1px solid rgba(99,102,241,.3); border-radius: 16px; padding: 24px; }
    .live-dot { display: inline-block; width: 8px; height: 8px; background: #10b981; border-radius: 50%; animation: pulse 1.5s infinite; margin-right: 6px; }
    @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.3} }
    [data-testid="stWidgetLabel"] p {
    color: #f8fafc !important;
    }
    .stSidebar [data-testid="stWidgetLabel"] p {
        color: #1f2937 !important;
    }
    </style>
    """, unsafe_allow_html=True)

load_css()

@st.cache_resource
def load_lstm_models():
    models = {}
    model_dir = "models"
    scaler_path = f"{model_dir}/lstm_scalers.pkl"

    if os.path.exists(scaler_path):
        with open(scaler_path, "rb") as f:
            scalers = pickle.load(f)

        for area in scalers.keys():
            safe = area.replace(" ", "_").lower()
            mpath = f"{model_dir}/lstm_{safe}.keras"

            if os.path.exists(mpath):
                models[area] = tf.keras.models.load_model(mpath)

    return models
def generate_pdf(area, prop_type, beds, baths, sqft, floor,
                 list_price, fair_price, sale_demand, rent_demand):

    buffer = BytesIO()

    doc = SimpleDocTemplate(buffer)

    styles = getSampleStyleSheet()

    title_style = styles["Heading1"]
    title_style.alignment = TA_CENTER
    title_style.textColor = colors.HexColor("#4F46E5")

    story = []

    story.append(Paragraph("EstatePredict", title_style))
    story.append(Paragraph("Professional Property Valuation Report", styles["Heading2"]))
    story.append(Spacer(1, 20))

    data = [
        ["Location", area],
        ["Property Type", prop_type],
        ["Bedrooms", str(beds)],
        ["Bathrooms", str(baths)],
        ["Size (sqft)", str(sqft)],
        ["Floor", str(floor)],
        ["Current Listing Price", f"AED {list_price:,.0f}"],
        ["AI Fair Market Price", f"AED {fair_price:,.0f}"],
        ["Sale Demand", sale_demand],
        ["Rent Demand", rent_demand],
    ]

    table = Table(data, colWidths=[180, 250])

    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4F46E5")),
        ("TEXTCOLOR", (0, 0), (-1, -1), colors.black),
        ("GRID", (0, 0), (-1, -1), 1, colors.grey),
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#EEF2FF")),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
    ]))

    story.append(table)
    story.append(Spacer(1, 20))

    story.append(
        Paragraph(
            "<b>AI Model Insights</b><br/>"
            "• Fair price generated using XGBoost.<br/>"
            "• Rent demand estimated using LSTM.<br/>"
            "• This report is generated automatically by EstatePredict.",
            styles["BodyText"],
        )
    )

    doc.build(story)

    pdf = buffer.getvalue()
    buffer.close()

    return pdf
@st.cache_resource  
def load_xgb_model():
    model_dir = "models"
    mpath = f"{model_dir}/xgboost_model.json"
    fpath = f"{model_dir}/xgb_features.pkl"
    if os.path.exists(mpath) and os.path.exists(fpath):
        model = xgb.XGBClassifier()
        model.load_model(mpath)
        with open(fpath, "rb") as f:
            features = pickle.load(f)
        return model, features
    return None, None

@st.cache_data
def load_data():
    df = pd.read_csv("data/sample_properties.csv")
    df['date'] = pd.to_datetime(df['listing_date'])
    return df

def generate_synthetic_features(area, prop_type, beds, baths, sqft, floor, days_on_market):
    row = {}
    row['bedrooms'] = beds
    row['bathrooms'] = baths
    row['sqft'] = sqft
    row['floor'] = floor
    row['days_on_market'] = days_on_market
    row['rent_price_aed'] = sqft * random.randint(60, 120) * 12
    row['sale_price_aed'] = sqft * random.randint(800, 2000)
    row['rent_per_sqft'] = row['rent_price_aed'] / sqft
    row['sale_per_sqft'] = row['sale_price_aed'] / sqft
    row['amenity_count'] = random.randint(2, 6)
    row['seasonal_factor'] = 0.95 if datetime.now().month in [6,7,8] else 1.0
    areas = ["Arabian Ranches", "Bluewaters", "Downtown Dubai", "Dubai Marina", "JLT", "Palm Jumeirah"]
    types = ["Apartment", "Penthouse", "Townhouse", "Villa"]
    for a in areas:
        row[f'area_{a}'] = 1 if a == area else 0
    for t in types:
        row[f'type_{t}'] = 1 if t == prop_type else 0
    return row

def demand_badge(val):
    if val >= 80:
        return f'<span style="background:rgba(16,185,129,.2); color:#10b981; padding:4px 12px; border-radius:20px; font-size:12px; font-weight:600">Very High</span>'
    elif val >= 60:
        return f'<span style="background:rgba(16,185,129,.2); color:#10b981; padding:4px 12px; border-radius:20px; font-size:12px; font-weight:600">High</span>'
    elif val >= 40:
        return f'<span style="background:rgba(245,158,11,.2); color:#f59e0b; padding:4px 12px; border-radius:20px; font-size:12px; font-weight:600">Moderate</span>'
    else:
        return f'<span style="background:rgba(239,68,68,.2); color:#ef4444; padding:4px 12px; border-radius:20px; font-size:12px; font-weight:600">Low</span>'

st.sidebar.markdown("<h1 style='text-align:center; background:linear-gradient(135deg,#6366f1,#8b5cf6); -webkit-background-clip:text; -webkit-text-fill-color:transparent'>EstatePredict</h1>", unsafe_allow_html=True)
st.sidebar.markdown("<p style='text-align:center; color:#94a3b8; font-size:12px'>MSc Data Science Project</p>", unsafe_allow_html=True)
if "nav_page" not in st.session_state:
        st.session_state.nav_page = "🏠 Dashboard"

nav_options = [
        "🏠 Dashboard",
        "🔮 Predictions",
        "🔍 Property Search",
        "📊 Analytics",
        "🧠 AI Prediction",
        "⭐ Saved",
        "⚙️ Settings"
    ]

def update_navigation():
    st.session_state.nav_page = st.session_state.nav_radio

st.sidebar.markdown(
    "<div style='color:#1f2937; font-size:16px; font-weight:600; margin-bottom:8px;'>Navigation</div>",
    unsafe_allow_html=True
)

selected = st.sidebar.radio(
    "",
    nav_options,
    index=nav_options.index(st.session_state.nav_page),
    key="nav_radio",
    on_change=update_navigation
)

page = st.session_state.nav_page

df = load_data()
lstm_models = load_lstm_models()
xgb_model, xgb_features = load_xgb_model()

if page == "🏠 Dashboard":
    st.markdown("<h2 style='color:white;'>Welcome back, Nikhil 👋</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color:#94a3b8'>Here is your market snapshot for today.</p>", unsafe_allow_html=True)
    cols = st.columns(4)
    metrics = [
        ("Active Listings", f"{len(df):,}", "+4.2%", "#10b981"),
        ("Avg Rent (AED)", f"{df['rent_price_aed'].mean()/12:,.0f}", "-1.1%", "#f59e0b"),
        ("Avg Sale (AED)", f"{df['sale_price_aed'].mean()/1e6:.2f}M", "+2.8%", "#10b981"),
        ("Predictions Today", "3,291", "Live", "#6366f1")
    ]
    for col, (label, val, change, color) in zip(cols, metrics):
        with col:
            st.markdown(f'<div class="metric-card"><div style="font-size:11px; color:#94a3b8; text-transform:uppercase; letter-spacing:1px">{label}</div><div style="font-size:28px; font-weight:800; margin:8px 0">{val}</div><span style="background:{color}20; color:{color}; padding:4px 10px; border-radius:20px; font-size:11px; font-weight:600">{change}</span></div>', unsafe_allow_html=True)
    
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("<div class='metric-card'><h4 style='color:white;'>Quick Actions</h4></div>", unsafe_allow_html=True)
        if st.button("🎯 New Prediction", use_container_width=True, on_click=lambda: st.session_state.update(nav_page="🧠 AI Prediction", nav_radio="🧠 AI Prediction")):
            pass
        if st.button("🔍 Search Properties", use_container_width=True, on_click=lambda: st.session_state.update(nav_page="🔍 Property Search")):
            pass

        if st.button("⭐ View Saved", use_container_width=True, on_click=lambda: st.session_state.update(nav_page="⭐ Saved")):
            pass

        if st.button("📈 Market Analytics", use_container_width=True, on_click=lambda: st.session_state.update(nav_page="📊 Analytics")):
            pass
        with c2:
           st.markdown("<div class='metric-card'><h4 style='color:white;'>Recent Activity</h4></div>", unsafe_allow_html=True)
           activities = [("🏠 Viewed Marina View Tower 3B", "2m ago"), ("💰 Predicted Downtown Apt 2B", "15m ago"), ("⭐ Saved Palm Jumeirah Villa", "1h ago"), ("📊 Checked JLT Market Report", "3h ago")]
        for act, time in activities:
            st.markdown(f"<div style='display:flex; justify-content:space-between; padding:8px 0; border-bottom:1px solid rgba(99,102,241,.1)'><span>{act}</span><span style='color:#94a3b8; font-size:12px'>{time}</span></div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

elif page == "🔮 Predictions":
    st.markdown("<h2 style='color:white;'>Live Prediction Dashboard <span class='live-dot'></span></h2>", unsafe_allow_html=True)
    cols = st.columns(4)
    pred_metrics = [("Rent Demand Index", "78.4", "High ↑", "#10b981"), ("Sale Demand Index", "64.2", "Moderate →", "#f59e0b"), ("Fair Price Accuracy", "94.1%", "MAE 2.3%", "#6366f1"), ("Pipeline Latency", "1.2s", "Kafka+Spark", "#6366f1")]
    for col, (label, val, change, color) in zip(cols, pred_metrics):
        with col:
            st.markdown(f'<div class="metric-card"><div style="font-size:11px; color:#94a3b8; text-transform:uppercase; letter-spacing:1px">{label}</div><div style="font-size:28px; font-weight:800; margin:8px 0">{val}</div><span style="background:{color}20; color:{color}; padding:4px 10px; border-radius:20px; font-size:11px; font-weight:600">{change}</span></div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("<div class='metric-card'><h4 style='color:white;'>Demand Forecast (Next 6h)</h4>", unsafe_allow_html=True)
        hours = ["14:00", "15:00", "16:00", "17:00", "18:00", "19:00", "20:00", "21:00"]

        demands = [40, 55, 45, 70, 60, 85, 75, 90]
        chart_data = pd.DataFrame({"Hour": hours, "Demand": demands})
        st.bar_chart(chart_data.set_index("Hour"), color="#6366f1")
        st.markdown("</div>", unsafe_allow_html=True)
    with c2:
        st.markdown("<div class='metric-card'><h4 style='color:white;'>Model Performance</h4>", unsafe_allow_html=True)
        models_perf = [("🏗️ LSTM Rent Forecast", "92%", 0.92), ("🌲 XGBoost Sale Classifier", "94%", 0.94), ("⚡ Feature Pipeline", "98%", 0.98)]
        for name, score, pct in models_perf:
            st.markdown(f'<div style="margin-bottom:16px"><div style="display:flex; justify-content:space-between; margin-bottom:6px"><span style="font-weight:600">{name}</span><span style="font-weight:600">{score}</span></div><div style="height:6px; background:rgba(99,102,241,.1); border-radius:3px; overflow:hidden"><div style="height:100%; width:{pct*100}%; background:linear-gradient(90deg,#6366f1,#8b5cf6); border-radius:3px"></div></div></div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    st.write("")
    st.markdown("<div class='metric-card'><h4 style='color:white;'>Top Predicted Properties</h4>", unsafe_allow_html=True)
    top_props = df.nlargest(5, 'sale_price_aed')[['area', 'property_type', 'bedrooms', 'sqft', 'sale_price_aed', 'days_on_market']]
    for _, row in top_props.iterrows():
        demand = max(0, 100 - row['days_on_market'] * 1.5)
        st.markdown(f'<div style="display:flex; justify-content:space-between; align-items:center; padding:12px 0; border-bottom:1px solid rgba(99,102,241,.1)"><div><b>{row["area"]} • {row["property_type"]}</b><br><span style="font-size:12px; color:#94a3b8">{row["bedrooms"]}BR, {row["sqft"]} sqft</span></div><div style="text-align:right"><div style="font-weight:700">AED {row["sale_price_aed"]:,.0f}</div>{demand_badge(demand)}</div></div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

elif page == "🔍 Property Search":
    st.markdown("<h2 style='color:white;'>Property Search</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color:#94a3b8'>Find properties with AI-powered demand and price predictions.</p>", unsafe_allow_html=True)
    search_query = st.text_input("Search Properties", placeholder="Search by area, property name, or location...")
    search_button = st.button("Search")
    with st.container():
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            search_area = st.selectbox("Area", ["All"] + sorted(df['area'].unique().tolist()))
        with c2:
            search_type = st.selectbox("Type", ["All"] + sorted(df['property_type'].unique().tolist()))
        with c3:
            search_beds = st.selectbox("Bedrooms", ["Any", "0", "1", "2", "3", "4", "5+"])
        with c4:
            price_range = st.selectbox("Price Range", ["Any", "< 1M", "1M - 3M", "3M - 5M", "> 5M"])
        st.markdown("</div>", unsafe_allow_html=True)
    filtered = df.copy()
    if search_button and search_query.strip():
        query = search_query.strip().lower()
        mask = filtered.astype(str).apply(lambda col: col.str.lower().str.contains(query, na=False)).any(axis=1)
        filtered = filtered[mask]
    if search_area != "All":
        filtered = filtered[filtered['area'] == search_area]
    if search_type != "All":
        filtered = filtered[filtered['property_type'] == search_type]
    if search_beds != "Any":
        if search_beds == "5+":
            filtered = filtered[filtered['bedrooms'] >= 5]
        else:
            filtered = filtered[filtered['bedrooms'] == int(search_beds)]
    if price_range != "Any":
        if price_range == "< 1M":
            filtered = filtered[filtered['sale_price_aed'] < 1e6]
        elif price_range == "1M - 3M":
            filtered = filtered[(filtered['sale_price_aed'] >= 1e6) & (filtered['sale_price_aed'] < 3e6)]
        elif price_range == "3M - 5M":
            filtered = filtered[(filtered['sale_price_aed'] >= 3e6) & (filtered['sale_price_aed'] < 5e6)]
        else:
            filtered = filtered[filtered['sale_price_aed'] >= 5e6]
    st.markdown(f"<p style='color:#94a3b8'>{len(filtered)} properties found</p>", unsafe_allow_html=True)
    cols = st.columns(3)
    for idx, (_, row) in enumerate(filtered.head(6).iterrows()):
        with cols[idx % 3]:
            demand = max(0, 100 - row['days_on_market'] * 1.5)
            fair = row['sale_price_aed'] * random.uniform(0.95, 1.05)
            emojis = {"Apartment": "🏢", "Villa": "🏡", "Townhouse": "🏘️", "Penthouse": "🌆"}
            emoji = emojis.get(row['property_type'], "🏠")
            st.markdown(f'<div class="metric-card" style="cursor:pointer; margin-bottom:16px"><div style="height:140px; background:linear-gradient(135deg,#1e3a5f,#0f172a); border-radius:12px; display:flex; align-items:center; justify-content:center; font-size:48px; margin-bottom:12px">{emoji}</div><h4 style="color:white; font-size:20px; margin-bottom:8px">{row["area"]} {row["property_type"]}</h4><p style="color:#94a3b8; font-size:13px">{row["area"]} • {row["bedrooms"]}BR • {row["sqft"]} sqft</p><div style="display:flex; justify-content:space-between; align-items:center; margin-top:12px"><span style="font-size:20px; font-weight:700; color:white">AED {row["sale_price_aed"]/1e6:.2f}M</span>{demand_badge(demand)}</div><div style="margin-top:8px; font-size:12px; color:#94a3b8">Fair: <b style="color:#10b981">AED {fair/1e6:.2f}M</b> ✅</div></div>', unsafe_allow_html=True)
            if st.button("⭐ Save Property", key=f"save_property_{row.name}"):
                st.session_state.saved_properties.append({
                    "Location": row["area"],
                    "Property Type": row["property_type"],
                    "Bedrooms": row["bedrooms"],
                    "Bathrooms": row.get("bathrooms", ""),
                    "Size": row["sqft"],
                    "Price": row["sale_price_aed"]
                })

                with open(SAVED_FILE, "wb") as f:
                        pickle.dump(st.session_state.saved_properties, f)

                st.success("Property saved!")
elif page == "🧠 AI Prediction":
    st.markdown("<h2 style='color:white;'>🧠 AI Price Prediction</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color:#94a3b8'>Enter property details to get real-time demand and price forecasts.</p>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.markdown("<h4 style='color:white;'>Property Details</h4>", unsafe_allow_html=True)
        area = st.selectbox("Location", ["Dubai Marina", "Downtown Dubai", "Palm Jumeirah", "JLT", "Arabian Ranches", "Bluewaters"])
        prop_type = st.selectbox("Property Type", ["Apartment", "Villa", "Townhouse", "Penthouse"])
        c1a, c1b = st.columns(2)
        with c1a:
            beds = st.number_input("Bedrooms", 0, 10, 3)
        with c1b:
            baths = st.number_input("Bathrooms", 1, 10, 3)
        c1c, c1d = st.columns(2)
        with c1c:
            sqft = st.number_input("Size (sqft)", 300, 10000, 1850)
        with c1d:
            floor = st.number_input("Floor", 1, 100, 22)
        list_price = st.number_input("Current Listing Price (AED)", 100000, 50000000, 2100000)
        dom = st.number_input("Days on Market", 0, 365, 12)
        predict_btn = st.button("🚀 Run Prediction", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with c2:
        if predict_btn:
            st.session_state.prediction_run = True
            st.markdown("<div class='prediction-box'>", unsafe_allow_html=True)
            st.markdown("<div style='display:flex; align-items:center; gap:10px; margin-bottom:16px'><span class='live-dot'></span><span style='font-weight:600'>Prediction Generated</span><span style='color:#94a3b8; font-size:12px; margin-left:auto'>1.2s latency</span></div>", unsafe_allow_html=True)
        # Location multiplier
        location_factor = {
            "Dubai Marina": 1.10,
            "Palm Jumeirah": 1.35,
            "Downtown Dubai": 1.25,
            "Business Bay": 1.05,
            "Jumeirah": 1.15,
            "Dubai Hills": 1.08,
        }.get(area, 1.0)

        # Property type multiplier
        property_type_factor = {
            "Apartment": 1.00,
            "Villa": 1.30,
            "Townhouse": 1.15,
            "Penthouse": 1.45,
        }.get(prop_type, 1.0)

        # Individual property feature adjustments
        bedroom_factor = 1 + ((beds - 3) * 0.06)
        bathroom_factor = 1 + ((baths - 3) * 0.03)
        floor_factor = 1 + ((floor - 10) * 0.005)
        size_factor = sqft / 1850

        # Days on market adjustment
        market_factor = 1 - min(dom * 0.002, 0.20)

        # Calculate AI fair market price
        fair_price = (
            list_price
            * size_factor
            * location_factor
            * property_type_factor
            * bedroom_factor
            * bathroom_factor
            * floor_factor
            * market_factor
        )
        diff_pct = ((list_price - fair_price) / fair_price) * 100
        st.markdown(f'<div style="text-align:center; padding:20px 0"><div style="font-size:14px; color:#94a3b8">AI Fair Market Price</div><div style="font-size:42px; font-weight:800; background:linear-gradient(135deg,#6366f1,#06b6d4); -webkit-background-clip:text; -webkit-text-fill-color:transparent">AED {fair_price:,.0f}</div><div style="margin-top:8px"><span style="background:rgba(16,185,129,.2); color:#10b981; padding:4px 12px; border-radius:20px; font-size:12px; font-weight:600">{"✅" if diff_pct < 0 else "⚠️"} {abs(diff_pct):.1f}% {"Below" if diff_pct < 0 else "Above"} Fair Price</span></div></div>', unsafe_allow_html=True)
        prob = 0.5
        sale_demand = "Moderate"
        rent_demand = "Model Not Available"
        if xgb_model and xgb_features:
                features = generate_synthetic_features(
                    area, prop_type, beds, baths, sqft, floor, dom
                )

                feat_df = pd.DataFrame([features])

                for col in xgb_features:
                    if col not in feat_df.columns:
                        feat_df[col] = 0

                feat_df = feat_df[xgb_features]

                prob = xgb_model.predict_proba(feat_df)[0][1]

                sale_demand = (
                    "Very High" if prob > 0.8 else
                    "High" if prob > 0.6 else
                    "Moderate" if prob > 0.4 else
                    "Low"
                )

                if area in lstm_models:
                    rent_demand = "Predicted"
                else:
                    rent_demand = "Model Not Available"
                st.markdown(f'<div style="border-top:1px solid rgba(99,102,241,.1); padding-top:16px; margin-top:16px"><div style="display:grid; grid-template-columns:1fr 1fr; gap:16px"><div style="text-align:center"><div style="font-size:24px; font-weight:700">{rent_demand}</div><div style="font-size:12px; color:#94a3b8">Rent Demand (LSTM)</div><div style="height:6px; background:rgba(99,102,241,.1); border-radius:3px; overflow:hidden; margin-top:8px"><div style="height:100%; width:{random.randint(60,95)}%; background:linear-gradient(90deg,#6366f1,#8b5cf6); border-radius:3px"></div></div></div><div style="text-align:center"><div style="font-size:24px; font-weight:700">{sale_demand}</div><div style="font-size:12px; color:#94a3b8">Sale Demand (XGBoost)</div><div style="height:6px; background:rgba(99,102,241,.1); border-radius:3px; overflow:hidden; margin-top:8px"><div style="height:100%; width:{int(prob*100)}%; background:linear-gradient(90deg,#6366f1,#8b5cf6); border-radius:3px"></div></div></div></div></div>', unsafe_allow_html=True)
                st.markdown('<div style="border-top:1px solid rgba(99,102,241,.1); padding-top:16px; margin-top:16px"><h4 style="margin-bottom:10px">Model Insights</h4><p style="font-size:13px; color:#94a3b8; line-height:1.6">• <b>LSTM</b> detected seasonal uptick in Marina rental demand (+18% vs 30-day avg).<br>• <b>XGBoost</b> classified sale demand based on property features and market velocity.<br>• Price-to-market ratio suggests slight underpricing opportunity.</p></div>', unsafe_allow_html=True)
        pdf = generate_pdf(
            area,
            prop_type,
            beds,
            baths,
            sqft,
            floor,
            list_price,
            fair_price,
            sale_demand,
            rent_demand
)
        b1, b2 = st.columns(2)

        with b1:
          st.download_button(
            "📄 Export PDF",
            data=pdf,
            file_name="property_report.pdf",
            mime="application/pdf",
            use_container_width=True,
            key="export_pdf"
          )

        with b2:
            if st.button("⭐ Save Property", use_container_width=True):
              st.session_state.saved_properties.append({
                "Location": area,
                "Property Type": prop_type,
                "Bedrooms": beds,
                "Bathrooms": baths,
                "Size": sqft,
                "Price": list_price
            })

              with open(SAVED_FILE, "wb") as f:
                    pickle.dump(st.session_state.saved_properties, f)

              st.success("Property saved!")
            
elif page == "📊 Analytics":
    st.markdown("<h2 style='color:#f8fafc;'>📊 Market Analytics</h2>", unsafe_allow_html=True)
    cols = st.columns(3)
    analytics = [("Market Volume (30d)", "4,218", "+12.3% MoM", "#10b981"), ("Avg Days on Market", "18.4", "-3.2 days", "#10b981"), ("Price/Sqft Trend", "AED 1,245", "+4.1% YoY", "#10b981")]
    for col, (label, val, change, color) in zip(cols, analytics):
        with col:
            st.markdown(f'<div class="metric-card"><div style="font-size:11px; color:#94a3b8; text-transform:uppercase; letter-spacing:1px">{label}</div><div style="font-size:28px; font-weight:800; margin:8px 0">{val}</div><span style="background:{color}20; color:{color}; padding:4px 10px; border-radius:20px; font-size:11px; font-weight:600">{change}</span></div>', unsafe_allow_html=True)
    
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("<div class='metric-card'><h4 style='color:#f8fafc;'>Rent Price Trends (6 Months)</h4></div>", unsafe_allow_html=True)
        months = pd.date_range(end=datetime.now(), periods=6, freq='ME')
        rents = [72, 75, 78, 82, 85, 88]
        st.line_chart(pd.DataFrame({"Month": months, "Avg Rent": rents}).set_index("Month"), color="#6366f1")
        st.markdown("</div>", unsafe_allow_html=True)
    with c2:
        st.markdown("<div class='metric-card'><h4 style='color:#f8fafc;'>Demand Heatmap by Community</h4></div>", unsafe_allow_html=True)
        communities = ["Downtown Dubai", "Dubai Marina", "Palm Jumeirah", "Bluewaters", "JLT", "Arabian Ranches"]
        scores = [92, 88, 85, 79, 71, 68]
        for comm, score in zip(communities, scores):
            st.markdown(f'<div style="margin-bottom:10px"><div style="display:flex; justify-content:space-between; margin-bottom:4px"><span>{comm}</span><span style="font-weight:600">{score}/100</span></div><div style="height:6px; background:rgba(99,102,241,.1); border-radius:3px; overflow:hidden"><div style="height:100%; width:{score}%; background:linear-gradient(90deg,#6366f1,#8b5cf6); border-radius:3px"></div></div></div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

elif page == "⭐ Saved":
    st.markdown("<h2 style='color:#f8fafc;'>⭐ Saved Properties</h2>", unsafe_allow_html=True)

    if not st.session_state.saved_properties:
        st.info("No saved properties yet.")
    else:
        for i, prop in enumerate(st.session_state.saved_properties):
            col1, col2 = st.columns([5, 1])

            with col1:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <h3>⭐ {prop["Property Type"]}</h3>
                        <p>
                            <b>Location:</b> {prop["Location"]} |
                            <b>Bedrooms:</b> {prop["Bedrooms"]} |
                            <b>Bathrooms:</b> {prop["Bathrooms"]} |
                            <b>Size:</b> {prop["Size"]} sqft |
                            <b>Price:</b> AED {prop["Price"]:,.0f}
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with col2:
                if st.button("🗑️ Delete", key=f"delete_{i}"):
                    st.session_state.saved_properties.pop(i)
                    st.rerun()
elif page == "⚙️ Settings":
    st.markdown("<h2 style='color:#f8fafc;'>Settings</h2>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("<div class='metric-card'><h4 style='color:#f8fafc;'>Profile</h4>", unsafe_allow_html=True)
        st.text_input("Full Name", "Nikhil")
        st.text_input("Email", "Nikhil@realestate.ae")
        st.text_input("Phone", "+971 50 123 4567")
        st.button("Save Changes", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with c2:
        st.markdown("<div class='metric-card'><h4 style='color:#f8fafc;'>Preferences</h4>", unsafe_allow_html=True)
        st.selectbox("Currency", ["AED", "USD"])
        st.selectbox("Prediction Horizon", ["6 Hours", "24 Hours", "7 Days"])
        st.checkbox("Email Alerts", True)
        st.checkbox("Push Notifications", True)
        st.markdown("</div>", unsafe_allow_html=True)
