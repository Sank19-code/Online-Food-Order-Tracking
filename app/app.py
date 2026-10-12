"""
TrackBite - Online Food Order Tracking (23CSE301 ML Capstone GUI)

Run from the repository root:
    streamlit run app/app.py

Two trained models power the screen:
  * ETA        - tuned Gradient Boosting Regressor  (minutes from order to doorstep)
  * Delay risk - tuned Gradient Boosting Classifier (P(delivery takes > 25 min))
Both are loaded from models/*.pkl. If the files are missing or were saved with a
different scikit-learn version, they are re-trained once from data/train.csv.
"""
import datetime as dt
import math
import os
import sys

import altair as alt
import joblib
import numpy as np
import pandas as pd
import pydeck as pdk
import sklearn
import streamlit as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))
from pipeline import DELAY_THRESHOLD_MIN, clean_and_engineer, load_raw_data, make_order_frame  # noqa: E402
from train_models import CLF_PATH, REG_PATH, train_all  # noqa: E402

st.set_page_config(page_title='TrackBite · Food Order Tracking', page_icon='🛵', layout='wide')

# --------------------------------------------------------------------------------------
# Look & feel: food-delivery app styling (warm saffron, white cards, rounded corners)
# --------------------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap');
html, body, [class*="css"], .stMarkdown, .stText, button, input, label { font-family: 'Poppins', sans-serif; }
#MainMenu, footer, [data-testid="stDecoration"] { visibility: hidden; }
[data-testid="stAppDeployButton"], [data-testid="stToolbar"] { display:none; }
[data-testid="stHeader"] { background:transparent; height:0; }
.block-container { padding-top: 1.6rem; padding-bottom: 2rem; max-width: 1280px; }
:root { --saffron:#E8630A; --saffron-soft:#FFEBD8; --brown:#2B1A10; --muted:#7A6555;
        --green:#16834A; --green-soft:#E3F5EA; --red:#C62828; --red-soft:#FDE7E5; --card:#FFFFFF; --line:#F1DCC8; }
.tb-header { display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:12px;
             background:linear-gradient(100deg,#E8630A 0%,#F58A2C 60%,#F7A64B 100%); color:#fff;
             padding:16px 22px; border-radius:20px; margin-bottom:18px; box-shadow:0 8px 24px rgba(232,99,10,.25); }
.tb-logo { font-size:1.65rem; font-weight:700; letter-spacing:-.5px; }
.tb-logo small { display:block; font-size:.78rem; font-weight:400; opacity:.92; letter-spacing:0; }
.tb-loc { background:rgba(255,255,255,.18); border:1px solid rgba(255,255,255,.35); border-radius:999px;
          padding:7px 16px; font-size:.88rem; }
.tb-card { background:var(--card); border:1px solid var(--line); border-radius:18px; padding:18px 20px;
           box-shadow:0 2px 10px rgba(43,26,16,.05); margin-bottom:14px; }
.tb-h { font-weight:600; font-size:1.02rem; margin:0 0 8px 0; color:var(--brown); }
.tb-muted { color:var(--muted); font-size:.85rem; }
.tb-eta { display:flex; align-items:flex-end; gap:14px; flex-wrap:wrap; }
.tb-eta .big { font-size:3.1rem; line-height:1; font-weight:700; color:var(--brown); }
.tb-eta .unit { font-size:1.1rem; font-weight:600; color:var(--muted); margin-bottom:6px; }
.tb-chip { display:inline-block; border-radius:999px; padding:5px 13px; font-size:.82rem; font-weight:600; }
.tb-chip.ok { background:var(--green-soft); color:var(--green); }
.tb-chip.bad { background:var(--red-soft); color:var(--red); }
.tb-chip.warn { background:#FFF1CC; color:#8A5A00; }
.tb-chip.info { background:var(--saffron-soft); color:#9A3F00; }
.tb-bar { height:8px; background:#F3E6DA; border-radius:999px; overflow:hidden; margin:8px 0 4px; }
.tb-bar > div { height:100%; border-radius:999px; }
.tb-steps { display:flex; justify-content:space-between; position:relative; margin:18px 4px 4px; }
.tb-steps:before { content:''; position:absolute; top:15px; left:6%; right:6%; height:3px; background:#F1DCC8; }
.tb-step { position:relative; text-align:center; width:20%; font-size:.74rem; color:var(--muted); }
.tb-step .dot { width:32px; height:32px; margin:0 auto 6px; border-radius:50%; background:#fff; border:3px solid #F1DCC8;
                display:flex; align-items:center; justify-content:center; font-size:.95rem; position:relative; z-index:1; }
.tb-step.done .dot { background:var(--saffron); border-color:var(--saffron); color:#fff; }
.tb-step.now .dot { background:#fff; border-color:var(--saffron); box-shadow:0 0 0 5px rgba(232,99,10,.18); }
.tb-step.done, .tb-step.now { color:var(--brown); font-weight:600; }
.tb-step .t { display:block; font-weight:400; color:var(--muted); }
.tb-rider { display:flex; gap:14px; align-items:center; }
.tb-avatar { width:52px; height:52px; border-radius:50%; background:var(--saffron-soft); display:flex;
             align-items:center; justify-content:center; font-size:1.6rem; }
.tb-factor { display:flex; justify-content:space-between; align-items:center; padding:8px 0; border-bottom:1px dashed var(--line); font-size:.9rem; }
.tb-factor:last-child { border-bottom:none; }
.tb-factor b { color:var(--red); white-space:nowrap; margin-left:10px; }
.tb-banner { border-radius:14px; padding:10px 14px; font-size:.88rem; margin-bottom:12px; }
.tb-banner.warn { background:#FFF4D6; color:#7A5300; border:1px solid #F5DC94; }
.tb-banner.okb { background:var(--green-soft); color:var(--green); border:1px solid #BFE5CD; }
.tb-metric { background:#fff; border:1px solid var(--line); border-radius:16px; padding:14px 16px; }
.tb-metric .v { font-size:1.6rem; font-weight:700; color:var(--brown); }
.tb-metric .l { font-size:.8rem; color:var(--muted); }
div[data-testid="stExpander"] { background:#fff; border-radius:16px; border:1px solid var(--line); }
.stTabs [data-baseweb="tab-list"] { gap:18px; }
.stTabs [data-baseweb="tab"] { font-weight:600; }
.stButtonGroup > div { flex-wrap: wrap !important; row-gap: 6px; }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------------------------------------------
# Reference data
# --------------------------------------------------------------------------------------
CITY_NAMES = {'BANG': 'Bengaluru', 'CHEN': 'Chennai', 'HYD': 'Hyderabad', 'MUM': 'Mumbai', 'PUNE': 'Pune',
              'COIMB': 'Coimbatore', 'MYS': 'Mysuru', 'INDO': 'Indore', 'JAP': 'Jaipur', 'SUR': 'Surat',
              'VAD': 'Vadodara', 'RANCHI': 'Ranchi', 'KOL': 'Kolkata', 'KOC': 'Kochi', 'GOA': 'Goa',
              'DEH': 'Dehradun', 'LUDH': 'Ludhiana', 'KNP': 'Kanpur', 'ALH': 'Prayagraj', 'AGR': 'Agra',
              'AURG': 'Aurangabad', 'BHP': 'Bhopal'}
KITCHENS = [("Amma's Kitchen", 'Home-style Thali'), ('Spice Route Café', 'North Indian'),
            ('Tiffin Junction', 'South Indian Tiffin'), ('Masala Box', 'Snacks · Chaat'),
            ('Podi & Pickle House', 'Andhra Meals'), ('Biryani Corner', 'Biryani'), ('Chai Point Stop', 'Beverages'),
            ('Dosa Darbar', 'South Indian'), ('Thali Express', 'Gujarati Thali'), ('Green Leaf Meals', 'Banana-leaf Meals'),
            ('Snack Shack', 'Street Food'), ('Royal Tandoor', 'Tandoor · Kebabs')]
WEATHER = {'Sunny': '☀️ Sunny', 'Cloudy': '☁️ Cloudy', 'Windy': '🌬️ Windy', 'Fog': '🌫️ Fog',
           'Stormy': '⛈️ Stormy', 'Sandstorms': '🏜️ Sandstorm'}
TRAFFIC = ['Low', 'Medium', 'High', 'Jam']
VEHICLES = {'motorcycle': '🏍️ Motorcycle', 'scooter': '🛵 Scooter', 'electric_scooter': '⚡ E-scooter',
            'bicycle': '🚲 Bicycle'}
CITY_TYPES = {'Metropolitian': 'Metropolitan', 'Urban': 'Urban', 'Semi-Urban': 'Semi-urban'}
ORDER_TYPES = {'Meal': '🍛 Meal', 'Snack': '🥟 Snack', 'Drinks': '🥤 Drinks', 'Buffet': '🍱 Buffet'}
VEHICLE_COND = {0: 'Needs service', 1: 'Fair', 2: 'Good', 3: 'Excellent'}


@st.cache_resource(show_spinner='Loading the trained models…')
def load_models():
    """Load the saved models; re-train once if missing or saved with another sklearn version."""
    try:
        reg, clf = joblib.load(REG_PATH), joblib.load(CLF_PATH)
        if reg.get('sklearn_version') == sklearn.__version__ == clf.get('sklearn_version'):
            return reg, clf
    except Exception:  # noqa: BLE001 - any load problem -> rebuild from data
        pass
    with st.spinner('First run: training the models from data/train.csv (about 1 minute)…'):
        return train_all(verbose=False)


@st.cache_data(show_spinner=False)
def load_orders():
    df = clean_and_engineer(load_raw_data(), verbose=False)
    df['city_code'] = df['Delivery_person_ID'].str.extract(r'^([A-Z]+)RES')[0]
    df = df[df['city_code'].isin(CITY_NAMES)]
    return df


@st.cache_data(show_spinner=False)
def restaurants_by_city(df):
    """Up to 12 busiest real restaurant locations per city, given friendly demo names."""
    valid = df[(df['Restaurant_latitude'].abs() > 1) & (df['Restaurant_longitude'].abs() > 1)]
    out = {}
    for code, g in valid.groupby('city_code'):
        locs = (g.assign(lat=g['Restaurant_latitude'].abs().round(6), lon=g['Restaurant_longitude'].abs().round(6))
                .groupby(['lat', 'lon']).size().sort_values(ascending=False).head(12).reset_index(name='n'))
        out[code] = [{'name': KITCHENS[i % len(KITCHENS)][0], 'cuisine': KITCHENS[i % len(KITCHENS)][1],
                      'lat': r.lat, 'lon': r.lon, 'orders': int(r.n)} for i, r in locs.iterrows()]
    return out


reg_bundle, clf_bundle = load_models()
orders = load_orders()
RESTAURANTS = restaurants_by_city(orders)
CITY_LIST = sorted(RESTAURANTS, key=lambda c: CITY_NAMES[c])


def predict(frame):
    eta = reg_bundle['model'].predict(reg_bundle['preprocessor'].transform(frame))
    risk = clf_bundle['model'].predict_proba(clf_bundle['preprocessor'].transform(frame))[:, 1]
    return eta, risk


# --------------------------------------------------------------------------------------
# Session state: default order + "load a real order" callback
# --------------------------------------------------------------------------------------
DEFAULTS = {'city': 'BANG', 'order_type': 'Meal', 'distance': 6.5, 'city_type': 'Metropolitian',
            'order_hour': 19, 'prep': 10, 'weekend': False, 'festival': False,
            'weather': 'Cloudy', 'traffic': 'Medium', 'age': 29, 'rating': 4.7,
            'vehicle': 'motorcycle', 'v_cond': 2, 'multi': 1, 'actual': None, 'sim_t': 0}
for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)


def load_real_order():
    """Fill the form with a random real order from the dataset (and remember its true time)."""
    r = orders.dropna(subset=['Delivery_person_Age', 'Delivery_person_Ratings', 'Weatherconditions',
                              'Road_traffic_density', 'multiple_deliveries', 'City']).sample(1).iloc[0]
    code = r['city_code']
    st.session_state.update({
        'city': code, 'distance': float(round(min(max(r['distance_km'], 1.0), 21.0), 1)),
        'city_type': r['City'], 'order_hour': int(r['order_hour']),
        'prep': int(r['prep_time_min']) if r['prep_time_min'] in (5, 10, 15) else 10,
        'weekend': bool(r['is_weekend']), 'festival': bool(r['Festival_binary']),
        'weather': r['Weatherconditions'], 'traffic': r['Road_traffic_density'],
        'age': int(min(max(r['Delivery_person_Age'], 15), 50)), 'rating': float(round(r['Delivery_person_Ratings'], 1)),
        'vehicle': r['Type_of_vehicle'], 'v_cond': int(r['Vehicle_condition']), 'multi': int(r['multiple_deliveries']),
        'order_type': r['Type_of_order'] if r['Type_of_order'] in ORDER_TYPES else 'Meal',
        'actual': {'time': float(r['Time_taken']), 'id': r['ID'], 'rider': r['Delivery_person_ID']}, 'sim_t': 0})
    rest = RESTAURANTS[code]
    d2 = [(x['lat'] - abs(r['Restaurant_latitude'])) ** 2 + (x['lon'] - abs(r['Restaurant_longitude'])) ** 2 for x in rest]
    st.session_state[f'rest_{code}'] = rest[int(np.argmin(d2))]['name']


def reset_order():
    for k, v in DEFAULTS.items():
        st.session_state[k] = v


def clear_actual():
    st.session_state['actual'] = None


# --------------------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------------------
city_now = st.session_state['city']
st.markdown(f"""
<div class="tb-header">
  <div class="tb-logo">🛵 TrackBite<small>Online Food Order Tracking · ML-powered delivery ETA</small></div>
  <div class="tb-loc">📍 Delivering in <b>{CITY_NAMES[city_now]}</b> · {CITY_TYPES.get(st.session_state['city_type'], '')}</div>
</div>""", unsafe_allow_html=True)

tab_track, tab_model, tab_about = st.tabs(['🛵  Track order', '📊  Model insights', 'ℹ️  About'])

# ======================================================================================
# TAB 1 - order form (left) + live tracking (right)
# ======================================================================================
with tab_track:
    left, right = st.columns([1, 1.08], gap='large')

    with left:
        b1, b2 = st.columns(2)
        b1.button('🎲 Load a real order from the dataset', on_click=load_real_order, width='stretch',
                  help='Fills every field with a random order from train.csv so you can compare prediction vs. reality.')
        b2.button('↺ Reset', on_click=reset_order, width='stretch')

        # --- Restaurant ------------------------------------------------------------------
        st.markdown('<p class="tb-h">🍽️ Restaurant</p>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        c1.selectbox('City', CITY_LIST, format_func=CITY_NAMES.get, key='city', on_change=clear_actual)
        city = st.session_state['city']
        rest_names = [r['name'] for r in RESTAURANTS[city]]
        rest_key = f'rest_{city}'
        if st.session_state.get(rest_key) not in rest_names:
            st.session_state[rest_key] = rest_names[0]
        c2.selectbox('Kitchen', rest_names, key=rest_key, on_change=clear_actual)
        rest = next(r for r in RESTAURANTS[city] if r['name'] == st.session_state[rest_key])
        st.caption(f"{rest['cuisine']} · real pickup point from the dataset ({rest['orders']} orders)")

        st.pills('What are you ordering?', list(ORDER_TYPES), format_func=ORDER_TYPES.get,
                             key='order_type', help='Shown for realism only: EDA found order type has no effect '
                                                    'on delivery time (≈26.2–26.4 min for every type), so it is not a model feature.')

        # --- Delivery & timing -----------------------------------------------------------
        st.markdown('<p class="tb-h">📦 Delivery & timing</p>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        c1.slider('Distance to your door (km)', 1.0, 21.0, step=0.1, key='distance', on_change=clear_actual)
        c2.selectbox('Area type', list(CITY_TYPES), format_func=CITY_TYPES.get, key='city_type', on_change=clear_actual)
        c1, c2 = st.columns(2)
        c1.select_slider('Order placed at', options=list(range(24)), key='order_hour', on_change=clear_actual,
                         format_func=lambda h: dt.time(h).strftime('%I %p').lstrip('0'))
        c2.select_slider('Kitchen prep time (min)', options=[5, 10, 15], key='prep', on_change=clear_actual)
        c1, c2 = st.columns(2)
        c1.toggle('Weekend order', key='weekend', on_change=clear_actual)
        c2.toggle('Festival day 🎉', key='festival', on_change=clear_actual)

        # --- Conditions ------------------------------------------------------------------
        st.markdown('<p class="tb-h">🌦️ Road & weather</p>', unsafe_allow_html=True)
        st.pills('Weather', list(WEATHER), format_func=WEATHER.get, key='weather', on_change=clear_actual)
        st.segmented_control('Traffic density', TRAFFIC, key='traffic', on_change=clear_actual)

        # --- Delivery partner ------------------------------------------------------------
        with st.expander('🧑 Delivery partner details', expanded=True):
            c1, c2 = st.columns(2)
            c1.slider('Partner age', 15, 50, key='age', on_change=clear_actual)
            c2.slider('Partner rating ★', 1.0, 5.0, step=0.1, key='rating', on_change=clear_actual)
            c1, c2 = st.columns(2)
            c1.selectbox('Vehicle', list(VEHICLES), format_func=VEHICLES.get, key='vehicle', on_change=clear_actual)
            c2.select_slider('Vehicle condition', options=[0, 1, 2, 3], format_func=VEHICLE_COND.get,
                             key='v_cond', on_change=clear_actual)
            st.select_slider('Other orders batched on this trip', options=[0, 1, 2, 3], key='multi',
                             on_change=clear_actual)

    # Segmented controls can be de-selected (value None): fall back to defaults
    s = {k: (st.session_state.get(k) if st.session_state.get(k) is not None else v) for k, v in DEFAULTS.items()}

    inputs = dict(rider_age=s['age'], rider_rating=s['rating'], distance_km=s['distance'], prep_time_min=s['prep'],
                  order_hour=s['order_hour'], vehicle_condition=s['v_cond'], multiple_deliveries=s['multi'],
                  festival=s['festival'], is_weekend=s['weekend'], weather=s['weather'], traffic=s['traffic'],
                  city_type=s['city_type'], vehicle_type=s['vehicle'])
    eta_arr, risk_arr = predict(make_order_frame(**inputs))
    eta, risk = float(eta_arr[0]), float(risk_arr[0])
    eta_min = int(round(eta))
    margin = max(2, int(round(reg_bundle['metrics']['MAE'])))
    delayed = risk >= 0.5

    # What-if analysis: how many minutes does each condition add vs. a "smooth" alternative?
    scenarios = []
    if s['traffic'] != 'Low':
        scenarios.append((f"{s['traffic']} traffic on the route", {'traffic': 'Low'}))
    if s['weather'] != 'Sunny':
        scenarios.append((f"{WEATHER[s['weather']]} weather", {'weather': 'Sunny'}))
    if s['multi'] > 0:
        scenarios.append((f"Partner is carrying {s['multi']} other order{'s' if s['multi'] > 1 else ''}", {'multiple_deliveries': 0}))
    if s['festival']:
        scenarios.append(('Festival-day demand', {'festival': False}))
    if s['rating'] < 4.8:
        scenarios.append((f"Partner rating {s['rating']:.1f}★ (top partners ≈ 4.8★)", {'rider_rating': 4.8}))
    if s['v_cond'] < 2:
        scenarios.append((f"Vehicle condition: {VEHICLE_COND[s['v_cond']].lower()}", {'vehicle_condition': 2}))
    if s['distance'] > 9.2:
        scenarios.append((f"Longer trip ({s['distance']:.1f} km vs. 9.2 km median)", {'distance_km': 9.2}))
    factors = []
    if scenarios:
        frames = pd.concat([make_order_frame(**{**inputs, **chg}) for _, chg in scenarios], ignore_index=True)
        alt_eta, _ = predict(frames)
        factors = sorted([(lbl, eta - a) for (lbl, _), a in zip(scenarios, alt_eta) if eta - a >= 0.5],
                         key=lambda x: -x[1])[:4]

    with right:
        # --- ETA card ------------------------------------------------------------------
        placed = dt.datetime.combine(dt.date.today(), dt.time(s['order_hour'], 0))
        arrive = placed + dt.timedelta(minutes=eta)
        # The two models are trained separately, so they can disagree near the 25-min line.
        # If the ETA is past the SLA (or the risk is close to 50%), show an amber "at risk" state.
        at_risk = not delayed and (eta > DELAY_THRESHOLD_MIN or risk >= 0.35)
        if delayed:
            chip, bar_col = f'<span class="tb-chip bad">⚠️ Likely delayed · {risk * 100:.0f}% risk</span>', '#C62828'
        elif at_risk:
            chip, bar_col = f'<span class="tb-chip warn">⏳ At risk of delay · {risk * 100:.0f}% risk</span>', '#D98A00'
        else:
            chip, bar_col = f'<span class="tb-chip ok">✅ On time · {(1 - risk) * 100:.0f}% confidence</span>', '#16834A'
        actual_html = ''
        if s.get('actual'):
            a = s['actual']
            err = eta - a['time']
            actual_html = (f'<div class="tb-banner okb" style="margin-top:12px">📒 Real order <b>{a["id"].strip()}</b> '
                           f'actually took <b>{a["time"]:.0f} min</b> → model is off by <b>{abs(err):.1f} min</b> '
                           f'({"over" if err > 0 else "under"}-estimate).</div>')
        st.markdown(f"""
<div class="tb-card">
  <div class="tb-muted">Order from <b>{rest['name']}</b> · {ORDER_TYPES[s['order_type']]} · {s['distance']:.1f} km</div>
  <div class="tb-eta" style="margin-top:6px">
    <span class="big">{eta_min}</span><span class="unit">mins</span>
    <span class="tb-muted" style="margin-bottom:8px">Arriving by <b>{arrive:%I:%M %p}</b> · likely {eta_min - margin}–{eta_min + margin} min</span>
  </div>
  <div style="margin-top:10px">{chip} <span class="tb-chip info">SLA: {DELAY_THRESHOLD_MIN:.0f} min</span></div>
  <div class="tb-bar"><div style="width:{risk * 100:.0f}%; background:{bar_col}"></div></div>
  <div class="tb-muted">Probability this order takes longer than {DELAY_THRESHOLD_MIN:.0f} min (delay classifier)</div>
  {actual_html}
</div>""", unsafe_allow_html=True)

        # --- Live status timeline (simulated clock) --------------------------------------
        pickup_at = min(float(s['prep']), eta * 0.5)
        total = max(eta_min, 1)
        if st.session_state['sim_t'] > total:
            st.session_state['sim_t'] = total
        t = st.session_state['sim_t']
        stages = [('🧾', 'Order placed', 0.0), ('👩‍🍳', 'Preparing', 0.5), ('🛍️', 'Picked up', pickup_at),
                  ('🛵', 'On the way', pickup_at + 1), ('🏠', 'Delivered', float(total))]
        current = max(i for i, (_, _, at) in enumerate(stages) if t >= at) if t > 0 else 0
        steps_html = ''
        for i, (ico, lbl, at) in enumerate(stages):
            cls = 'done' if i < current else ('now' if i == current else '')
            when = (placed + dt.timedelta(minutes=at)).strftime('%I:%M %p')
            steps_html += f'<div class="tb-step {cls}"><div class="dot">{ico}</div>{lbl}<span class="t">{when}</span></div>'
        status_msg = {0: 'Restaurant has received your order', 1: f"{rest['name']} is preparing your food",
                      2: 'Your partner has picked up the order', 3: f"On the way · about {max(0, eta_min - t)} min to go",
                      4: 'Delivered — enjoy your meal! 🎉'}[current]
        st.markdown(f"""
<div class="tb-card">
  <p class="tb-h">📍 Live status <span class="tb-muted" style="font-weight:400">· {status_msg}</span></p>
  <div class="tb-steps">{steps_html}</div>
</div>""", unsafe_allow_html=True)
        st.slider('⏱️ Simulate the clock: minutes since the order was placed', 0, total, key='sim_t')

        # --- Map: restaurant -> rider -> home ---------------------------------------------
        bearing = math.radians((sum(map(ord, rest['name'])) * 37) % 360)
        lat0, lon0 = rest['lat'], rest['lon']
        dlat = s['distance'] / 111.0 * math.cos(bearing)
        dlon = s['distance'] / (111.0 * math.cos(math.radians(lat0))) * math.sin(bearing)
        lat1, lon1 = lat0 + dlat, lon0 + dlon
        # gentle curve so the path looks like a road route rather than a ruler line
        ctrl_lat, ctrl_lon = (lat0 + lat1) / 2 - dlon * 0.25, (lon0 + lon1) / 2 + dlat * 0.25
        ts = np.linspace(0, 1, 30)
        path = [[float((1 - u) ** 2 * lon0 + 2 * (1 - u) * u * ctrl_lon + u ** 2 * lon1),
                 float((1 - u) ** 2 * lat0 + 2 * (1 - u) * u * ctrl_lat + u ** 2 * lat1)] for u in ts]
        frac = 0.0 if t < pickup_at else min(1.0, (t - pickup_at) / max(eta - pickup_at, 1))
        rider = path[int(round(frac * (len(path) - 1)))]
        points = pd.DataFrame([
            {'lon': lon0, 'lat': lat0, 'label': f"🍽️ {rest['name']}", 'color': [43, 26, 16], 'r': 60.0},
            {'lon': lon1, 'lat': lat1, 'label': '🏠 You', 'color': [22, 131, 74], 'r': 60.0},
            {'lon': rider[0], 'lat': rider[1], 'label': '🛵 Delivery partner', 'color': [232, 99, 10], 'r': 90.0}])
        deck = pdk.Deck(
            layers=[pdk.Layer('PathLayer', data=[{'path': path}], get_path='path', get_color=[232, 99, 10, 160],
                              get_width=30, width_min_pixels=5, cap_rounded=True, joint_rounded=True),
                    pdk.Layer('ScatterplotLayer', data=points, get_position='[lon, lat]', get_fill_color='color',
                              get_radius='r', radius_min_pixels=7, radius_max_pixels=16, stroked=True, get_line_color=[255, 255, 255],
                              line_width_min_pixels=3, pickable=True)],
            initial_view_state=pdk.ViewState(latitude=(lat0 + lat1) / 2, longitude=(lon0 + lon1) / 2,
                                             zoom=max(10.0, 14.2 - math.log2(max(s['distance'], 1.0))), pitch=0),
            tooltip={'text': '{label}'}, map_style=None)
        st.pydeck_chart(deck, height=290)

        # --- Partner + factors ---------------------------------------------------------
        c1, c2 = st.columns([1, 1.25])
        with c1:
            st.markdown(f"""
<div class="tb-card">
  <p class="tb-h">Your delivery partner</p>
  <div class="tb-rider"><div class="tb-avatar">{VEHICLES[s['vehicle']].split()[0]}</div>
  <div><b>{s['rating']:.1f} ★</b> · {s['age']} yrs<br><span class="tb-muted">{VEHICLES[s['vehicle']].split(' ', 1)[1]} ·
  {VEHICLE_COND[s['v_cond']].lower()} condition<br>{'Direct delivery' if s['multi'] == 0 else f"+{s['multi']} batched order(s)"}</span></div></div>
</div>""", unsafe_allow_html=True)
        with c2:
            if factors:
                rows = ''.join(f'<div class="tb-factor"><span>{lbl}</span><b>+{d:.0f} min</b></div>' for lbl, d in factors)
                st.markdown(f'<div class="tb-card"><p class="tb-h">⏳ What is adding time</p>{rows}'
                            f'<div class="tb-muted" style="margin-top:6px">Model what-if: ETA change if this one condition were ideal.</div></div>',
                            unsafe_allow_html=True)
            else:
                st.markdown('<div class="tb-card"><p class="tb-h">⏳ What is adding time</p>'
                            '<div class="tb-banner okb">Smooth conditions — nothing is slowing this order down.</div></div>',
                            unsafe_allow_html=True)

# ======================================================================================
# TAB 2 - model insights
# ======================================================================================
with tab_model:
    rm, cm = reg_bundle['metrics'], clf_bundle['metrics']
    st.markdown('<p class="tb-h">How good are the predictions? (held-out test set, '
                f'{reg_bundle["n_test"]:,} orders)</p>', unsafe_allow_html=True)
    cards = [(f"{rm['R2']:.3f}", 'ETA model · R²'), (f"{rm['MAE']:.2f} min", 'ETA model · MAE'),
             (f"{rm['within_5_min'] * 100:.0f}%", 'ETA within ±5 min'), (f"{cm['Accuracy'] * 100:.1f}%", 'Delay model · accuracy'),
             (f"{cm['F1']:.3f}", 'Delay model · weighted F1'), (f"{cm['ROC-AUC']:.3f}", 'Delay model · ROC-AUC')]
    cols = st.columns(6)
    for col, (v, l) in zip(cols, cards):
        col.markdown(f'<div class="tb-metric"><div class="v">{v}</div><div class="l">{l}</div></div>', unsafe_allow_html=True)
    st.write('')

    def importance_chart(bundle, title, color):
        imp = (pd.Series(bundle['model'].feature_importances_, index=bundle['feature_names'])
               .sort_values(ascending=False).head(10).rename('importance').reset_index()
               .rename(columns={'index': 'feature'}))
        imp['feature'] = (imp['feature'].str.replace('Delivery_person_', 'Partner ', regex=False)
                          .str.replace('Road_traffic_density_', 'Traffic: ', regex=False)
                          .str.replace('Weatherconditions_', 'Weather: ', regex=False)
                          .str.replace('Type_of_vehicle_', 'Vehicle: ', regex=False)
                          .str.replace('City_', 'Area: ', regex=False).str.replace('_', ' ', regex=False))
        return (alt.Chart(imp, title=title).mark_bar(color=color, cornerRadiusEnd=4)
                .encode(x=alt.X('importance:Q', title='Impurity-based importance'),
                        y=alt.Y('feature:N', sort='-x', title=None, axis=alt.Axis(labelLimit=220)),
                        tooltip=['feature', alt.Tooltip('importance:Q', format='.3f')]).properties(height=330))

    c1, c2 = st.columns(2)
    c1.altair_chart(importance_chart(reg_bundle, 'ETA model (Gradient Boosting Regressor) — top 10 features', '#E8630A'),
                    width='stretch')
    c2.altair_chart(importance_chart(clf_bundle, 'Delay model (Gradient Boosting Classifier) — top 10 features', '#8C3B12'),
                    width='stretch')

    st.markdown(f"""
<div class="tb-card">
<p class="tb-h">Pipeline behind the screen</p>
<ol style="margin:0 0 0 18px; line-height:1.9">
<li><b>Clean</b> — strip text, turn hidden "NaN" tokens into missing values, cap ratings above 5, fix 3,640 zero coordinates.</li>
<li><b>Engineer</b> — haversine <code>distance_km</code>, kitchen <code>prep_time_min</code>, <code>order_hour</code>, rush-hour, weekend and festival flags.</li>
<li><b>Pre-process</b> — median-impute + standard-scale 10 numeric features, mode-impute + one-hot 4 categoricals (fitted on the training split only).</li>
<li><b>Predict</b> — {reg_bundle['name']} for minutes, {clf_bundle['name']} for P(delay &gt; {DELAY_THRESHOLD_MIN:.0f} min).</li>
</ol>
<div class="tb-muted" style="margin-top:8px">Both models were chosen after comparing 10 algorithms per track with GridSearchCV and 5-fold CV — see the notebooks.
Trained with scikit-learn {reg_bundle.get('sklearn_version', '?')} on {reg_bundle['n_train']:,} orders.</div>
</div>""", unsafe_allow_html=True)

# ======================================================================================
# TAB 3 - about
# ======================================================================================
with tab_about:
    st.markdown(f"""
<div class="tb-card">
<p class="tb-h">About this app</p>
TrackBite is the interactive GUI for our <b>23CSE301 Machine Learning capstone</b> on the <i>Online Food Order Tracking</i>
dataset ({len(orders):,} real deliveries across {len(CITY_LIST)} Indian cities). Fill in an order — or load a real one —
and the app shows what a food-delivery tracker would: an ETA, an arrival time, a delay warning and the live delivery stages.
<br><br>
<b>Notes</b>
<ul>
<li>Restaurant names and cuisines are invented labels on top of real pickup coordinates from the dataset.</li>
<li>The map route is illustrative (a curve of the chosen length); the model only uses the distance.</li>
<li>"Delayed" means the order takes more than {DELAY_THRESHOLD_MIN:.0f} minutes, the threshold used in the classification track.</li>
</ul>
Source code, notebooks and README: <a href="https://github.com/Sank19-code/Online-Food-Order-Tracking" target="_blank">github.com/Sank19-code/Online-Food-Order-Tracking</a>
</div>""", unsafe_allow_html=True)
