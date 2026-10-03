import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(APP_DIR.parent))

import streamlit as st


st.set_page_config(page_title="DemandLens", page_icon=":material/monitoring:", layout="wide")

home = st.Page(str(APP_DIR / "views" / "home.py"), title="Home Page", icon=":material/home:", default=True)
overview = st.Page(str(APP_DIR / "views" / "overview.py"), title="Overview", icon=":material/bar_chart:")
forecast_explorer = st.Page(str(APP_DIR / "views" / "forecast_explorer.py"), title="Forecast Explorer", icon=":material/trending_up:")
anomaly_view = st.Page(str(APP_DIR / "views" / "anomaly_view.py"), title="Anomaly View", icon=":material/warning:")
ai_report = st.Page(str(APP_DIR / "views" / "ai_report.py"), title="AI Report", icon=":material/smart_toy:")

pg = st.navigation([home, overview, forecast_explorer, anomaly_view, ai_report])
pg.run()
