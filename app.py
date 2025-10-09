import streamlit as st
import pandas as pd
import os
from datetime import datetime

st.set_page_config(page_title="Face Recognition Attendance", page_icon="🧠", layout="wide")

st.title("📅 Face Recognition Attendance Dashboard")

# --- Load Attendance files ---
attendance_dir = "Attendance"
if not os.path.exists(attendance_dir):
    st.error("No Attendance directory found.")
else:
    files = [f for f in os.listdir(attendance_dir) if f.endswith(".csv")]
    if not files:
        st.warning("No attendance records found.")
    else:
        files.sort(reverse=True)
        selected_file = st.selectbox("Select Attendance File", files)

        if selected_file:
            path = os.path.join(attendance_dir, selected_file)
            df = pd.read_csv(path)
            st.subheader(f"📋 Records for {selected_file.replace('Attendance_', '').replace('.csv','')}")
            st.dataframe(df)

            st.markdown("---")
            st.subheader("📊 Summary")

            total_present = df['NAME'].nunique()
            st.metric("Total Unique Students Present", total_present)

            st.bar_chart(df['NAME'].value_counts())

            if st.button("🔁 Refresh Data"):
                st.experimental_rerun()
