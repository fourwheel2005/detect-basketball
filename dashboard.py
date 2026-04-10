# ไฟล์ dashboard.py
import streamlit as st
import pandas as pd
import plotly.express as px
import os

st.set_page_config(page_title="AI Referee Dashboard", layout="wide")

st.title("🏀 AI Basketball Referee - Analytics Dashboard")
st.markdown("วิเคราะห์สถิติการทำฟาวล์จากระบบ AI")

log_file = "basketball_foul_logs.csv"

if os.path.exists(log_file):
    # อ่านข้อมูล CSV
    df = pd.read_csv(log_file)
    df['Date_Time'] = pd.to_datetime(df['Date_Time'])
    
    # แบ่งคอลัมน์แสดงผล
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("สถิติการฟาวล์ทั้งหมด")
        foul_counts = df['Foul_Type'].value_counts().reset_index()
        foul_counts.columns = ['Foul_Type', 'Count']
        
        # กราฟแท่ง
        fig_bar = px.bar(foul_counts, x='Foul_Type', y='Count', color='Foul_Type', 
                         title="จำนวนการทำฟาวล์แต่ละประเภท")
        st.plotly_chart(fig_bar, use_container_width=True)
        
    with col2:
        st.subheader("ข้อมูลล่าสุด (Log Data)")
        st.dataframe(df.sort_values(by='Date_Time', ascending=False).head(20), use_container_width=True)
else:
    st.warning(f"ยังไม่พบไฟล์ข้อมูลสถิติ ({log_file}) กรุณารันระบบกรรมการ AI เพื่อเก็บข้อมูลก่อนครับ")