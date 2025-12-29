import streamlit as st
import pandas as pd
import joblib
import plotly.express as px 

# --- 1. SETUP & LOAD DATA ---
st.set_page_config(page_title="AI Job Risk 2030", page_icon="📊", layout="wide")

# โหลดโมเดล
try:
    model = joblib.load('real_ai_model_lvl2.pkl')
    le_job = joblib.load('le_job.pkl')
    le_edu = joblib.load('le_edu.pkl')
except FileNotFoundError:
    st.error("ไม่พบไฟล์โมเดล! กรุณารัน train_model.py ก่อน")
    st.stop()

# โหลดข้อมูลดิบ (สำหรับทำ Dashboard ใน Tab 2)
# @st.cache_data ช่วยให้โหลดเร็วขึ้น ไม่ต้องโหลดใหม่ทุกครั้งที่กดปุ่ม
@st.cache_data
def load_data():
    df = pd.read_csv('AI_Impact_on_Jobs_2030.csv')
    return df

try:
    df_global = load_data()
except:
    st.warning("ไม่พบไฟล์ CSV สำหรับทำ Dashboard (แต่ส่วนทำนายยังใช้งานได้)")
    df_global = None

# --- 2. MAIN HEADER ---
st.title("📊 AI Impact & Future of Work 2030")
st.write("แพลตฟอร์มวิเคราะห์ความเสี่ยงและแนวโน้มอาชีพในอนาคต")

# สร้าง Tabs แยกหน้าจอ
tab1, tab2 = st.tabs(["🔮 ประเมินความเสี่ยง (Simulator)", "📈 แนวโน้มตลาดแรงงาน (Dashboard)"])

# ==========================================
# TAB 1: PREDICTION 
# ==========================================
with tab1:
    st.header("ประเมินความเสี่ยงรายบุคคล")
    
    with st.form("risk_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            job_list = list(le_job.classes_)
            default_job = job_list.index('Data Scientist') if 'Data Scientist' in job_list else 0
            selected_job = st.selectbox("อาชีพของคุณ", job_list, index=default_job)
            
            edu_list = list(le_edu.classes_)
            selected_edu = st.selectbox("วุฒิการศึกษา", edu_list)

        with col2:
            ai_exposure = st.slider("ความเกี่ยวข้องกับ AI (AI Exposure)", 0.0, 1.0, 0.5)
            st.caption("0.0 = งานแรงงาน | 1.0 = งานหน้าคอมฯ")
            experience = st.number_input("ประสบการณ์ (ปี)", 0, 40, 5)

        st.markdown("---")
        st.write("**สถานะสกิลปัจจุบัน (Current Skills)**")
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            has_tech = st.checkbox("มีทักษะด้าน Tech / Coding / Data")
        with col_s2:
            has_soft = st.checkbox("มีทักษะด้าน Management / Communication")

        submitted = st.form_submit_button("🚀 วิเคราะห์ความเสี่ยง")

    if submitted:
        # เตรียมข้อมูล
        job_val = le_job.transform([selected_job])[0]
        edu_val = le_edu.transform([selected_edu])[0]
        tech_val = 1 if has_tech else 0
        soft_val = 1 if has_soft else 0

        input_data = pd.DataFrame([[job_val, edu_val, ai_exposure, experience, tech_val, soft_val]],
                                  columns=['Job_Title_Encoded', 'Education_Level_Encoded', 
                                           'AI_Exposure_Index', 'Years_Experience',
                                           'Tech_Skills', 'Soft_Skills'])
        
        # ทำนายผล
        prediction = model.predict(input_data)[0]
        risk_score = max(0, min(100, prediction)) # Safety Guard

        st.divider()
        col_res1, col_res2 = st.columns([1, 2])
        
        with col_res1:
             # แสดงผลแบบ Gauge Chart (มาตรวัด) สวยๆ
             st.metric(label="ความเสี่ยงปัจจุบัน", value=f"{risk_score:.1f}%")
             if risk_score > 70:
                st.error("High Risk")
             elif risk_score > 30:
                st.warning("Medium Risk")
             else:
                st.success("Low Risk")

        with col_res2:
            st.info("💡 **คำแนะนำ:** " + ("ควรพัฒนาทักษะใหม่ๆ ทันที" if risk_score > 50 else "รักษามาตรฐานและติดตามเทคโนโลยีสม่ำเสมอ"))

        # --- SIMULATOR SECTION ---
        st.markdown("### 🛡️ Upskilling Simulator")
        sim_col1, sim_col2 = st.columns(2)

        # Logic เดิม (Sim 1 Tech)
        if tech_val == 1:
            sim_col1.metric("ถ้าเรียน Tech เพิ่ม", f"{risk_score:.1f}%", "มีแล้ว ✅", delta_color="off")
        else:
            sim_tech = input_data.copy()
            sim_tech['Tech_Skills'] = 1
            new_risk = max(0, min(100, model.predict(sim_tech)[0]))
            diff = risk_score - new_risk
            sim_col1.metric("ถ้าเรียน Tech เพิ่ม", f"{new_risk:.1f}%", f"-{diff:.1f}%", delta_color="normal")

        # Logic เดิม (Sim 2 Soft)
        if soft_val == 1:
            sim_col2.metric("ถ้าเรียน Management เพิ่ม", f"{risk_score:.1f}%", "มีแล้ว ✅", delta_color="off")
        else:
            sim_soft = input_data.copy()
            sim_soft['Soft_Skills'] = 1
            new_risk = max(0, min(100, model.predict(sim_soft)[0]))
            diff = risk_score - new_risk
            sim_col2.metric("ถ้าเรียน Management เพิ่ม", f"{new_risk:.1f}%", f"-{diff:.1f}%", delta_color="normal")

# ==========================================
# TAB 2: DASHBOARD 
# ==========================================
with tab2:
    st.header("📈 ภาพรวมตลาดแรงงาน (Market Insights)")
    
    if df_global is not None:
        # กราฟที่ 1: ความเสี่ยงเฉลี่ยแยกตามระดับการศึกษา
        st.subheader("1. การศึกษายิ่งสูง ความเสี่ยงยิ่งต่ำจริงไหม?")
        avg_risk_edu = df_global.groupby('Education_Level')['Automation_Probability_2030'].mean().reset_index()
        fig1 = px.bar(avg_risk_edu, x='Education_Level', y='Automation_Probability_2030', 
                      color='Automation_Probability_2030', color_continuous_scale='Reds',
                      title="Average Automation Risk by Education Level")
        st.plotly_chart(fig1, use_container_width=True)

        # กราฟที่ 2: Scatter Plot ระหว่าง AI Exposure vs Risk
        st.subheader("2. ยิ่งใกล้ชิด AI ยิ่งเสี่ยงตกงาน?")
        # สุ่มมาสัก 500 จุดพอ เดี๋ยวคอมค้าง
        sample_df = df_global.sample(500) 
        fig2 = px.scatter(sample_df, x='AI_Exposure_Index', y='Automation_Probability_2030',
                          color='Risk_Category', size='Average_Salary', hover_data=['Job_Title'],
                          title="Correlation: AI Exposure vs Automation Probability")
        st.plotly_chart(fig2, use_container_width=True)
        
        st.info("Tip: กราฟนี้บอกว่างานที่ต้องยุ่งกับ AI เยอะๆ (ขวา) มักจะมีความเสี่ยงสูง (บน) แต่ถ้าเงินเดือนสูง (วงใหญ่) อาจจะรอดได้")

    else:
        st.error("ไม่สามารถโหลดข้อมูล Dashboard ได้ กรุณาเช็คไฟล์ CSV")