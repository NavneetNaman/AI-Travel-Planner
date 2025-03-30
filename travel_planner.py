import os
import streamlit as st
from datetime import date, timedelta
from llama_cpp import Llama

# 🔥 Load Mistral Model
@st.cache_resource()
def load_model():
    MODEL_PATH = r"C:\Users\naman\.ai-navigator\models\mistralai\Mistral-7B-Instruct-v0.2\Mistral-7B-Instruct-v0.2_Q4_K_M.gguf"
    
    if not os.path.exists(MODEL_PATH):
        st.error("❌ Model file not found! Please check the path.")
        return None
    
    return Llama(model_path=MODEL_PATH, n_gpu_layers=50, n_ctx=4096, n_batch=256)

llm = load_model()

# 🎯 **Title**
st.title("🗺️ AI Travel Planner")

# 🖼️ **Load Image**
image_path = r"C:\Users\naman\OneDrive\Desktop\ai-ml internship\img.png"
if os.path.exists(image_path):
    st.image(image_path, caption="🌍 Explore the world!", use_container_width=True)
else:
    st.warning("⚠️ Image not found! Using default image.")
    st.image("https://upload.wikimedia.org/wikipedia/commons/6/63/Travel_Illustration.png", 
             caption="🌍 Explore the world!", use_container_width=True)

# 🎨 **User Inputs**
destination = st.text_input("📍 Enter your destination:")
start_date = st.date_input("📅 Start date of travel:", min_value=date.today())
end_date = st.date_input("📅 End date of travel:", min_value=start_date)
budget = st.selectbox("💰 Select your budget:", ["Low", "Moderate", "High"])
preferences = st.selectbox("🎭 Travel preferences:", ["Nature", "History", "Adventure", "Culture", "Relaxation"])
accommodation = st.selectbox("🏨 Accommodation:", ["Luxury", "Mid-range", "Budget"])
dietary = st.selectbox("🥗 Dietary:", ["None", "Vegetarian", "Vegan", "Gluten-free"])
mobility = st.selectbox("♿ Mobility concerns:", ["No", "Yes"])

if end_date < start_date:
    st.error("❌ End date cannot be before the start date!")

# 🔘 **Generate Button**
if st.button("🚀 Generate Itinerary"):
    if destination and start_date and end_date and start_date <= end_date:
        with st.spinner("🧳 Creating your itinerary..."):
            if not llm:
                st.error("❌ Model not loaded. Please check your setup.")
            else:
                trip_duration = (end_date - start_date).days + 1

                # 🔥 **Corrected Prompt**
                prompt = f"""Create a {trip_duration}-day itinerary for {destination}.
                - Budget: {budget}
                - Preferences: {preferences}
                - Accommodation: {accommodation}
                - Dietary: {dietary}
                - Mobility: {mobility}

                Format strictly as:
                """

                for i in range(trip_duration):
                    day = start_date + timedelta(days=i)
                    prompt += f"""
**DAY {i+1} ({day.strftime('%Y-%m-%d')})**  
**Morning:**  
[Detailed description of morning activity]  

**Afternoon:**  
[Detailed description of afternoon activity]  

**Evening:**  
[Detailed description of evening activity]  
"""

                prompt += "\nKeep responses concise, avoid repetition."

                # 🚀 **Streaming Response (Fixes Cache Error)**
                response = llm(prompt, max_tokens=min(1500, trip_duration * 300), temperature=0.6, top_p=0.9, stream=True)

                itinerary_placeholder = st.empty()  # Placeholder for live output
                
                itinerary_text = ""
                for chunk in response:
                    itinerary_text += chunk["choices"][0]["text"]
                    itinerary_placeholder.markdown(itinerary_text.replace("\n", "\n\n"), unsafe_allow_html=True)
    else:
        st.error("⚠️ Please fill in all required fields correctly.")
