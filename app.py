import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import time
import warnings
warnings.filterwarnings('ignore')

# Banner logo
st.image("FITNESS-TRACKER_logo.png", use_column_width=True)

# Toast
st.toast("Keep working hard 🦾 ⏳")

# App Title
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>FITNESS-TRACKER 💪</h1>", unsafe_allow_html=True)

# About Section
with st.expander("About This WebApp"):
    st.write("This FITNESS-TRACKER is a budget-friendly virtual fitness coach for those who are passionate about tracking their calorie burn without needing expensive gadgets.")
    st.header("Key Features:")
    st.write("1. User input form")
    st.write("2. Machine Learning model (RandomForestRegressor)")
    st.write("3. Plotly Scatter plot")
    st.write("4. Fitness Insights")
    st.write("5. Model Performance Evaluations")

# How to Use Section
with st.expander("How to use the WebApp"):
    st.write("1. Use the sidebar on the left")
    st.write("2. Enter your personal and workout details")
    st.write("3. Submit your data to predict calories burned")
    st.write("4. Analyze insights and model feedback")

# Sidebar Input
st.sidebar.header("Enter your data")

if "age" not in st.session_state:
    st.session_state.age = 25
if "gender" not in st.session_state:
    st.session_state.gender = "Male"
if "bmi" not in st.session_state:
    st.session_state.bmi = 25.0

def user_input_features():
    st.session_state.age = st.sidebar.number_input("Enter Age", min_value=10, max_value=100, value=st.session_state.age)
    st.session_state.gender = st.sidebar.selectbox("Select Gender", ["Male", "Female"], index=0 if st.session_state.gender == "Male" else 1)
    st.session_state.bmi = st.sidebar.number_input("Enter BMI", min_value=10.0, max_value=50.0, value=st.session_state.bmi)
    duration = st.sidebar.slider("Daily Workout Duration (min):", 0, 30, 15)
    heart_rate = st.sidebar.slider("Heart Rate during Workout (BPM):", 60, 130, 80)
    body_temp = st.sidebar.slider("Body Temperature (°C) during Workout:", 36, 42, 41)
    gender_encoded = 1 if st.session_state.gender == "Male" else 0

    data_model = {
        "Age": st.session_state.age,
        "BMI": st.session_state.bmi,
        "Duration": duration,
        "Heart_Rate": heart_rate,
        "Body_Temp": body_temp,
        "Gender": gender_encoded
    }

    features = pd.DataFrame([data_model])
    return features

df = user_input_features()

st.write("---")
with st.expander("Your Parameters"):
    st.write(df)
st.write("---")

# Load Dataset
calories = pd.read_csv("calories.csv")
exercise = pd.read_csv("exercise.csv")
exercise_df = exercise.merge(calories, on="User_ID")
exercise_df.drop(columns="User_ID", inplace=True)

exercise_train_data, exercise_test_data = train_test_split(exercise_df, test_size=0.2, random_state=1)

for data in [exercise_train_data, exercise_test_data]:
    data["BMI"] = data["Weight"] / ((data["Height"] / 100) ** 2)
    data["BMI"] = round(data["BMI"], 2)

exercise_train_data = exercise_train_data[["Gender", "Age", "BMI", "Duration", "Heart_Rate", "Body_Temp", "Calories"]]
exercise_test_data = exercise_test_data[["Gender", "Age", "BMI", "Duration", "Heart_Rate", "Body_Temp", "Calories"]]
exercise_train_data = pd.get_dummies(exercise_train_data, drop_first=True)
exercise_test_data = pd.get_dummies(exercise_test_data, drop_first=True)

X_train = exercise_train_data.drop("Calories", axis=1)
y_train = exercise_train_data["Calories"]
X_test = exercise_test_data.drop("Calories", axis=1)
y_test = exercise_test_data["Calories"]

random_reg = RandomForestRegressor(n_estimators=1000, max_features=3, max_depth=6)
random_reg.fit(X_train, y_train)

df = df.reindex(columns=X_train.columns, fill_value=0)
prediction = random_reg.predict(df)

# Visualize
st.write("### Workout Duration vs Calories Burned")
latest_iteration = st.empty()
bar = st.progress(0)
for i in range(100):
    bar.progress(i + 1)
    time.sleep(0.01)

fii = px.scatter(exercise_train_data, x="Duration", y="Body_Temp", size="Calories")
fii.update_layout(width=700, height=450)
fii.add_scatter(
    x=[df["Duration"].values[0]],
    y=[df["Body_Temp"].values[0]],
    mode="markers+text",
    marker=dict(size=18, color="red", symbol="star"),
    name="Your Input",
    text=["You"],
    textposition="top center"
)
st.plotly_chart(fii)

# Prediction
st.write("---")
st.write("### Prediction: ")
bar = st.progress(0)
for i in range(100):
    bar.progress(i + 1)
    time.sleep(0.01)
st.write(f"🔥 *{round(prediction[0], 2)} kilocalories burned today as per your data.*")

# Similar Results
st.write("---")
st.write("### Similar Results: ")
bar = st.progress(0)
for i in range(100):
    bar.progress(i + 1)
    time.sleep(0.01)

calorie_range = [prediction[0] - 10, prediction[0] + 10]
similar_data = exercise_df[
    (exercise_df["Calories"] >= calorie_range[0]) & 
    (exercise_df["Calories"] <= calorie_range[1])
]

if not similar_data.empty:
    st.write("### People with Similar Calorie Burn:")
    st.write(similar_data.sample(min(5, len(similar_data))))
else:
    st.write("⚠️ No similar results found in the dataset.")

# General Info
st.write("---")
st.header("General Information: ")

boolean_age = (exercise_df["Age"] < df["Age"].values[0]).tolist()
boolean_duration = (exercise_df["Duration"] < df["Duration"].values[0]).tolist()
boolean_body_temp = (exercise_df["Body_Temp"] < df["Body_Temp"].values[0]).tolist()
boolean_heart_rate = (exercise_df["Heart_Rate"] < df["Heart_Rate"].values[0]).tolist()

st.write("📊 You are older than", round(sum(boolean_age) / len(boolean_age), 2) * 100, "% of other people.")
st.write("⏱️ Your exercise duration is higher than", round(sum(boolean_duration) / len(boolean_duration), 2) * 100, "% of other people.")
st.write("❤️ Your heart rate is higher than", round(sum(boolean_heart_rate) / len(boolean_heart_rate), 2) * 100, "% of other people.")
st.write("🌡️ Your body temperature is higher than", round(sum(boolean_body_temp) / len(boolean_body_temp), 2) * 100, "% of other people.")

# Model Evaluation
st.write("---")
y_pred = random_reg.predict(X_test)
r2 = r2_score(y_test, y_pred)

st.write("### R² Score of the Model")
st.write(f"R² Score: {round(r2, 2)}")
if r2 >= 0.9:
    st.success("Excellent Model: Very accurate and reliable!")
elif r2 >= 0.75:
    st.info("Good Model: Explains a significant portion of the variance.")
elif r2 >= 0.5:
    st.warning("Average Model: Usable but could be improved.")
elif r2 >= 0:
    st.error("Poor Model: Explains very little variance.")
else:
    st.error("Very Bad Model: Worse than a random guess!")

st.write("### Thank You !! 🙏")
