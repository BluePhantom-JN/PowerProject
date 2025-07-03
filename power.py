import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from torch import nn, tensor, float32
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

# 🏠 Set page configuration
st.set_page_config(layout="wide", page_title="Power Consumption Dashboard")
st.title("⚡ Household Power & Weather Dashboard")

path = r"G:\python\code\New DataSets\household_power_consumption.txt"
df = pd.read_csv(path,delimiter=';',dtype=str)
df['Sub_metering_3']=df['Sub_metering_3'].fillna(value=0)
df=df.replace('?',0)
df = df.astype({'Global_active_power':float,'Global_reactive_power':float,'Sub_metering_1':float,'Sub_metering_2':float,'Sub_metering_3':float,'Voltage':float,'Global_intensity':float})

path1 = r"G:\python\code\New DataSets\temprature data.csv"

temp = pd.read_csv(path1)
temp['datetime'] = pd.to_datetime(temp['datetime'])
temp.rename(columns={'datetime':'Date'},inplace=True)
temp.drop(columns=['name', 'tempmax', 'tempmin', 'feelslikemax',
       'feelslikemin', 'feelslike', 'precip', 'precipprob',
       'precipcover', 'preciptype', 'snow', 'snowdepth', 'windgust',
       'windspeed', 'winddir', 'sealevelpressure', 'cloudcover', 'visibility',
       'solarradiation', 'solarenergy', 'uvindex', 'severerisk', 'sunrise',
       'sunset', 'moonphase', 'conditions', 'description', 'icon', 'stations'],inplace=True)


# 📦 Assume df and temp are already loaded
# You must load this before running app.py:
# from my_data import df, temp  # just make sure `df` and `temp` are accessible

# -------------------- Preprocessing --------------------
df1 = df.copy()
df1['total_power(kw)'] = df['Global_active_power'] + df['Global_reactive_power']
df1['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
df1['Time'] = pd.to_datetime(df['Time'], format='%H:%M:%S', errors='coerce').dt.time

temp['Date'] = pd.to_datetime(temp['Date'])

# Group and merge
df2 = df1.groupby('Date')[['Global_active_power', 'Global_reactive_power',
                           'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3',
                           'total_power(kw)']].sum().reset_index()
df3 = df1.groupby('Date')[['Voltage', 'Global_intensity']].mean().reset_index()
df2 = pd.merge(df2, df3, on='Date', how='left')
df4 = pd.merge(df2, temp, on='Date', how='left').fillna(0)

# -------------------- EDA Section --------------------
st.header("🔍 Exploratory Data Analysis")
df4['day'] = df4['Date'].dt.day

# Line plot: daily average power
col1, col2 = st.columns(2)
with col1:
    st.subheader("📈 Daily Avg Total Power")
    daily_avg = df4.groupby('day')['total_power(kw)'].mean()
    fig, ax = plt.subplots()
    ax.plot(daily_avg.index, daily_avg.values, marker='o')
    ax.set_xlabel("Day of Month")
    ax.set_ylabel("Avg Total Power (kW)")
    ax.grid(True)
    st.pyplot(fig)

with col2:
    st.subheader("🕒 Hourly Avg Power")
    df['hour'] = pd.to_datetime(df['Time'], errors='coerce').dt.hour
    hourly_avg = df.groupby('hour')['Global_active_power'].mean()
    fig2, ax2 = plt.subplots()
    ax2.plot(hourly_avg.index, hourly_avg.values, marker='o', color='orange')
    ax2.set_xlabel("Hour")
    ax2.set_ylabel("Avg Power")
    ax2.grid(True)
    st.pyplot(fig2)

# Distribution
st.subheader("🔋 Distribution of Total Power")
fig3 = plt.figure()
sns.histplot(df4['total_power(kw)'], bins=50, kde=True)
st.pyplot(fig3)

# Submetering
st.subheader("📊 Sub Metering Bar Chart")
fig4 = df2[['Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']].sum().reset_index().plot(kind='bar', x='index').get_figure()
st.pyplot(fig4)

# -------------------- Outlier Removal --------------------
for col in ['Global_active_power', 'Global_intensity', 'Voltage']:
    z_col = f'{col}_z'
    df4[z_col] = (df4[col] - df4[col].mean()) / df4[col].std()
    df4 = df4[df4[z_col].between(-3, 3)]

df4.drop(columns=[col for col in df4.columns if col.endswith('_z')], inplace=True)

# -------------------- Correlation --------------------
st.header(" Correlation Heatmap")
fig_corr = plt.figure(figsize=(10, 5))
sns.heatmap(df4.corr(), annot=True, cmap='coolwarm', fmt=".2f")
st.pyplot(fig_corr)

# -------------------- Machine Learning --------------------
st.header(" ML Model Performance")

y = df4['Global_active_power']
x = df4.drop(columns=['Global_active_power', 'Date', 'total_power(kw)', 'day'])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(),
    "Gradient Boosting": GradientBoostingRegressor()
}

# Combine results
all_results = []

for name, model in models.items():
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)

    st.subheader(f"📎 {name}")
    st.write("R² Score:", round(r2, 4))
    st.write("MSE:", round(mse, 4))

    # Create a DataFrame with actual vs predicted
    df_result = pd.DataFrame({
        "Model": name,
        "Actual": y_test.values,
        "Predicted": y_pred,
        "Error": y_test.values - y_pred
    })

    all_results.append(df_result)

# Concatenate all model results
final_result = pd.concat(all_results).reset_index(drop=True)

st.subheader("Test vs Predicted Comparison")
st.dataframe(final_result.head(100))  # Show first 100 rows for brevity

# -------------------- Neural Network --------------------
st.header(" Neural Network (PyTorch)")

x_tensor = tensor(x.values, dtype=float32)
y_tensor = tensor(y.values, dtype=float32).unsqueeze(1)

x_train, x_test, y_train, y_test = train_test_split(x_tensor, y_tensor, test_size=0.25, random_state=42)

class PowerNet(nn.Module):
    def __init__(self, input_dim):
        super(PowerNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        return self.fc4(x)

model = PowerNet(x.shape[1])
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
criterion = nn.MSELoss()

for epoch in range(2000):
    model.train()
    pred = model(x_train)
    loss = criterion(pred, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# 🧪 Evaluate the model
model.eval()
with torch.no_grad():
    output = model(x_test)
    y_pred_nn = output.numpy().flatten()
    y_test_np = y_test.numpy().flatten()

#  Create comparison table
df_nn_result = pd.DataFrame({
    "Model": "Neural Network",
    "Actual": y_test_np,
    "Predicted": y_pred_nn,
    "Error": y_test_np - y_pred_nn
})

# Show results
st.subheader(" Test vs Predicted Comparison (Neural Net)")
st.dataframe(df_nn_result.head(100))

# Show metrics
r2 = r2_score(y_test_np, y_pred_nn)
mse = mean_squared_error(y_test_np, y_pred_nn)
st.subheader(" Neural Network Results")
st.write("R² Score:", round(r2, 4))
st.write("MSE:", round(mse, 4))

