import streamlit as st
import joblib
import numpy as np
import pandas as pd

st.set_page_config(layout="wide")
model = joblib.load("mobil.sav")

st.title("Aplikasi Prediksi Harga Mobil")
st.divider()

df = pd.read_csv(
    r"C:\Users\shinji\.cache\kagglehub\datasets\sukhmandeepsinghbrar\car-price-prediction-dataset\versions\1\cardekho.csv"
)
df["year"] = df["year"].astype(str)
df = df.dropna(how="any")

fuel_mapping = {
    "Diesel": 1,
    "Petrol": 3,
    "LPG": 2,
    "CNG": 0,
}
seller_type_mapping = {
    "Individual": 1,
    "Dealer": 0,
    "Trustmark Dealer": 2,
}
transmission_mapping = {
    "Manual": 1,
    "Automatic": 0,
}
owner_mapping = {
    "First Owner": 0,
    "Second Owner": 2,
    "Third Owner": 4,
    "Fourth & Above Owner": 1,
    "Test Drive Car": 3,
}


with st.sidebar:
    st.sidebar.header("Opsi Kriteria")
    year = st.selectbox("Tahun Mobil", list(range(2000, 2021)))
    km_driven = st.slider(
        "Jarak Tempuh", min_value=0, max_value=df["km_driven"].max(), step=1000
    )
    fuel = st.selectbox("Jenis Bahan Bakar", list(fuel_mapping.keys()))
    seller_type = st.selectbox("Tipe Penjual", list(df["seller_type"].unique()))
    transmission = st.selectbox("Transmisi", list(df["transmission"].unique()))
    owner = st.selectbox("Jenis Kepemilik", list(df["owner"].unique()))
    millage = st.slider("Konsumsi Bahan Bakar", min_value=0, max_value=50)
    engine = st.slider("Kapasitas Mesin", min_value=0, max_value=3500)
    max_power = st.slider("Tenaga Maksimum", min_value=0, max_value=300)
    seats = st.slider("Jumlah Kursi", min_value=0, max_value=12)


fuel_num = fuel_mapping[fuel]
seller_type_num = seller_type_mapping[seller_type]
transmission_num = transmission_mapping[transmission]
owner_num = owner_mapping[owner]

input_data = np.array(
    [
        [
            year,
            km_driven,
            fuel_num,
            seller_type_num,
            transmission_num,
            owner_num,
            millage,
            engine,
            max_power,
            seats,
        ]
    ]
)
prediction = model.predict(input_data)
prediction_idr = prediction[0] * 187
new = [
    "name",
    "year",
    "km_driven",
    "fuel",
    "seller_type",
    "transmission",
    "owner",
    "mileage(km/ltr/kg)",
    "engine",
    "max_power",
    "seats",
    "selling_price (INR)",
    "selling_price (IDR)",
]

if prediction[0] >= 0:
    st.write("### Prediksi Harga Mobil:")
    st.write(f"INR: Rs. {prediction[0]:,.0f}")
    st.write(f"IDR: Rp. {prediction_idr:,.0f}")

    diff = abs(df["selling_price"] - prediction[0])

    car_name = df.iloc[(df["selling_price"] - prediction[0]).abs().argsort()[:10]]
    car_name = car_name.rename(columns={"selling_price": "selling_price (INR)"})
    car_name["selling_price (IDR)"] = car_name["selling_price (INR)"] * 187
    car_name = car_name.reindex(columns=new)
    st.write("### Mobil yang cocok diharga tersebut:")
    st.write(car_name.sort_values("year", ascending=False).reset_index(drop=True))
else:
    st.error(
        "Mohon maaf, tidak ada mobil yang cocok dengan kriteria yang anda masukan. Silahkan coba lagi.",
        icon=":material/warning:",
    )
