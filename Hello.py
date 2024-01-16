# Copyright (c) Streamlit Inc. (2018-2022) Snowflake Inc. (2022)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
import streamlit as st
import yfinance as yf
from prophet import Prophet
import pandas as pd
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Input pengguna untuk simbol saham
simbol = st.sidebar.text_input("Masukkan simbol saham (contoh: BBCA.jk catatan: tambahkan simbol .jk setelah simbol jika ingin menampilkan saham indonesia):", "NVDA")

# Input pengguna untuk tanggal awal dan akhir
tanggal_awal = st.sidebar.date_input("Pilih tanggal awal:", pd.to_datetime('1999-03-01'))
tanggal_akhir = st.sidebar.date_input("Pilih tanggal akhir:", pd.to_datetime('2022-12-31'))

# Input pengguna untuk periode dataframe masa depan (dalam tahun)
periode_masa_depan_tahun = st.sidebar.slider("Pilih periode masa depan (tahun):", 1, 5, 1)

# Konversi tahun menjadi hari
periode_masa_depan_hari = periode_masa_depan_tahun * 365

# Ambil data saham menggunakan Yahoo Finance
data = yf.download(simbol, start=tanggal_awal, end=tanggal_akhir)

# Periksa apakah data tersedia
if data.empty:
    st.error("Tidak ada data yang tersedia untuk simbol saham dan rentang tanggal yang dipilih. Harap sesuaikan masukan.")
else:
    # Anggap 'data' adalah DataFrame dengan kolom 'Date' dan 'Close'
    df = data.reset_index()[['Date', 'Close']]
    df.columns = ['ds', 'y']

    # Konversi kolom 'ds' ke format datetime
    df['ds'] = pd.to_datetime(df['ds'])

    # Urutkan DataFrame berdasarkan kolom 'ds'
    df = df.sort_values(by='ds')

    # Buat dan latih model Prophet
    m = Prophet().fit(df)

    # Buat dataframe masa depan untuk prediksi
    masa_depan = m.make_future_dataframe(periods=periode_masa_depan_hari)

    # Prediksi nilai
    prediksi = m.predict(masa_depan)

    # Hitung metrik akurasi
    df_eval = pd.merge(df, prediksi[['ds', 'yhat']], on='ds', how='inner')
    mae = mean_absolute_error(df_eval['y'], df_eval['yhat'])
    mse = mean_squared_error(df_eval['y'], df_eval['yhat'])
    rmse = mean_squared_error(df_eval['y'], df_eval['yhat'], squared=False)
    mape = (abs(df_eval['y'] - df_eval['yhat']) / df_eval['y']).mean() * 100
    r_squared = r2_score(df_eval['y'], df_eval['yhat'])

    # Buat aplikasi Streamlit dan tampilkan data
    st.title('Aplikasi Prediksi Saham')

    st.subheader('Data Saham')
    nama_perusahaan = yf.Ticker(simbol).info['longName']
    st.write("Nama Perusahaan:", nama_perusahaan)

    # Tampilkan data sebelumnya berdasarkan tanggal awal dan akhir yang dipilih
    st.write("Rentang Data Terpilih:")
    st.write(data)

    st.subheader('Data Prediksi')

    # Plot data yang diprediksi dan data aktual menggunakan plotly
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=prediksi['ds'], y=prediksi['yhat'], mode='lines', name='prediksi'))
    fig.add_trace(go.Scatter(x=df['ds'], y=df['y'], mode='lines', name='aktual'))

    fig.update_layout(
        title='Data Prediksi vs Aktual',
        xaxis_title='Tanggal',
        yaxis_title='Harga Saham',
        legend_title='Jenis Data',
    )

    st.plotly_chart(fig)

    st.subheader('Metrik Akurasi')

    # Tampilkan metrik akurasi dengan penjelasan
    st.write('Mean Absolute Error (MAE): {:.2f}'.format(mae))
    st.write('Mean Squared Error (MSE): {:.2f}'.format(mse))
    st.write('Root Mean Squared Error (RMSE): {:.2f}'.format(rmse))
    st.write('Mean Absolute Percentage Error (MAPE): {:.2f}%'.format(mape))
    st.write('R-squared (R2): {:.2f}'.format(r_squared))
