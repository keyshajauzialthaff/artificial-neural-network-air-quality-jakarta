# artificial-neural-network-air-quality-jakarta

# Prediksi Kualitas Udara PM2.5 DKI Jakarta Menggunakan Artificial Neural Network (ANN)

Repository ini berisi implementasi **Artificial Neural Network (ANN)** dengan algoritma **Backpropagation** untuk memprediksi konsentrasi polutan udara **PM2.5** harian di Stasiun Bundaran HI, DKI Jakarta.

---

## Tentang Proyek
Polusi udara merupakan masalah krusial di perkotaan besar seperti DKI Jakarta. Penelitian ini membangun model prediktif non-linear menggunakan parameter gas polutan pendukung untuk memperkirakan kadar PM2.5.

- **Dataset:** Data harian Stasiun Bundaran HI (Periode Mei - November 2023, 214 observasi)
- **Variabel Input ($X$):** 
  - Sulfur Dioksida ($\text{SO}_2$)
  - Karbon Monoksida ($\text{CO}$)
  - Ozon ($\text{O}_3$)
  - Nitrogen Dioksida ($\text{NO}_2$)
- **Variabel Target ($Y$):** Partikulat Halus ($\text{PM}_{2.5}$)

---

## Arsitektur Model & Metodologi
1. **Pre-processing:** 
   - Penanganan *missing value* menggunakan *mean imputation*.
   - Normalisasi data menggunakan **Min-Max Scaling** (skala 0 s.d. 1).
2. **Pembagian Data:** 
   - *Training Data*: 80% (171 observasi)
   - *Testing Data*: 20% (43 observasi)
3. **Arsitektur ANN:** 
   - **Input Layer:** 4 neuron
   - **Hidden Layer:** 2 Layer (Layer 1: 5 neuron, Layer 2: 3 neuron / Arsitektur 4-5-3-1)
   - **Output Layer:** 1 neuron
   - **Fungsi Aktivasi:** Logistik Sigmoid (Hidden Layer) & Linear (Output Layer)
   - **Learning Rate Terbaik:** $0.00005$

---

## Hasil Evaluasi Model (Data Testing)
Berdasarkan hasil pengujian pada data testing, diperoleh performa model sebagai berikut:
- **MAPE (Mean Absolute Percentage Error):** $11.14\%$
- **MSE (Mean Squared Error):** $124.46$
- **Koefisien Determinasi ($R^2$):** $0.2609$ ($26.09\%$ variasi data mampu dijelaskan oleh model)

---

## Library R
Pastikan Anda telah menginstal software **R** atau **RStudio** beserta library berikut:
```R
install.packages("dplyr")
install.packages("neuralnet")
install.packages("caret")
install.packages("ggplot2")
install.packages("readxl")
