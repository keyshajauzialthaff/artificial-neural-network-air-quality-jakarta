# ============================================================
# 1. PERSIAPAN DATA DAN LIBRARY
# ============================================================

# Load library yang diperlukan
library(dplyr)
library(neuralnet)
library(caret)
library(ggplot2)
library(readxl)

# Baca data dari file Excel
data_kelompok <- read_excel("C:/Data/Matkul Sem 7/Machine Learning/Tugas Kelompok/ANN/Data Kelompok ANN.xlsx")

# Pilih kolom yang digunakan
df <- data_kelompok %>% 
  select(tanggal, pm_duakomalima, sulfur_dioksida, karbon_monoksida, ozon, nitrogen_dioksida)

# Ubah tipe data
df <- df %>%
  mutate(
    tanggal = as.Date(tanggal),
    across(c(pm_duakomalima, sulfur_dioksida, karbon_monoksida, ozon, nitrogen_dioksida), as.numeric)
  )

# Imputasi nilai NA dengan rata-rata kolom
df <- df %>%
  mutate(across(where(is.numeric), ~ ifelse(is.na(.), mean(., na.rm = TRUE), .)))

# Tampilkan summary data
cat("=== SUMMARY DATA ===\n")
summary(df)

# ============================================================
# 2. NORMALISASI DATA (MIN-MAX SCALING)
# ============================================================

# Fungsi normalisasi min-max
normalize <- function(x) { 
  (x - min(x)) / (max(x) - min(x)) 
}

# Terapkan normalisasi
df_norm <- df %>%
  mutate(across(where(is.numeric), normalize))

# Simpan nilai min dan max untuk denormalisasi nanti
mins <- sapply(df[, -1], min)
maxs <- sapply(df[, -1], max)

cat("\n=== NILAI MIN DAN MAX UNTUK DENORMALISASI ===\n")
print("Min values:")
print(mins)
print("Max values:")
print(maxs)

# ============================================================
# 3. PEMBAGIAN DATA TRAINING DAN TESTING
# ============================================================

set.seed(123)
index <- sample(1:nrow(df_norm), round(0.8 * nrow(df_norm)))
train <- df_norm[index, ]
test <- df_norm[-index, ]

cat("\n=== PEMBAGIAN DATA ===\n")
cat("Jumlah data training:", nrow(train), "\n")
cat("Jumlah data testing:", nrow(test), "\n")

# ============================================================
# 4. TUNING HYPERPARAMETER: LEARNING RATE
# ============================================================

# Daftar learning rate yang akan diuji
learning_rates_to_test <- c(0.0005, 0.0001, 0.00005)

# Dataframe untuk menyimpan hasil tuning
hasil_tuning <- data.frame(
  learning_rate = numeric(),
  MSE_denorm = numeric(),
  MAPE_denorm = numeric(),
  R_squared = numeric(),
  Steps = numeric()
)

cat("\n=== MEMULAI PROSES TUNING LEARNING RATE ===\n")

# Loop untuk setiap learning rate
for (lr_value in learning_rates_to_test) {
  
  cat(paste("\n--- Sedang Melatih dengan Learning Rate:", lr_value, "---\n"))
  
  set.seed(123) 
  
  # Pelatihan Neural Network
  nn <- neuralnet(
    pm_duakomalima ~ sulfur_dioksida + karbon_monoksida + ozon + nitrogen_dioksida,
    data = train,
    hidden = c(5, 3),
    algorithm = "backprop",  
    learningrate = lr_value, 
    linear.output = TRUE, 
    threshold = 0.01,
    stepmax = 5e6
  )
  
  # Plot arsitektur neural network
  plot(nn)
  
  # Prediksi pada data testing
  temp_test <- subset(test, select = c("sulfur_dioksida", "karbon_monoksida", "ozon", "nitrogen_dioksida"))
  nn.results <- compute(nn, temp_test)
  
  # Simpan hasil prediksi
  results <- data.frame(
    actual = test$pm_duakomalima,
    prediction = nn.results$net.result
  )
  
  # ============================================================
  # 5. DENORMALISASI DATA
  # ============================================================
  
  actual_denorm <- results$actual * (maxs["pm_duakomalima"] - mins["pm_duakomalima"]) + mins["pm_duakomalima"]
  pred_denorm <- results$prediction * (maxs["pm_duakomalima"] - mins["pm_duakomalima"]) + mins["pm_duakomalima"]
  
  results_denorm <- data.frame(
    Indeks = 1:length(actual_denorm),
    Actual = actual_denorm,
    Prediction = pred_denorm
  )
  
  # ============================================================
  # 6. EVALUASI MODEL (DATA TESTING)
  # ============================================================
  
  MSE <- mean((results_denorm$Actual - results_denorm$Prediction)^2)
  MAPE <- mean(abs(results_denorm$Actual - results_denorm$Prediction) / results_denorm$Actual) * 100
  SS_res <- sum((results_denorm$Actual - results_denorm$Prediction)^2)
  SS_tot <- sum((results_denorm$Actual - mean(results_denorm$Actual))^2)
  R2 <- 1 - (SS_res / SS_tot)
  
  # Tampilkan hasil evaluasi
  cat("   Steps taken:", nn$steps, "\n")
  cat("   MSE (denormalized):", MSE, "\n")
  cat("   MAPE (denormalized):", MAPE, "%\n")
  cat("   R-Squared:", R2, "\n")
  
  # Simpan hasil tuning
  hasil_tuning <- rbind(hasil_tuning, data.frame(
    learning_rate = lr_value,
    MSE_denorm = MSE,
    MAPE_denorm = MAPE,
    R_squared = R2, 
    Steps = nn$result.matrix["steps", 1] 
  ))
}

# ============================================================
# 7. ANALISIS OUTPUT NEURAL NETWORK
# ============================================================

cat("\n=== ANALISIS OUTPUT NEURAL NETWORK ===\n")

# Prediksi pada data training untuk analisis layer output
temp_train <- subset(train, select = c("sulfur_dioksida", "karbon_monoksida", "ozon", "nitrogen_dioksida"))
nn.train.results <- compute(nn, temp_train)

# Output dari berbagai layer
cat("\n--- OUTPUT DARI SETIAP LAYER ---\n")

# Input ke Hidden Layer 1
input_ke_HL1_train <- nn.train.results$neurons[[1]]
cat("Dimensi Input ke Hidden Layer 1:", dim(input_ke_HL1_train), "\n")

# Output Hidden Layer 1
output_HL1_train <- nn.train.results$neurons[[2]]
cat("Dimensi Output Hidden Layer 1:", dim(output_HL1_train), "\n")

# Output Hidden Layer 2
output_HL2_train <- nn.train.results$neurons[[3]]
cat("Dimensi Output Hidden Layer 2:", dim(output_HL2_train), "\n")

# Output Final (Prediksi)
prediksi_train <- nn.train.results$net.result
cat("Dimensi Output Final:", length(prediksi_train), "\n")

# ============================================================
# 8. EVALUASI PADA DATA TRAINING
# ============================================================

results_train <- data.frame(
  actual = train$pm_duakomalima,
  prediction = nn.train.results$net.result
)

# Denormalisasi data training
actual_train_denorm <- results_train$actual * (maxs["pm_duakomalima"] - mins["pm_duakomalima"]) + mins["pm_duakomalima"]
pred_train_denorm <- results_train$prediction * (maxs["pm_duakomalima"] - mins["pm_duakomalima"]) + mins["pm_duakomalima"]

results_train_denorm <- data.frame(
  Actual = actual_train_denorm,
  Prediction = pred_train_denorm
)

# Hitung metrik evaluasi (training)
MSE_train <- mean((results_train_denorm$Actual - results_train_denorm$Prediction)^2)
MAPE_train <- mean(abs(results_train_denorm$Actual - results_train_denorm$Prediction) / results_train_denorm$Actual) * 100
SS_res_train <- sum((results_train_denorm$Actual - results_train_denorm$Prediction)^2)
SS_tot_train <- sum((results_train_denorm$Actual - mean(results_train_denorm$Actual))^2)
R2_train <- 1 - (SS_res_train / SS_tot_train)

cat("\n--- 📚 HASIL EVALUASI DATA TRAINING ---\n")
cat("MSE (denormalized):", MSE_train, "\n")
cat("MAPE (denormalized):", MAPE_train, "%\n")
cat("R-Squared:", R2_train, "\n")

# ============================================================
# 9. HASIL TUNING DAN LEARNING RATE TERBAIK
# ============================================================

cat("\n=== HASIL PERBANDINGAN LEARNING RATE ===\n")
print(hasil_tuning)

# Cari learning rate terbaik (berdasarkan MSE terendah)
lr_terbaik <- hasil_tuning[which.min(hasil_tuning$MSE_denorm), ]

cat("\n--- ⭐ LEARNING RATE TERBAIK DITEMUKAN ---\n")
print(lr_terbaik)

# ============================================================
# 10. EVALUASI FINAL MODEL
# ============================================================

cat("\n=== 📊 EVALUASI FINAL MODEL (DATA TESTING) ===\n")

# Gunakan model dengan learning rate terbaik untuk prediksi final
final_lr <- lr_terbaik$learning_rate
cat("Menggunakan Learning Rate:", final_lr, "untuk evaluasi final\n")

# Hitung metrik evaluasi final
MSE_final <- mean((results_denorm$Actual - results_denorm$Prediction)^2)
MAPE_final <- mean(abs((results_denorm$Actual - results_denorm$Prediction) / results_denorm$Actual)) * 100
SST_final <- sum((results_denorm$Actual - mean(results_denorm$Actual))^2)
SSR_final <- sum((results_denorm$Prediction - mean(results_denorm$Actual))^2)
R2_final <- SSR_final / SST_final

cat("📈 Mean Squared Error (MSE):", MSE_final, "\n")
cat("📉 Mean Absolute Percentage Error (MAPE):", MAPE_final, "%\n")
cat("⭐ Coefficient of Determination (R²):", R2_final, "\n")

# ============================================================
# 11. VISUALISASI HASIL PREDIKSI
# ============================================================

cat("\n=== 🖼️ MEMBUAT VISUALISASI HASIL PREDIKSI ===\n")

# Data untuk plotting
plot_data <- data.frame(
  Indeks = 1:length(results_denorm$Actual),
  Actual = results_denorm$Actual,
  Prediction = results_denorm$Prediction
)

# Buat plot perbandingan aktual vs prediksi
p <- ggplot(plot_data, aes(x = Indeks)) +
  geom_line(aes(y = Actual, color = "Aktual"), size = 1) +
  geom_line(aes(y = Prediction, color = "Prediksi"), linetype = "dashed", size = 1) +
  labs(
    title = "Perbandingan Nilai Aktual vs Prediksi PM2.5",
    subtitle = paste("Learning Rate:", final_lr, "| MSE:", round(MSE_final, 4)),
    x = "Indeks Data Testing",
    y = "Nilai PM2.5 (Denormalisasi)"
  ) +
  scale_color_manual(
    name = "Keterangan", 
    values = c("Aktual" = "blue", "Prediksi" = "red")
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold"),
    plot.subtitle = element_text(hjust = 0.5)
  )

# Tampilkan plot
print(p)

cat("\n=== ✅ PROSES SELESAI ===\n")