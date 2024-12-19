# Cargar librerías
library(readr)
library(MASS) 
library(caret)     
library(ggplot2)

data <- read_csv("/Users/mariajoseespinosa/Documents/ITAM/9. AGO-DIC 2024/ESTADISTICA APLICADA III/PARCIALES/PROEYCTO/cancer.csv")

# Eliminar la primera columna
data <- data[ , -1]

# Convertir Diagnosis a factor (variable objetivo)
data$Diagnosis <- as.factor(data$Diagnosis)

# Dividir en datos de entrenamiento y prueba (70%-30%)
set.seed(123) 
train_index <- createDataPartition(data$Diagnosis, p = 0.7, list = FALSE)
train_data <- data[train_index, ]
test_data <- data[-train_index, ]

# --- Análisis Discriminante Lineal (LDA) ---
lda_model <- lda(Diagnosis ~ ., data = train_data)

# Mostrar el modelo
print(lda_model)

# Predecir en los datos de prueba
lda_predictions <- predict(lda_model, test_data)

# Evaluar la matriz de confusión
confusion_matrix <- confusionMatrix(lda_predictions$class, test_data$Diagnosis)
print(confusion_matrix)

# Visualizar las predicciones de LDA
lda_data <- data.frame(
  LD1 = lda_predictions$x[,1], 
  Diagnosis = test_data$Diagnosis
)

# --- Histogramas de los Scores con límites en el eje X ---
# 1. Histograma para el grupo Benigno (B)
ggplot(subset(lda_data, Diagnosis == "B"), aes(x = LD1, y = ..density..)) +
  geom_histogram(fill = "blue", bins = 30, alpha = 0.6, color = "black") +
  labs(title = "Histograma de los Scores LDA: Benigno",
       x = "Puntajes del Primer Componente LDA (LD1)", y = "Densidad") +
  xlim(-5, 6) +  
  theme_minimal()

# 2. Histograma para el grupo Maligno (M)
ggplot(subset(lda_data, Diagnosis == "M"), aes(x = LD1, y = ..density..)) +
  geom_histogram(fill = "red", bins = 30, alpha = 0.6, color = "black") +
  labs(title = "Histograma de los Scores LDA: Maligno",
       x = "Puntajes del Primer Componente LDA (LD1)", y = "Densidad") +
  xlim(-5, 6) +  
  theme_minimal()

# --- Gráfica de densidad ---
ggplot(lda_data, aes(x = LD1, fill = Diagnosis)) +
  geom_density(alpha = 0.5) +
  labs(title = "Distribución de las proyecciones del LDA",
       x = "Primer Componente de LDA", y = "Densidad") +
  theme_minimal()

# --- Validación cruzada Leave-One-Out ---
cv_lda <- train(
  Diagnosis ~ ., 
  data = train_data, 
  method = "lda", 
  trControl = trainControl(method = "LOOCV")
)

# Mostrar el resultado de la validación cruzada
print(cv_lda)