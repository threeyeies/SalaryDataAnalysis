# ------------NAMMING CONVENTIONS IN R---------
# camel case para variables: soy_una_variable
#-----------------------------------------------

library(caret)
library(dplyr)
library(ggplot2)
library(corrplot)
library(FactoMineR) 
library(factoextra)

# -----------------------------------------------------------------------------
# 2: PREPARACIÓN DE LOS DATOS

#Obteniendo los datos
training_data <- read.csv("Salary_Data.csv")

#Mostrando 
str(training_data)

# Revisando que renglones contienen datos nulos
training_data[!complete.cases(training_data), ]

# Quitando datos nulos 
no_null_dt <- na.omit(training_data)
no_null_dt[!complete.cases(no_null_dt), ]

# Transformando la variable genero categorica a numerica
no_null_dt$Gender <- ifelse(no_null_dt$Gender== "Male", 1, 0)

#---------------------------------------------------------------------------
# 3: Análisis de Componentes Principales (PCA)

# Seleccionar solo las variables numéricas para el PCA
numeric_vars <- no_null_dt %>% select(Age, Salary, Years.of.Experience, Gender)

# Estandarización de las variables
scaled_data <- scale(numeric_vars)

# Realizar PCA
pca_result <- PCA(scaled_data, graph = FALSE)

# Mostrar resumen del PCA (contribución de cada componente)
summary(pca_result)

# Visualización de la varianza explicada por cada componente
fviz_eig(pca_result, addlabels = TRUE, ylim = c(0, 100))

# Visualización de las variables en el plano de los primeros dos componentes principales
fviz_pca_var(pca_result, col.var = "contrib", 
             gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07"), 
             repel = TRUE) # Evita superposición de etiquetas

# Visualización de las observaciones en el plano de los primeros dos componentes principales
fviz_pca_ind(pca_result, col.ind = "cos2", 
             gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07"),
             repel = TRUE) # Evita superposición de etiquetas

# ------------------------------------------------------------------------------
# 4: Exploración y visualización de los datos (EDA)

# Resumen estadístico de las variables de interés
summary(no_null_dt$Salary)
summary(no_null_dt$Years.of.Experience)

# Gráfica de dispersión con línea de tendencia y R cuadrada
ggplot(no_null_dt, aes(x = Years.of.Experience, y = Salary)) +
  geom_point(color = "blue", size = 2) +  # puntos de dispersión
  geom_smooth(method = "lm", se = FALSE, color = "red") +  # línea de tendencia
  labs(title = "Scatter Plot of Salary vs Years of Experience",
       x = "Years of Experience",
       y = "Salary") +
  theme_minimal()

# Ajuste del modelo lineal simple
linear_model <- lm(Salary ~ Years.of.Experience, data = no_null_dt)

# Mostrar el resumen del modelo lineal para ver R²
summary(linear_model)

# Extraer el valor de R² del modelo lineal
r_squared <- summary(linear_model)$r.squared
cat("R² del modelo lineal:", r_squared, "\n")
# -------------------------------------------------------------------------------
# 5: Creación de un Modelo Predictivo
set.seed (6699)

trCtrl = trainControl(method = "cv", number = 10)

#gbm : Modelo lineal generalizado---> regresión

boostFit = train (Salary ~ Age + Gender + Education.Level + Job.Title +
                    Years.of.Experience, trControl = trCtrl,
                  method = "gbm", data = no_null_dt, verbose = FALSE)

warnings()

# Problema con la matriz de confusion, predicción
confusionMatrix (no_null_dt$Salary, predict (boostFit, no_null_dt))


