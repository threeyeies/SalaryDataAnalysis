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

#Obteniendo los datos de entrenamiento
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
str(no_null_dt)







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


#----------------------------------------------------------------------
# 5 Generación del modelo de regresion lineañ

# Creacion  del modelo lineal simple
linear_model <- lm(Salary ~ Years.of.Experience, data = no_null_dt)

# Mostrar el resumen del modelo lineal para ver R²
summary(linear_model)

# Extraer el valor de R² del modelo lineal
r_squared <- summary(linear_model)$r.squared
cat("R² del modelo lineal:", r_squared, "\n")


# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------
# 5: Probando el modelo

# Datos de entrada para realizar predicciones (nuevo empleado)
nuevo_empleado <- data.frame(Years.of.Experience = c(5, 10, 15), Gender = c(1, 0, 1), Age = c(30, 45, 35))

# 1. Predecir los salarios para los nuevos empleados basados en sus años de experiencia
predicciones <- predict(linear_model, nuevo_empleado)

# 2. Mostrar los resultados de las predicciones
print("Predicciones de salarios para nuevos empleados:")
print(predicciones)



#--------------------------------------------------------------------------------
#6: Creación de un Modelo Predictivo (un modelo efectivo)

# Estableciendo la semilla = numero de registros tratados (sin nulos, etc)
set.seed(6699)

# Control del entrenamiento: validación cruzada con 10 iteraciones
trCtrl <- trainControl(method = "cv", number = 10)

# Modelo GBM (Gradient Boosting Machine) para predicción de Salary  (ESPERAR A QUE TERMINE!!!)
gbm_model <- train(Salary ~ Age + Gender + Education.Level + Job.Title +
                     Years.of.Experience, trControl = trCtrl,
                   method = "gbm", data = no_null_dt, verbose = FALSE)

# Mostrar las advertencias que ocurriesen durante la generación del modelo GBM
warnings()

# Mostrar el resumen del modelo 
print(gbm_model)

# Mostrar los resultados de la validación cruzada
print(gbm_model$results)

# Mostrar métricas de cada fold en la validación cruzada
print(gbm_model$resample)

# Mostrar la precisión del modelo a lo largo de las iteraciones de la validación cruzada
plot(gbm_model)

# Realizar predicciones sobre los datos de entrenamiento 
salary_predictions <- predict(gbm_model, no_null_dt)

no_null_dtAdapt <- no_null_dt # Copy a nueva variable de los datos de entrenamiento para hacer la matriz de correlación
no_null_dtAdapt$predicted = predict (gbm_model, no_null_dtAdapt) # Predecir y agregar columna a la nueva variable anterior

# Mostrar las primeras predicciones y los valores reales
head(data.frame(Real = no_null_dt$Salary, Predicted = salary_predictions))



# Tabla de contingencia que muestra la relación entre los valores reales de 'Salary' y los valores predichos.
# Esto permite ver cuántas predicciones coinciden con los valores reales.
table(no_null_dtAdapt$Salary, no_null_dtAdapt$predicted)

# Crear un data frame llamado 'actuals_preds_train' que contiene dos columnas: actuals y predict
# Se utiliza 'cbind()' para combinar estas dos columnas.
actuals_preds_train <- data.frame(cbind(actuals=no_null_dtAdapt$Salary, predicted=no_null_dtAdapt$predicted))

# Mostrar las primeras filas del data frame 'actuals_preds' para tener una vista rápida de los datos reales y predichos
head(actuals_preds_train) 

# Matriz de correlación entre las columnas 'actuals' y 'predicted' 
# Muestra como se relacionan los valores reales y los valores predichos.
correlation_accuracy_train <- cor(actuals_preds_train)


print(correlation_accuracy_train) # Imprimir la matriz de correlación
View(correlation_accuracy_train) #Abrir la variable desde environment que contiene a la matriz de correlación (opcional)





















#confusionMatrix (no_null_dt$Salary, predict (gbm_model, no_null_dt))
# ------------------------------------------------------------------------------
# INCOMPATIBILIDAD CON LA MATEIZ DE CONFUSIÓN: GMB predice valores continuos
# ------Por lo tanto no es posible entregar una matriz de validación cruzada--------
# Debido a que  en modelos de regresión, como el modelo GBM que se está...
# utilizando para predecir el salario, NO SE PUEDE UTILIZAR una.. 
# matriz de confusión directamente, ya que esta se emplea en...
# modelos de clasificación (donde se predicen clases discretas, no valores continuos).
# Sin embargo, hay varias métricas y métodos análogos para evaluar...
# la calidad de un modelo de regresión que desempeñan un papel similar...
# al de la matriz de confusión en clasificación. 
# Algunas de las más comunes son:
#----------------------------------------------------------------------------------

# Curva de Predicción Real (Real vs. Predicted Plot):
# Gráfico que compara los valores predichos con los reales, 
# lo que permite visualizar qué tan bien el modelo ajusta los datos
plot(no_null_dt$Salary, salary_predictions,
     main="Real vs Predicted",
     xlab="Real Salary", ylab="Predicted Salary",
     pch=19, col="blue")
abline(0,1,col="red",lwd=2) # Línea de referencia ideal



# Gráfico de Residuales
# Un gráfico de los errores residuales (la diferencia entre los valores reales y los predichos) 
# permite verificar si los residuos están distribuidos de manera uniforme, lo cual es un indicador de un buen ajuste.
residuals <- no_null_dt$Salary - salary_predictions
plot(salary_predictions, residuals,
     main="Residuals vs Predicted",
     xlab="Predicted Salary", ylab="Residuals",
     pch=19, col="purple")
abline(h=0, col="red", lwd=2)


# Distribución de Errores
# Con la distribución de los errores residuales se asegura que... 
# siguen una distribución aproximadamente normal, 
# lo que es un buen indicativo de que el modelo no está sesgado.
hist(residuals, breaks=20, col="grey", main="Distribution of Residuals")


# Cálculo de RMSE (Root Mean Squared Error)
rmse_value <- sqrt(mean((no_null_dt$Salary - salary_predictions)^2))
cat("RMSE del modelo: ", rmse_value, "\n")

# Cálculo de R² para ver la proporción de varianza explicada por el modelo
r_squared <- cor(no_null_dt$Salary, salary_predictions)^2
cat("R² del modelo: ", r_squared, "\n")


#------------------------------------------------------------------------------
# 7: Carga y Evaluación del conjunto de datos de prueba

# Obteniendo datos de prueba (Testing)
testing_data <- read.csv("Salary_Testing_Data.csv")

# Visualizando
str(testing_data)

# Revisando que renglones contienen datos nulos
testing_data[!complete.cases(testing_data), ]

no_null_dtt <- testing_data

# Quitando datos nulos 
no_null_dtt <- na.omit(testing_data)
no_null_dtt[!complete.cases(no_null_dtt), ]

# Transformando la variable genero categorica a numerica
no_null_dtt$Gender <- ifelse(no_null_dtt$Gender== "Male", 1, 0)
str(no_null_dtt)


#------------------------------------------------------------------------------
# 8: Validando el modelo con datos de prueba (Testing)

# Prediciendo con el modelo predictivo GBM con los datos de prueba
no_null_dtt$predicted = predict (gbm_model, no_null_dtt)

# Tabla de contingencia que muestra la relación entre los valores reales de 'Salary' y los valores predichos.
# Esto permite ver cuántas predicciones coinciden con los valores reales.
table(no_null_dtt$Salary, no_null_dtt$predicted)

# Crear un data frame llamado 'actuals_preds' que contiene dos columnas:
# 1. 'actuals': los valores reales de salario provenientes del conjunto de datos 'no_null_dtt$Salary'.
# 2. 'predicted': los valores predichos por el modelo almacenados en 'no_null_dtt$predicted'.
# Se utiliza 'cbind()' para combinar estas dos columnas.
actuals_preds <- data.frame(cbind(actuals=no_null_dtt$Salary, predicted=no_null_dtt$predicted))

# Mostrar las primeras filas del data frame 'actuals_preds' para tener una vista rápida de los datos reales y predichos
head(actuals_preds) 

# Matriz de correlación entre las columnas 'actuals' y 'predicted' 
# Muestra como se relacionan los valores reales y los valores predichos.
correlation_accuracy <- cor(actuals_preds)


print(correlation_accuracy) # Imprimir la matriz de correlación
View(correlation_accuracy) #Abrir la variable desde environment que contiene a la matriz de correlación (opcional)

