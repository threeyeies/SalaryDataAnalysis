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


# ------------------------------------------------------------------------------
# 3: Exploración y visualización de los datos
summary (no_null_dt$Age)
boxplot (Age ~ Salary, data = no_null_dt,
         main = "Salary based on the Age of an individual",
         xlab = "Salary", ylab = "Age", col = "salmon")





