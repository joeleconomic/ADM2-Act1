#LIBRERÍAS
library(readr)
library(dplyr)
library(caret)
library(rpart)
library(rpart.plot)

# CARGA DE DATOS 
# Cargamos la base de datos y visualizamos su estructura inicial
datos <- read_csv("titanic.csv")
View(datos)

# 1.ANÁLISIS EXPLORATORIO ----

# Estructura del dataset
str(datos)

# Resumen estadístico general
summary(datos)


# -------------------------
# VALORES NULOS
# -------------------------

colSums(is.na(datos))
# El dataset presenta valores nulos en las variables Age, Cabin y Embarked

# Porcentaje de nulos
round(colSums(is.na(datos)) / nrow(datos) * 100, 2)
# Destacan Cabin con un 77.10% de nulos y Age con un 19.87%. Embarked presenta un bajo porcentaje (0.22%)

# -------------------------
# VARIABLES CATEGÓRICAS
# -------------------------

# Supervivencia
table(datos$Survived)
prop.table(table(datos$Survived))

# Género
table(datos$Sex)
prop.table(table(datos$Sex))

# Clase
table(datos$Pclass)
prop.table(table(datos$Pclass))

# Puerto de embarque
table(datos$Embarked)
prop.table(table(datos$Embarked))

# -------------------------
# VARIABLES NUMÉRICAS
# -------------------------

# Edad
summary(datos$Age)

# Tarifa
summary(datos$Fare)

# Familiares
summary(datos$SibSp)
summary(datos$Parch)

# -------------------------
# ANÁLISIS CRUZADO
# -------------------------

# Supervivencia por género
table(datos$Survived, datos$Sex)
prop.table(table(datos$Survived, datos$Sex), margin = 2)

# Supervivencia por clase
table(datos$Survived, datos$Pclass)
prop.table(table(datos$Survived, datos$Pclass), margin = 2)

# Supervivencia por puerto
table(datos$Survived, datos$Embarked)
prop.table(table(datos$Survived, datos$Embarked), margin = 2)

# -------------------------
# VISUALIZACIONES
# -------------------------

# Histograma de edades
hist(datos$Age, main = "Distribución de Edad", xlab = "Edad")

# Histograma de tarifas
hist(datos$Fare, main = "Distribución de Tarifas", xlab = "Fare")

# Boxplot edad vs supervivencia
boxplot(Age ~ Survived, data = datos,
        main = "Edad vs Supervivencia",
        xlab = "Sobrevivió (0=No, 1=Sí)",
        ylab = "Edad")

# Boxplot tarifa vs supervivencia
boxplot(Fare ~ Survived, data = datos,
        main = "Tarifa vs Supervivencia",
        xlab = "Sobrevivió (0=No, 1=Sí)",
        ylab = "Fare")


#2. MODELOS DE CLASIFICACIÓN 
# 2.1 LIMPIEZA Y PREPROCESAMIENTO DE LOS DATOS

# ARGUMENTACIÓN DEL PREPROCESAMIENTO:
# Excluimos variables sin capacidad predictiva, ya que son identificadores (PassengerId, Name, Ticket).
# Eliminamos la variable Cabin por su elevada proporción de nulos, evitando sesgos en el análisis.
# Para la variable Age, reemplazamos los valores nulos por la media de la edad redondeada.
# Tranformamos en binaria la variable género y convertimos a factor la clase y el puerto.

datos_limpios <- datos %>%
  # Paso 1: Reemplazamos los nulos de la variable Age por la media.
  mutate(Age = ifelse(is.na(Age), mean(Age, na.rm = TRUE), Age),
         Age = round(Age)) %>%
  
  # Paso 2: Eliminación de identificadores y variables con exceso de nulos.
  select(-Cabin, -PassengerId, -Name, -Ticket) %>%
  
  # Paso 3: Eliminación de las filas residuales con nulos (Embarked)
  na.omit() %>%
  
  # Paso 4: Transformación de variables.
  mutate(
    Sex_numerico = ifelse(Sex == "male", 1, 0),             
    Embarked_factor = as.factor(Embarked),                  
    Pclass_factor = as.factor(Pclass)                     
  )

View(datos_limpios)

# 2.2 PARTICIÓN DE LOS DATOS EN ENTRENAMIENTO Y TESTEO

# Seleccionamos exclusivamente las columnas predictoras definitivas.
datos_regresion <- datos_limpios %>% 
  select(Survived, Pclass_factor, Sex_numerico, Age, SibSp, Parch, Fare, Embarked_factor)
#Para la creación de estos modelos predictivos, hemos seleccionado variables que representan las características más relevantes, tales como el género, la edad, la clase del billete, el número de familiares a bordo, la tarifa pagada y el puerto de embarque. 
#La elección de estos factores nos permite comprobar con datos reales si normas históricas como "las mujeres y los niños primero" o la diferencia de clases sociales tuvieron un impacto directo y cuantificable en las posibilidades de sobrevivir.
#No se han tenido en cuenta el resto de variables debido a que eran identificadores, los cuáles  no tienen capacidad predictiva.

# Partición de la muestra (80% Entrenamiento, 20% Testeo).
set.seed(123)
indice <- sample(1:nrow(datos_regresion), size = round(0.8 * nrow(datos_regresion))) 

train <- datos_regresion[indice, ]  
test  <- datos_regresion[-indice, ] 

# 2.3. MODELADO ESTADÍSTICO Y EXTRACCIÓN DE PATRONES

# A) REGRESIÓN LOGÍSTICA

modelo <- glm(Survived ~ Pclass_factor + Sex_numerico + Age + SibSp + Parch + Fare + Embarked_factor, 
              data = train, family = "binomial")

summary(modelo)

# B) ÁRBOL DE DECISIÓN

modeloarbol <- rpart(Survived ~ Pclass_factor + Sex_numerico + Age + SibSp + Parch + Fare + Embarked_factor, 
                     data = train, method = "class")

# Visualización gráfica 
rpart.plot(modeloarbol)

# 3. VALIDACIÓN Y RENDIMIENTO DE LOS MODELOS 

# Evaluación Logit
prob_pred_logit <- predict(modelo, newdata = test, type = "response")
clase_pred_logit <- ifelse(prob_pred_logit > 0.5, 1, 0)
pred_clase_factor_logit <- factor(clase_pred_logit, levels = c(0, 1)) 

cat("\n--- MATRIZ DE CONFUSIÓN: REGRESIÓN LOGÍSTICA ---\n")
# CORRECCIÓN 2: Convertimos test$Survived a factor directamente aquí dentro
confusionMatrix(pred_clase_factor_logit, factor(test$Survived, levels = c(0, 1)), positive = "1")
