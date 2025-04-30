#---------------------------------------------------------------------
# OPEN STREET MAPS

#---------------------------------------------------------
# Union bases de datos
#---------------------------------------------------------

# Limpiar environment -----------------------------------------------------
rm(list = ls())
gc() 
cat('\014')
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

# Librerias ---------------------------------------------------------------
require(pacman)
p_load(tidyverse, # Dataframes
       rio, 
       plotly, # Gráficos interactivos
       leaflet, # Mapas interactivos
       units, # unidades
       sf, # Leer/escribir/manipular datos espaciales
       osmdata, # OpenStreetMap (OSM)
       tidymodels, 
       randomForest, # Bosques
       rattle, # Interfaz gráfica para el modelado de datos
       spatialsample,
       xgboost, # Muestreo espacial para modelos de aprendizaje automático
       tmaptools,
       terra,
       geojsonR,
       stringi, # Manipulación de texto 
       tm, # Stop word
       SnowballC, # Reeducir palabras a su raíz (Stemming)
       wordcloud, # Nube de palabras
       RColorBrewer) 

# Directorios -------------------------------------------------------------
data <- paste0(getwd(),'/Data/') # Directorio de base de datos
# views  <- paste0(getwd(),'/views/')  # Directorio para guardar imagenes

# Lectura de datos --------------------------------------------------------
train <- read_csv(paste0(data,'train.csv'))
test  <- read_csv(paste0(data,'test.csv'))

# Etiquetas ---------------------------------------------------------------
test <- test %>%
  mutate(grupo ='test')%>%
  select(grupo, everything()) #(2)

train <- train%>%
  mutate(grupo = 'train')%>%
  select(grupo, everything()) #(1)

# Data completa ----------------------------------------------------
df <- bind_rows(train, test)
write_csv(df, paste0(data,"df.csv"))
rm(train,test)

#---------------------------------------------------------
# LIMPIEZA DE BASES DE DATOS
#---------------------------------------------------------
summary(df)
df %>%
  count(bathrooms, sort = TRUE) # 


# Tratar la variables de texto -------------------------------------------
#------> DESCRIPTION -----------------------------------------------------
df <- df %>%
  mutate(
    description = description %>%
      stri_trans_general("Latin-ASCII") %>%   # eliminar tildes
      tolower() %>%                           # convertir a minúsculas
      str_replace_all("[^a-z0-9]", " ") %>%   # mantener solo letras y números
      removeWords(stopwords("spanish")) %>% # quitar stopwords en español
      str_squish()  # eliminar espacios extra
  )

df <- df %>%
  mutate(numbers_found = str_extract_all(description, "\\d+"))

# Mapa de palabras 
corpus <- Corpus(VectorSource(df$description)) # corpus
corpus <- corpus %>%
  tm_map(content_transformer(tolower)) %>%
  tm_map(removePunctuation) %>%
  tm_map(removeNumbers) %>%
  tm_map(removeWords, stopwords("spanish")) %>%
  tm_map(stripWhitespace) 
tdm <- TermDocumentMatrix(corpus)
m <- as.matrix(tdm)
frecuencia <- sort(rowSums(m), decreasing = TRUE)
palabras <- data.frame(word = names(frecuencia), freq = frecuencia) 

# Nube de palabras
set.seed(123)  # Para reproducibilidad
wordcloud(
  words = palabras$word,
  freq = palabras$freq,
  min.freq = 2,
  max.words = 500,
  random.order = FALSE,
  colors = brewer.pal(8, "Dark2")
)


# Número de habitaciones 
sin_hab <- "habitaciones|cuartos|dormitorios|alcobas|recamaras|cuarto|habitacion|alcoba|recamara"
df <- df %>%
  mutate(n_habitaciones = as.integer(
    sub(paste0(".*?(\\d+)\\s+(", sin_hab, ").*"), "\\1", description)
  ))

df <- df %>%
  mutate(
    palabra_antes = str_extract(description, patron_antes),
    palabra_despues = str_extract(description, patron_despues),
    stem_antes = wordStem(palabra_antes, language = "spanish"),
    stem_despues = wordStem(palabra_despues, language = "spanish")
  )


# Valores faltantes -------------------------------------------------------
# Calcular cantidad y porcentaje de NA por columna
# Primero vemos la cantidad
apply(df, 2, function(x) sum(is.na(x)))
# Ahora el porcentaje
apply(df, 2, function(x) round(sum(is.na(x)/length(x))*100,2))
# Vemos que las variables mas problematicas van a ser <surface_total>, <surface_covered>, <rooms> y <bathrooms>. Por lo que vamos a intentar encontrar maneras para 
# completar los NA.

# Para tratar los valores faltantes de <bathrooms> y <rooms> es importante ver que <bedrooms> si esta completa. Por lo que una buena idea seria imputar la moda
# dependiendo del numero de alcobas que tenga el apto/casa. Sin embargo, se encuentra que cuando bedrooms es 0 rooms es NA, por lo que se busca una estrategía para extraer el número de 
# alcobas o cuartos

#Ya que no se pudo extraer el número de alcobas de todas las descripciones imputaremos por la moda, 
moda <- function(x) {
  return(as.numeric(names(which.max(table(x)))))
}


