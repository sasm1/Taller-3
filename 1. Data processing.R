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
apply(df, 2, function(x) sum(is.na(x)))
apply(df, 2, function(x) round(sum(is.na(x)/length(x))*100,2))

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

# Mapear números
numeros <- c("cero" = "0", "uno" = "1","dos" = "2","tres" = "3","cuatro" = "4",
             "cinco" = "5","seis" = "6","siete" = "7","ocho" = "8","nueve" = "9")
df <- df %>%
  mutate(description = str_replace_all(description, numeros))

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
  max.words = 100,
  random.order = FALSE,
  colors = brewer.pal(8, "Dark2")
); rm(m, tdm, frecuencia, palabras)

write.csv(palabras, "Palabras.csv")
#-------------------------------------------------------------------------------
# REEMPLAZAR NAs o 0s
# ---------------------------  -------------------------------------------------

# -FUNCIONES--------------------------------------------------------------------
extraer_contexto <- function(texto) {
  palabras <- str_split(texto, "\\s+")[[1]]
  indices <- which(palabras %in% claves)
  antes <- if (any(indices > 1)) palabras[pmax(indices - 1, 1)] else character(0)
  despues <- if (any(indices < length(palabras))) palabras[pmin(indices + 1, length(palabras))] else character(0)
  list(
    palabra_antes = paste(unique(antes), collapse = ", "),
    palabra_despues = paste(unique(despues), collapse = ", ")
  )
}

# --> Habitaciones
claves <- c("habitaciones", "alcobas", "alcoba", "habitacion")
df <- df %>% mutate(contexto = map(description, extraer_contexto)) %>%
  unnest_wider(contexto)

df <- df %>%
  mutate(
    palabra_antes_num = suppressWarnings(as.numeric(palabra_antes)), 
    bedrooms = if_else(
      bedrooms == 0 & !is.na(palabra_antes_num),
      palabra_antes_num,
      bedrooms
    )
  ) %>%
  select(-palabra_antes_num, -palabra_antes, -palabra_despues)

# ---> Baños (completar NAs)
claves <- c("bano", "bao", "baos", "banos")
df <- df %>% mutate(contexto = map(description, extraer_contexto)) %>%
  unnest_wider(contexto)

df <- df %>%
  mutate(
    palabra_antes_num = suppressWarnings(as.numeric(palabra_antes)), 
    bathrooms = if_else(
      is.na(bathrooms) & !is.na(palabra_antes_num),
      palabra_antes_num, #Reemplazamos por numéricos
      bathrooms
    )
  ) %>%
  select(-palabra_antes_num, -palabra_antes, -palabra_despues)  #Logramos cubrir 3282

# ---> En el resto de observaciones vamos a imputar un baño por cada cuarto y vamos a 
# corregir lo que quedó mal con análisis de texto
df <- df %>% mutate(bathrooms = if_else(is.na(bathrooms) | bathrooms > 11, bedrooms, bathrooms))

# ---- Parqueadero o depósitos
df <- df %>%
  mutate(parqueadero = as.numeric(grepl("parqueadero|garaje|deposito|parqueaderos|garajes", description)))

# ---- Terrazas o Balcones
df <- df %>%
  mutate(terraza = as.numeric(grepl("terraza|balcon|patio|balcn", description, ignore.case = TRUE)))

# ---- Amenidades de Lujo (Walk-in Closet, Lavandería, Patio de ropas, cocina abierta)


# ---- Seguridad 
  
# ---- Deposito o Garaje
  
# ---- 
