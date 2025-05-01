
################################################################################
#               CREACIÓN DE BASE DE DATOS PARA MODELOS (HOT)                   #
################################################################################

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

################################################################################
# LIMPIEZA DE BASES DE DATOS
################################################################################
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
)

#write.csv(palabras, "Palabras.csv"); 
rm(m, tdm, frecuencia, palabras)
################################################################################
# REEMPLAZAR NAs o 0s
################################################################################

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

# --> Habitaciones (Corregir 0 Habitaciones)
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
  ) %>% # Reemplazar 0 habitaciones por # habitaciones descripción (180 0 a 112)
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

# ------> Para llenar Roooms (pues creamos las rooms literal, todo lo que no sea
# pa mimir o para hacer del 1 o del 2 xD)

# ---- Cocina
claves <- c("cocina")
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

# ---- Cocina lujosa
df <- df %>%
  mutate(cocina_lujo = as.numeric(grepl("cocina abierta|cocina amplia", description)))

# ---- Cocina (resto)
df <- df %>%
  mutate(cocina_estandar = as.numeric(grepl("cocina", description, ignore.case = TRUE) &
                                        !grepl("cocina abierta|cocina amplia", description, ignore.case = TRUE)))
# ---- Parqueadero o depósitos
df <- df %>%
  mutate(parqueadero = as.numeric(grepl("parqueadero|garaje|deposito|parqueaderos|garajes|depositos", description)))

# ---- Terrazas o Balcones o jardines o patios interiores
df <- df %>%
  mutate(terraza = as.numeric(grepl("terraza|balcon|patio|balcn|jardin|jardn", description)))

# ---- Sala/ Comedor 
df <- df %>%
  mutate(sala_comedor = as.numeric(grepl("sala|salacomedor|comedor|sala comedor", description)))

# ---- Patio de ropas o lavandería
df <- df %>%
  mutate(patio_lavanderia = as.numeric(grepl("lavandera|balcon|patio ropas|balcn|patio interior", description)))

# ---- Walk-in Closets (Lujo)
df <- df %>%
  mutate(walkin_closet = as.numeric(grepl("walkin|walk-in|walk-in closet", description)))

# ---- Estudio
df <- df %>% mutate(
  estudio = as.numeric(grepl("estudio|estudo|studio", description, ignore.case = TRUE) &
        !grepl("aparta\\s*estudio|apartaestudio", description, ignore.case = TRUE)))

# Entonces ahora sí, completamos rooms
df$rooms <- ifelse(is.na(df$rooms) | df$rooms == "", 
  df$estudio + df$walkin_closet + df$patio_lavanderia + df$sala_comedor + 
    df$terraza + df$parqueadero + df$cocina_estandar + df$cocina_lujo, 
  df$rooms
)

df <- df %>% # En dónde quedó sero vamos a poner la mediana por tipo de propiedad
  group_by(property_type) %>%
  mutate(rooms = ifelse(rooms == 0,median(rooms[rooms > 0], na.rm = TRUE), 
      rooms )) %>%ungroup()

# ---------> Llenar metros cuadrados 
df <- df %>%
  mutate(
    surface = case_when(
      !is.na(surface_covered) ~ as.character(surface_covered),
      TRUE ~ str_extract(
        description,
        "\\d+\\s*m2|\\d+\\s*mt|\\d+\\s*m|\\d+\\s*metros cuadrados|\\d+\\s*metros|\\d+m2|\\d+mt|\\d+m"
      )
    ),
    surface = str_extract(surface, "\\d+"),
    surface = as.numeric(surface)
  ) %>%
  group_by(bathrooms, bedrooms) %>%
  mutate(
    surface_imputado = case_when(
      is.na(surface) | surface < 30 ~ mean(surface[surface >= 30 & surface <= 50], na.rm = TRUE),
      surface > 1000 ~ 350, # Por histograma 
      TRUE ~ surface
    )
  ) %>%
  ungroup()

# Para los datos faltantes que nos quedaron lo hacemos por: 
df <- df %>%
  group_by(property_type, rooms, bedrooms) %>% # rooms, bedrooms y tipo de propiedad (son casas)
  mutate(surface_imputado = ifelse(is.na(surface_imputado), mean(surface_imputado, na.rm = TRUE), surface_imputado)) %>%
  ungroup() %>% # surface_total
  mutate(surface_imputado = ifelse(is.na(surface_imputado) & !is.na(surface_total), surface_total, surface_imputado)) %>%
  group_by(property_type) %>% #media
  mutate(surface_imputado = ifelse(is.na(surface_imputado), mean(surface_imputado, na.rm = TRUE), surface_imputado)) %>%
  ungroup()

################################################################################
# CREAR NUEVAS VARIABLES :) 
################################################################################
# VALOR X AMENIDADES------------------------------------------------------------
# ----- Closet 
df <- df %>% mutate(
  closet = as.numeric(grepl("closet", description, ignore.case = TRUE) &
                         !grepl("walk-in closet", description, ignore.case = TRUE)))

# ---- Amenidades del Conjunto (normales)
claves <- c("salon comunal","salon social","saln","recepcion","zonas comunes", "lobby",
            "visitantes","hall")


# variable Ascensor 
bd <- bd %>%
  mutate(ascensor = as.numeric(grepl("ascensor", description)))

# Variable vigilancia
bd <- bd %>%
  mutate(vigilancia = as.numeric(grepl("seguridad|vigilancia|porteria", description)))


# ---- Seguridad 
df <- df %>%
  mutate(vigilancia = as.numeric(grepl("seguridad|vigilancia|porteria", description)))
  
# ---- Piso
df <- df %>%
  mutate(
    piso_info = str_extract(piso_info, "(\\b\\w+\\b)?\\s*piso\\s*(\\b\\w+\\b)?")
  ) %>%
  mutate(
    piso_texto = case_when(
      str_detect(piso_info, "primer|uno|1er") ~ "1",
      str_detect(piso_info, "segundo|dos|2do") ~ "2",
      str_detect(piso_info, "tercer|tres|3er") ~ "3",
      str_detect(piso_info, "cuarto|cuatro|4to|4o") ~ "4",
      str_detect(piso_info, "quinto|cinco|5to|5o") ~ "5",
      str_detect(piso_info, "sexto|seis|6to|6o") ~ "6",
      str_detect(piso_info, "séptimo|septimo|siete|7mo|7o") ~ "7",
      str_detect(piso_info, "octavo|ocho|8vo|8o") ~ "8",
      str_detect(piso_info, "noveno|nueve|9no|9o") ~ "9",
      str_detect(piso_info, "décimo|decimo|diez|10mo|10o") ~ "10",
      str_detect(piso_info, "once|11avo|11abo") ~ "11",
      str_detect(piso_info, "doce|12avo|12abo") ~ "12",
      str_detect(piso_info, "trece|13avo|13abo") ~ "13",
      str_detect(piso_info, "catorce|14avo|14abo") ~ "14",
      str_detect(piso_info, "quince|15avo|15abo") ~ "15",
      str_detect(piso_info, "dieciséis|dieciseis|16avo|16abo") ~ "16",
      str_detect(piso_info, "diecisiete|17avo|17abo") ~ "17",
      str_detect(piso_info, "dieciocho|18avo|18abo") ~ "18",
      str_detect(piso_info, "diecinueve|19avo|19abo") ~ "19",
      str_detect(piso_info, "veinte|20avo|20abo") ~ "20",
      TRUE ~ str_extract(piso_info, "\\d+")
    )
  ) %>%
  mutate(
    piso_numerico = as.integer(piso_texto),
    piso_numerico = if_else(piso_numerico > 30, NA_integer_, piso_numerico)
  )  # Asumimos hasta piso 30 

df<- df %>%  # Llenamos con la mediana si no encuentra valor
  group_by(property_type) %>%
  mutate(
    piso = if_else(is.na(piso_numerico), median(piso_numerico, na.rm = TRUE), piso_numerico)
  ) %>%
  ungroup() %>% 
  select(-piso_texto, -piso_numerico, -piso_info)

# VALOR X ZONA -----------------------------------------------------------------