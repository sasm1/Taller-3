
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
       geojsonR, # Datos espaciales 
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
# CREAR NUEVAS VARIABLESINMUEBLE
################################################################################
# VALOR X AMENIDADES------------------------------------------------------------
# ----- Closet 
df <- df %>% mutate(
  closet = as.numeric(grepl("closet", description, ignore.case = TRUE) &
                         !grepl("walk-in closet", description, ignore.case = TRUE)))

# ---- Salon Comunal y Recepción 
df <- df %>%
  mutate(saloncomunal_recepcion = as.numeric(grepl("salon comunal| social|saln|recepcion|
                                             lobby|hall",description)))
# ---- Portería/ Seguridad
df <- df %>%
  mutate(seguridad = as.numeric(grepl("seguridad|vigilancia|porteria", description)))
  
# ---- Piso
df <- df %>%
  mutate(
    piso_info = str_extract(description, "(\\b\\w+\\b)?\\s*piso\\s*(\\b\\w+\\b)?")
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
    ),
    piso_numerico = as.integer(piso_texto),
    piso_numerico = if_else(piso_numerico > 30, NA_integer_, piso_numerico)
  )  # Asumimos hasta piso 30 

df<- df %>%  # Llenamos con la mediana si no encuentra valor
  group_by(property_type) %>%
  mutate(
    piso = if_else(is.na(piso_numerico), median(piso_numerico, na.rm = TRUE), piso_numerico)
  ) %>%
  select(-piso_texto, -piso_numerico, -piso_info)

# ---- Amenidades de lujo 
df <- df %>%
  mutate(lujos = as.numeric(grepl("gimnasio|bbq|club|cancha|sauna
                                  |squash|jacuzzi|gym|piscina|pisina|
                                  moderno|duplex", description)))

# ---- Remodelado o Nuevo
df <- df %>% mutate(remodelado = as.numeric(grepl("remodelado|remodelada|remodelad|
                                                  nuevo|remodlado|nuevos|nueva|nuevas", description)))

# --- CONVIRTIENDO A FACTOR
df <- df %>%
  mutate(
    cocina_lujo              = factor(cocina_lujo, levels = c(0, 1), labels = c("No", "Sí")),
    cocina_estandar          = factor(cocina_estandar, levels = c(0, 1), labels = c("No", "Sí")),
    parqueadero              = factor(parqueadero, levels = c(0, 1), labels = c("No", "Sí")),
    terraza                  = factor(terraza, levels = c(0, 1), labels = c("No", "Sí")),
    sala_comedor             = factor(sala_comedor, levels = c(0, 1), labels = c("No", "Sí")),
    patio_lavanderia         = factor(patio_lavanderia, levels = c(0, 1), labels = c("No", "Sí")),
    walkin_closet            = factor(walkin_closet, levels = c(0, 1), labels = c("No", "Sí")),
    estudio                  = factor(estudio, levels = c(0, 1), labels = c("No", "Sí")),
    closet                   = factor(closet, levels = c(0, 1), labels = c("No", "Sí")),
    saloncomunal_recepcion   = factor(saloncomunal_recepcion, levels = c(0, 1), labels = c("No", "Sí")),
    seguridad                = factor(seguridad, levels = c(0, 1), labels = c("No", "Sí")),
    lujos                    = factor(lujos, levels = c(0, 1), labels = c("No", "Sí")),
    remodelado               = factor(remodelado, levels = c(0, 1), labels = c("No", "Sí"))
  )


################################################################################
# CREAR NUEVAS VARIABLES PERO ESPACIALES
################################################################################

# VALOR X ZONA -----------------------------------------------------------------
bogota <- opq(bbox = getbb ("Bogotá Colombia"))
df_sf <- st_as_sf(df, coords = c("lon", "lat")) # Convertir a simple features
st_crs(df_sf) <- 4326 # Sistema de coordenadas
available_features()

#-------------------------------------------------------------------------------
# ---- Transmilenio y SITP
# Distancia de transporte público
tm_osm <- opq("Bogotá, Colombia") %>%
  add_osm_feature(key = "highway", value = "bus_stop") %>%
  osmdata_sf()
estaciones_tm <- tm_osm$osm_points
propiedades_sf <- st_as_sf(df, coords = c("lon", "lat"), crs = 4326)

coords_prop <- st_coordinates(propiedades_sf)
coords_tm <- st_coordinates(estaciones_tm)

dist_matrix <- geosphere::distm(coords_prop, coords_tm)
dist_min <- apply(dist_matrix, 1, min)

plot(st_geometry(estaciones_tm), main = "Estaciones de bus (incluye TM)")

df$dist_tm_metros <- dist_min
df

# SEPARANDO SITPS Y TRANSMILENIOS (DATA EXTERNA)
transmi <- geojson_read("Data_espacial/Estaciones_Troncales_de_TRANSMILENIO.geojson")
geometria.transmi <- transmi %>% mutate(longitud_estacion = st_coordinates(.)[, 1],
                                        latitud_estacion = st_coordinates(.)[, 2]) %>%
  select(nombre_estacion, latitud_estacion, longitud_estacion)


SITPs <- geojson_read("Data_espacial/Paraderos_Zonales_del_SITP.geojson")
geometria.SITPs <- SITPs %>% mutate(longitud = st_coordinates(.)[, 1],
                                    latitud = st_coordinates(.)[, 2]) %>%
  select(nombre, latitud, longitud)

transmi.sf <- st_as_sf(x = geometria.transmi, coords = c('longitud_estacion','latitud_estacion'),
                            crs = st_crs(df_sf)) # Convirtiendo en datos espaciales 
SITPs.sf <- st_as_sf(x = geometria.SITPs, coords = c('longitud','latitud'),crs = st_crs(df_sf))

rm(transmi,SITPs,geometria.transmi,geometria.SITPs)

# Matrices de distancias para transmilenio y sitp
matrix.distancias.tm    <- st_distance(x=df_sf, y = transmi.sf)
matrix.distancias.sitp  <- st_distance(x=df_sf, y = sitp.sf)

# Distancias minimas y agregar a base de datos
bd$distancia_tm   <- apply(matrix.distancias.tm, 1, min) 



# ---- Num Parques cercanos

# Obtener parques desde OpenStreetMap y transformar a EPSG:4326
parques_osm <- opq("Bogotá, Colombia") %>%
  add_osm_feature(key = "leisure", value = "park") %>%
  osmdata_sf()

parques <- parques_osm$osm_polygons %>% select(geometry) %>%
  st_transform(crs = 4326) 
propiedades_sf <- st_as_sf(df, coords = c("lon", "lat"), crs = 4326)
buffer <- st_buffer(propiedades_sf, dist = 500)

# Contar el número de parques dentro de cada buffer
df$num_parques <- lengths(st_intersects(buffer, parques))


# ---- Distancia al parque más cercano
# Calcular centroides de los parques
centroides <- st_centroid(parques) %>%
  mutate(lon = st_coordinates(.)[, 1], lat = st_coordinates(.)[, 2])

# Calcular la matriz de distancias entre propiedades y parques
dist_matrix <- st_distance(propiedades_sf, parques)
dim(dist_matrix)
dist_min <- apply(dist_matrix, 1, min) 

df <- df %>% mutate(distancia_min_parque = dist_min)
p <- ggplot(df%>%sample_n(5000), aes(x = dist_tm_metros, y = price)) +
  geom_point(col = "darkblue", alpha = 0.4) +
  labs(x = "Distancia mínima a un parque en metros (log-scale)", 
       y = "Valor de venta  (log-scale)",
       title = "Relación entre la proximidad a un parque y el precio del immueble") +
  scale_x_log10() +
  scale_y_log10(labels = scales::dollar) +
  theme_bw()
ggplotly(p)

#---- Densidad de servicios
# Obtener servicios y comercios de OSM
servicios <- opq("Bogotá, Colombia") %>%
  add_osm_feature(key = "amenity", 
                  value = c("restaurant", "cafe", "bank", "pharmacy", 
                            "supermarket", "school", "hospital", "marketplace")) %>%
  osmdata_sf()


puntos_servicios <- servicios$osm_points
radio <- 800  # metros
todos_buffers <- st_buffer(propiedades_sf, dist = radio)
intersecciones <- st_intersects(todos_buffers, puntos_servicios)
df$service_density <- lengths(intersecciones)

head(df)

# Ciclorutas 
datos.osm1[[2]] <- bogota %>%
  add_osm_feature(key = "highway", value= "cycleway")%>%
  osmdata_sf()

# centro urbanos de servicios
datos.osm1[[3]] <- bogota %>%
  add_osm_feature(key = "landuse", value= "commercial")%>%
  osmdata_sf()

nombres.datos.osm <- c('mall','cycleway','commercial')
names(datos.osm1) <- nombres.datos.osm
