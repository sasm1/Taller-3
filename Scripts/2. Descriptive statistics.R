################################################################################
# Limpiar environment -----------------------------------------------------
rm(list = ls())
gc() 
cat('\014')
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

################################################################################
# Cargar librerías -----------------------------------------------------

library(pacman)
p_load(sf,ggplot2,osmdata,dplyr)

################################################################################
load("Datos_limpios.RData")
################################################################################
# NAs ----------------------------------------------------------------
na_percent <- sapply(df, function(col) {
  mean(is.na(col)) * 100
})
na_percent <- data.frame(Column = names(na_percent),
                            NA_Percent = round(na_percent, 2))
na_percent

# HACER FACTORES ----------------------------------------------------------------
df$estrato <- as.factor(df$estrato)

# LLENAR ESTRATOS  --------------------------------------------------------------
colnames(df)
df <- df %>%
  mutate(
    estrato = case_when(
      is.na(estrato) & lujos == "Sí" & cocina_lujo == "Sí" & bedrooms > 3 ~ "5",
      is.na(estrato) ~ "3",
      TRUE ~ as.character(estrato)
    ),
    estrato = factor(estrato, levels = sort(unique(as.character(estrato))))
  )
# LLENAR CODIGO_ZONA_ESTRATO ---------------------------------------------------
df$codigo_zona_estrato <- as.numeric(as.character(df$codigo_zona_estrato))
df <- df %>%
  group_by(codigo_localidad, estrato) %>%
  mutate(
    codigo_zona_estrato = ifelse(
      is.na(codigo_zona_estrato),
      median(codigo_zona_estrato, na.rm = TRUE),
      codigo_zona_estrato
    )
  ) %>%
  ungroup()
df$codigo_zona_estrato <- as.factor(df$codigo_zona_estrato)


################################################################################
# CORRELACIONES ----------------------------------------------------------------
# Copia del data frame original
df_corr <- df
if ("geometry" %in% names(df_corr)) {
  df_corr$geometry <- NULL
}

# Variables categóricas a convertir a numérico
cols_to_convert <- c("cocina_lujo", "cocina_estandar", "parqueadero", "terraza", 
                     "sala_comedor", "patio_lavanderia", "walkin_closet", "estudio", 
                     "closet", "saloncomunal_recepcion", "seguridad", "lujos", 
                     "remodelado", "codigo_criterio", "GRUPOP_TER")
for (col in cols_to_convert) {
  if (col %in% names(df_corr)) {
    df_corr[[col]] <- as.numeric(as.factor(df_corr[[col]]))
  }
} # Convertir las seleccionadas a factor y luego a numérico

# Filtrar solo columnas numéricas con al menos 2 valores distintos (evita errores)
numeric_cols <- sapply(df_corr, function(x) is.numeric(x) && length(unique(x[!is.na(x)])) > 1)

# Subset con solo las columnas válidas
df_numeric <- df_corr[, numeric_cols]

# Asegurar que 'price' esté presente
if (!"price" %in% names(df_numeric)) {
  stop("La columna 'price' no es numérica o fue filtrada.")
}

# Calcular correlaciones con 'pairwise.complete.obs' para tolerar NA
cor_matrix <- cor(df_numeric, use = "pairwise.complete.obs")

cor_price <- cor_matrix["price", setdiff(colnames(cor_matrix), "price")]
correlaciones <- sort(abs(cor_price), decreasing = TRUE)
write.csv(correlaciones, "stores/Correlaciones_precio.csv")

################################################################################
# MAPAS 
barrios <- st_read("Data_espacial/barrios-bogota/barrios-bogota.geojson")
barrios <- st_transform(barrios, crs = 4326)

# PRECIO POR METRO CUADRADO ----------------------------------------------------
df <- df %>%
  mutate(precio_m2 = price / surface_covered)

precio_m2_barrio <- df %>%
  filter(GRUPOP_TER == "RESIDENCIAL") %>%
  group_by(nombre, GRUPOP_TER) %>%
  summarise(promedio_precio_m2 = mean(precio_m2, na.rm = TRUE) / 1e6, .groups = "drop")

barrios <- barrios %>%
  left_join(st_drop_geometry(precio_m2_barrio), by = "nombre")

# Excluir Bogotá rural
barrios_res <- barrios %>%
  filter(GRUPOP_TER == "RESIDENCIAL") %>%
  mutate(lat = st_coordinates(st_centroid(geometry))[,2]) %>%
  filter(lat > 4.45)

map_res <- ggplot() +
  geom_sf(data = barrios_res, aes(fill = promedio_precio_m2), color = "white", size = 0.1) +
  scale_fill_viridis_c(option = "plasma", na.value = "gray80") +
  labs(subtitle = "Surface covered",
       fill = "$ Millones COP/sqmt") +
  theme_minimal()

ggsave("precio_m2_residencial.png", plot = map_res,
       width = 10, height = 8, dpi = 300)

# DISTANCIA PROMEDIO A ESTACIÓN DE BICI ----------------------------------------

dist_promedio_bicis <- df %>%
  group_by(nombre) %>%
  summarise(dist_prom = mean(distancia_bicycle_rental, na.rm = TRUE), .groups = "drop")

barrios <- barrios %>%
  left_join(st_drop_geometry(dist_promedio_bicis), by = "nombre")

barrios_bicis <- barrios %>%
  mutate(lat = st_coordinates(st_centroid(geometry))[,2]) %>%
  filter(lat > 4.45)

map_bicis <- ggplot() +
  geom_sf(data = barrios_bicis, aes(fill = dist_prom), color = "white", size = 0.1) +
  scale_fill_viridis_c(option = "plasma", na.value = "gray80") +
  labs(fill = "mts") +
  theme_minimal()

ggsave("average_distance_bikes.png", plot = map_bicis,
       width = 10, height = 8, dpi = 300)
