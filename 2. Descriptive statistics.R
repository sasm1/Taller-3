################################################################################


# Limpiar environment -----------------------------------------------------
rm(list = ls())
gc() 
cat('\014')
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

################################################################################

load("Datos_limpios.RData")

df <- df %>% select(-fecha_acto_administrativo, -codigo_criterio,-CP_TERR_AR) # Se nos fue :c 

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
df_corr <- df  # copia del original
cols_to_convert <- c("cocina_lujo", "cocina_estandar", "parqueadero", "terraza", 
                     "sala_comedor", "patio_lavanderia", "walkin_closet", "estudio", 
                     "closet", "saloncomunal_recepcion", "seguridad", "lujos", 
                     "remodelado", "codigo_criterio", "GRUPOP_TER")

for (col in cols_to_convert) {
  df_corr[[col]] <- as.numeric(as.factor(df_corr[[col]]))
}

correlaciones <- sort(abs(cor(df_corr[, sapply(df_corr, is.numeric)], use = "complete.obs")["price", ]), decreasing = TRUE)
correlaciones; rm(df_corr)


