<<<<<<< HEAD
<<<<<<< HEAD
################################################################################

# Limpiar environment -----------------------------------------------------
rm(list = ls())
gc() 
cat('\014')
install.packages("rstudioapi")
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

# Cargar paquetes ---------------------------------------------------------
library(pacman)
# Cargar las librerías listadas e instalarlas en caso de ser necesario
p_load(tidyverse, # Manipular dataframes
       stringi, # Manipular cadenas de texto
       rio, # Importar datos fácilmente
       sf, # Leer/escribir/manipular datos espaciales
       tidymodels, # entrenamiento de modelos
       spatialsample, # Muestreo espacial para modelos de aprendizaje automático
       rsample, # Resamplear los datos
       dplyr,
       parsnip, # elastic net
       dials, # elastic net tunning
       recipies, # Recetas
       workflows)   # Crear worklows RN

# Cargar los datos --------------------------------------------------------
load("Datos_limpios.RData")

################################################################################
# GRADIENT BOOSTING


################################################################################
# ELASTIC NET ------------------------------------------------------------------
neural_cols <- c("grupo","price","surface_imputado","surface_covered","surface","bathrooms",
            "rooms","bedrooms","codigo_zona_estrato","estudio","estrato",
            "distancia_love_hotel","estrato","distancia_bicycle_rental","distancia_funeral_hall",
            "distancia_bicycle_parking","distancia_toilets","distancia_college")
neural <- df %>%  select(all_of(neural_cols)) %>%  filter(grupo == "train") 
neural$codigo_zona_estrato <- as.factor(neural$codigo_zona_estrato)
neural$estrato <- as.factor(neural$estrato)

coords <- st_coordinates(neural)
neural$long <- coords[,1]
neural$lat <- coords[,2]
neural <- st_drop_geometry(neural)

split <- initial_split(neural, prop = 0.8) #Split
neural_train <- training(split) # Training
neural_test  <- testing(split) ; rm(neural,split) # Test


# A. --- Creamos recetas
# Primera receta 
rec_1 <- recipe(price ~ surface_imputado + bathrooms + rooms + bedrooms + codigo_zona_estrato + estudio + distancia_love_hotel, data = neural_train) %>%
  step_unknown(all_nominal_predictors())%>%
  step_impute_mean(all_numeric_predictors()) %>%
  step_impute_mode(all_nominal_predictors()) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_zv(all_predictors()) %>%
  step_normalize(all_predictors())

# Segunda receta  
rec_2 <- recipe(price ~ long + lat + surface_covered + rooms + bedrooms, data = neural_train) %>%
  step_unknown(all_nominal_predictors())%>%
  step_impute_mean(all_numeric_predictors()) %>%
  step_dummy(all_nominal_predictors()) %>% 
  step_poly(long,lat, degree = 2) %>%
  step_novel(all_nominal_predictors()) %>% 
  step_zv(all_predictors()) %>% 
  step_normalize(all_predictors())

# Tercera receta  
rec_3 <- recipe(price ~ surface_imputado + bathrooms + rooms + estrato, data = neural_train) %>%
  step_unknown(all_nominal_predictors()) %>%
  step_novel(all_nominal_predictors()) %>%
  step_impute_mean(all_numeric_predictors()) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_zv(all_predictors()) %>%
  step_normalize(all_numeric_predictors())

# Cuarta receta  (TODO)
rec_4 <- recipe(price ~ . , data = neural_train) %>%
  step_unknown(all_nominal_predictors())%>%
  step_impute_mean(all_numeric_predictors()) %>%
  step_novel(all_nominal_predictors()) %>% 
  step_dummy(all_nominal_predictors()) %>% 
  step_zv(all_predictors())%>% 
  step_normalize(all_predictors()) 

# B. --- Tuneamos la búsqueda de los parámetros de la red
elastic_net_spec <- parsnip::linear_reg(
  penalty = tune(), # penalidad lamda
  mixture = tune()) %>% # lasso o ridge 
  set_engine("glmnet")

grid_values <- grid_regular(penalty(range = c(-2,1)), levels = 50) %>%
  expand_grid(mixture = c(0, 0.25,  0.5, 0.75,  1))

set.seed(86936)
db_fold <- rsample::vfold_cv(neural_train, v = 5)

# C. --- Definir workflows
# Definimos un flujo de trabajo (workflow)
# Iniciar un flujo de trabajo utilizando 'workflow()'

workflow_1 <- workflow() %>% 
  add_recipe(rec_1) %>%   
  add_model(elastic_net_spec)  

workflow_2 <- workflow() %>%
  add_recipe(rec_2) %>%
  add_model(elastic_net_spec)

workflow_3 <- workflow() %>%
  add_recipe(rec_3) %>%
  add_model(elastic_net_spec)

workflow_4 <- workflow() %>%
  add_recipe(rec_4) %>%
  add_model(elastic_net_spec)

# D. --- FIT AND PREDICT 
metrics <- metric_set(rmse, rsq, mae)
set.seed(86936)
tune_res1 <- tune::tune_grid(
  workflow_1,         # El flujo de trabajo que contiene: receta y especificación del modelo
  resamples = db_fold,  # Folds de validación cruzada
  grid = grid_values,        # Grilla de valores de penalización
  metrics = metrics  # metrica
)

tune_res2 <- tune::tune_grid(
  workflow_2,         
  resamples = db_fold,  # Folds de validación cruzada
  grid = grid_values,        # Grilla de valores de penalización
  metrics = metrics  # metrica
)

tune_res3 <- tune::tune_grid(
  workflow_3,         
  resamples = db_fold,  # Folds de validación cruzada
  grid = grid_values,        # Grilla de valores de penalización
  metrics = metrics  # metrica
)


tune_res4 <- tune::tune_grid(
  workflow_4,         
  resamples = db_fold,  # Folds de validación cruzada
  grid = grid_values,        # Grilla de valores de penalización
  metrics = metrics  # metrica
)

# E. ---- SELECCIONAR EL MODELO 
best_penalty_1 <- select_best(tune_res1, metric = "mae")
best_penalty_2 <- select_best(tune_res2, metric = "mae")
best_penalty_3 <- select_best(tune_res3, metric = "mae")
best_penalty_4 <- select_best(tune_res4, metric = "mae")


# F. ---- FINALIZAR LOS WORKFLOWS
EN_final1 <- finalize_workflow(workflow_1, best_penalty_1)
EN_final2 <- finalize_workflow(workflow_2, best_penalty_2)
EN_final3 <- finalize_workflow(workflow_3, best_penalty_3)
EN_final4 <- finalize_workflow(workflow_4, best_penalty_4)


# G. ---- AJUSTAR MODELOS
EN_final1_fit <- fit(EN_final1, data = neural_train)
EN_final2_fit <- fit(EN_final2, data = neural_train)
EN_final3_fit <- fit(EN_final3, data = neural_train)
EN_final4_fit <- fit(EN_final4, data = neural_train)

# H. ---- PREDICCIONES SOBRE TEST
predictiones_1 <- predict(EN_final1_fit , new_data = neural_test)
predictiones_2 <- predict(EN_final2_fit , new_data = neural_test)
predictiones_3 <- predict(EN_final3_fit , new_data = neural_test)
predictiones_4 <- predict(EN_final4_fit , new_data = neural_test)


# I. ---- COMPARAR LOS PERFORMANCE
mae_1 <- neural_test %>%
  bind_cols(predictiones_1) %>%
  yardstick::mae(truth = price, estimate = .pred)

resultados <- c(mae_1[[".estimate"]])

mae_2<- neural_test %>%
  bind_cols(predictiones_2) %>%
  yardstick::mae(truth = price, estimate = .pred)
resultados <- c(mae_2[[".estimate"]])


mae_3<- neural_test %>%
  bind_cols(predictiones_3) %>%
  yardstick::mae(truth = price, estimate = .pred)
resultados <- c(mae_3[[".estimate"]])

mae_4<- neural_test %>%
  bind_cols(predictiones_4) %>%
  yardstick::mae(truth = price, estimate = .pred)
resultados <- c(mae_4[[".estimate"]])

resultados <- c(
  modelo_1 = mae_1[[".estimate"]],
  modelo_2 = mae_2[[".estimate"]],
  modelo_3 = mae_3[[".estimate"]],
  modelo_4 = mae_4[[".estimate"]]
)

resultados_df <- tibble(
  modelo = paste0("modelo_", 1:4),
  mae = c(mae_1[[".estimate"]], mae_2[[".estimate"]], mae_3[[".estimate"]], mae_4[[".estimate"]])
)

resultados_df %>% arrange(mae) # Cómo que el más chimbita es el modelo 4 :) 


# J. ---- AHORA SÍ PA ENTREGAR:
neural_cols <- c("property_id", "grupo", "price", "surface_imputado", "surface_covered",
                 "surface", "bathrooms", "rooms", "bedrooms", "codigo_zona_estrato",
                 "estudio", "estrato", "distancia_love_hotel", "distancia_bicycle_rental",
                 "distancia_funeral_hall", "distancia_bicycle_parking",
                 "distancia_toilets", "distancia_college")

neural  <- df %>%  select(all_of(neural_cols))
neural$codigo_zona_estrato <- as.factor(neural$codigo_zona_estrato)
neural$estrato <- as.factor(neural$estrato)
coords <- st_coordinates(neural)
neural$long <- coords[,1]
neural$lat <- coords[,2]
neural <- st_drop_geometry(neural)

neural_train2 <- neural %>%  select(all_of(neural_cols)) %>%  filter(grupo == "train") 
neural_test_real <- neural %>%  select(all_of(neural_cols)) %>%  filter(grupo == "test"); rm(neural) 
neural_train2 <- neural_train2 %>% select(-property_id)


rec_4 <- recipe(price ~ . , data = neural_train2) %>%
  step_unknown(all_nominal_predictors())%>%
  step_impute_mean(all_numeric_predictors()) %>%
  step_novel(all_nominal_predictors()) %>% 
  step_dummy(all_nominal_predictors()) %>% 
  step_zv(all_predictors())%>% 
  step_normalize(all_predictors()) 

elastic_net_spec <- parsnip::linear_reg(
  penalty = tune(), # penalidad lamda
  mixture = tune()) %>% # lasso o ridge 
  set_engine("glmnet")

grid_values <- grid_regular(penalty(range = c(-2,1)), levels = 50) %>%
  expand_grid(mixture = c(0, 0.25,  0.5, 0.75,  1))

set.seed(86936)
db_fold <- rsample::vfold_cv(neural_train2, v = 5)

workflow_4 <- workflow() %>%
  add_recipe(rec_4) %>%
  add_model(elastic_net_spec)

tune_res4 <- tune::tune_grid(
  workflow_4,         
  resamples = db_fold,  # Folds de validación cruzada
  grid = grid_values,        # Grilla de valores de penalización
  metrics = metrics  # metrica
)

best_penalty_4 <- select_best(tune_res4, metric = "mae")
EN_final <- finalize_workflow(workflow_4, best_penalty_4)
EN_final4_fit <- fit(EN_final, data = neural_train2)
predictiones_EN <- predict(EN_final4_fit , new_data = neural_test_real)

predictions_EN <- data.frame(
  property_id = neural_test_real$property_id,
  price  = predictiones_EN$.pred
)

write.csv(predictions_EN,"EN_MODEL4.csv", row.names = FALSE)

rm(list = setdiff(ls(), c("df","coordenadas.x.centroides","coordenadas.y.centroides",
  "tune_res4","workflow_4", "rec_4","predictions_EN"  # ← corregido: se llama predictiones_EN, no predictions_EN
  )))

################################################################################
=======
################################################################################

# Limpiar environment -----------------------------------------------------
rm(list = ls())
gc() 
cat('\014')
install.packages("rstudioapi")
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

# Cargar paquetes ---------------------------------------------------------
library(pacman)
# Cargar las librerías listadas e instalarlas en caso de ser necesario
p_load(tidyverse, # Manipular dataframes
       stringi, # Manipular cadenas de texto
       rio, # Importar datos fácilmente
       sf, # Leer/escribir/manipular datos espaciales
       tidymodels, # entrenamiento de modelos
       spatialsample, # Muestreo espacial para modelos de aprendizaje automático
       rsample, # Resamplear los datos
       dplyr,
       parsnip, # elastic net
       dials, # elastic net tunning
       recipies, # Recetas
       workflows)   # Crear worklows RN

# Cargar los datos --------------------------------------------------------
load("Datos_limpios.RData")

################################################################################
# GRADIENT BOOSTING


################################################################################
# ELASTIC NET ------------------------------------------------------------------
neural_cols <- c("grupo","price","surface_imputado","surface_covered","surface","bathrooms",
            "rooms","bedrooms","codigo_zona_estrato","estudio","estrato",
            "distancia_love_hotel","estrato","distancia_bicycle_rental","distancia_funeral_hall",
            "distancia_bicycle_parking","distancia_toilets","distancia_college")
neural <- df %>%  select(all_of(neural_cols)) %>%  filter(grupo == "train") 
neural$codigo_zona_estrato <- as.factor(neural$codigo_zona_estrato)
neural$estrato <- as.factor(neural$estrato)

coords <- st_coordinates(neural)
neural$long <- coords[,1]
neural$lat <- coords[,2]
neural <- st_drop_geometry(neural)

split <- initial_split(neural, prop = 0.8) #Split
neural_train <- training(split) # Training
neural_test  <- testing(split) ; rm(neural,split) # Test


# A. --- Creamos recetas
# Primera receta 
rec_1 <- recipe(price ~ surface_imputado + bathrooms + rooms + bedrooms + codigo_zona_estrato + estudio + distancia_love_hotel, data = neural_train) %>%
  step_unknown(all_nominal_predictors())%>%
  step_impute_mean(all_numeric_predictors()) %>%
  step_impute_mode(all_nominal_predictors()) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_zv(all_predictors()) %>%
  step_normalize(all_predictors())

# Segunda receta  
rec_2 <- recipe(price ~ long + lat + surface_covered + rooms + bedrooms, data = neural_train) %>%
  step_unknown(all_nominal_predictors())%>%
  step_impute_mean(all_numeric_predictors()) %>%
  step_dummy(all_nominal_predictors()) %>% 
  step_poly(long,lat, degree = 2) %>%
  step_novel(all_nominal_predictors()) %>% 
  step_zv(all_predictors()) %>% 
  step_normalize(all_predictors())

# Tercera receta  
rec_3 <- recipe(price ~ surface_imputado + bathrooms + rooms + estrato, data = neural_train) %>%
  step_unknown(all_nominal_predictors()) %>%
  step_novel(all_nominal_predictors()) %>%
  step_impute_mean(all_numeric_predictors()) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_zv(all_predictors()) %>%
  step_normalize(all_numeric_predictors())

# Cuarta receta  (TODO)
rec_4 <- recipe(price ~ . , data = neural_train) %>%
  step_unknown(all_nominal_predictors())%>%
  step_impute_mean(all_numeric_predictors()) %>%
  step_novel(all_nominal_predictors()) %>% 
  step_dummy(all_nominal_predictors()) %>% 
  step_zv(all_predictors())%>% 
  step_normalize(all_predictors()) 

# B. --- Tuneamos la búsqueda de los parámetros de la red
elastic_net_spec <- parsnip::linear_reg(
  penalty = tune(), # penalidad lamda
  mixture = tune()) %>% # lasso o ridge 
  set_engine("glmnet")

grid_values <- grid_regular(penalty(range = c(-2,1)), levels = 50) %>%
  expand_grid(mixture = c(0, 0.25,  0.5, 0.75,  1))

set.seed(86936)
db_fold <- rsample::vfold_cv(neural_train, v = 5)

# C. --- Definir workflows
# Definimos un flujo de trabajo (workflow)
# Iniciar un flujo de trabajo utilizando 'workflow()'

workflow_1 <- workflow() %>% 
  add_recipe(rec_1) %>%   
  add_model(elastic_net_spec)  

workflow_2 <- workflow() %>%
  add_recipe(rec_2) %>%
  add_model(elastic_net_spec)

workflow_3 <- workflow() %>%
  add_recipe(rec_3) %>%
  add_model(elastic_net_spec)

workflow_4 <- workflow() %>%
  add_recipe(rec_4) %>%
  add_model(elastic_net_spec)

# D. --- FIT AND PREDICT 
metrics <- metric_set(rmse, rsq, mae)
set.seed(86936)
tune_res1 <- tune::tune_grid(
  workflow_1,         # El flujo de trabajo que contiene: receta y especificación del modelo
  resamples = db_fold,  # Folds de validación cruzada
  grid = grid_values,        # Grilla de valores de penalización
  metrics = metrics  # metrica
)

tune_res2 <- tune::tune_grid(
  workflow_2,         
  resamples = db_fold,  # Folds de validación cruzada
  grid = grid_values,        # Grilla de valores de penalización
  metrics = metrics  # metrica
)

tune_res3 <- tune::tune_grid(
  workflow_3,         
  resamples = db_fold,  # Folds de validación cruzada
  grid = grid_values,        # Grilla de valores de penalización
  metrics = metrics  # metrica
)


tune_res4 <- tune::tune_grid(
  workflow_4,         
  resamples = db_fold,  # Folds de validación cruzada
  grid = grid_values,        # Grilla de valores de penalización
  metrics = metrics  # metrica
)

# E. ---- SELECCIONAR EL MODELO 
best_penalty_1 <- select_best(tune_res1, metric = "mae")
best_penalty_2 <- select_best(tune_res2, metric = "mae")
best_penalty_3 <- select_best(tune_res3, metric = "mae")
best_penalty_4 <- select_best(tune_res4, metric = "mae")


# F. ---- FINALIZAR LOS WORKFLOWS
EN_final1 <- finalize_workflow(workflow_1, best_penalty_1)
EN_final2 <- finalize_workflow(workflow_2, best_penalty_2)
EN_final3 <- finalize_workflow(workflow_3, best_penalty_3)
EN_final4 <- finalize_workflow(workflow_4, best_penalty_4)


# G. ---- AJUSTAR MODELOS
EN_final1_fit <- fit(EN_final1, data = neural_train)
EN_final2_fit <- fit(EN_final2, data = neural_train)
EN_final3_fit <- fit(EN_final3, data = neural_train)
EN_final4_fit <- fit(EN_final4, data = neural_train)

# H. ---- PREDICCIONES SOBRE TEST
predictiones_1 <- predict(EN_final1_fit , new_data = neural_test)
predictiones_2 <- predict(EN_final2_fit , new_data = neural_test)
predictiones_3 <- predict(EN_final3_fit , new_data = neural_test)
predictiones_4 <- predict(EN_final4_fit , new_data = neural_test)


# I. ---- COMPARAR LOS PERFORMANCE
mae_1 <- neural_test %>%
  bind_cols(predictiones_1) %>%
  yardstick::mae(truth = price, estimate = .pred)

resultados <- c(mae_1[[".estimate"]])

mae_2<- neural_test %>%
  bind_cols(predictiones_2) %>%
  yardstick::mae(truth = price, estimate = .pred)
resultados <- c(mae_2[[".estimate"]])


mae_3<- neural_test %>%
  bind_cols(predictiones_3) %>%
  yardstick::mae(truth = price, estimate = .pred)
resultados <- c(mae_3[[".estimate"]])

mae_4<- neural_test %>%
  bind_cols(predictiones_4) %>%
  yardstick::mae(truth = price, estimate = .pred)
resultados <- c(mae_4[[".estimate"]])

resultados <- c(
  modelo_1 = mae_1[[".estimate"]],
  modelo_2 = mae_2[[".estimate"]],
  modelo_3 = mae_3[[".estimate"]],
  modelo_4 = mae_4[[".estimate"]]
)

resultados_df <- tibble(
  modelo = paste0("modelo_", 1:4),
  mae = c(mae_1[[".estimate"]], mae_2[[".estimate"]], mae_3[[".estimate"]], mae_4[[".estimate"]])
)

resultados_df %>% arrange(mae) # Cómo que el más chimbita es el modelo 4 :) 


# J. ---- AHORA SÍ PA ENTREGAR:
neural_cols <- c("property_id", "grupo", "price", "surface_imputado", "surface_covered",
                 "surface", "bathrooms", "rooms", "bedrooms", "codigo_zona_estrato",
                 "estudio", "estrato", "distancia_love_hotel", "distancia_bicycle_rental",
                 "distancia_funeral_hall", "distancia_bicycle_parking",
                 "distancia_toilets", "distancia_college")

neural  <- df %>%  select(all_of(neural_cols))
neural$codigo_zona_estrato <- as.factor(neural$codigo_zona_estrato)
neural$estrato <- as.factor(neural$estrato)
coords <- st_coordinates(neural)
neural$long <- coords[,1]
neural$lat <- coords[,2]
neural <- st_drop_geometry(neural)

neural_train2 <- neural %>%  select(all_of(neural_cols)) %>%  filter(grupo == "train") 
neural_test_real <- neural %>%  select(all_of(neural_cols)) %>%  filter(grupo == "test"); rm(neural) 
neural_train2 <- neural_train2 %>% select(-property_id)


rec_4 <- recipe(price ~ . , data = neural_train2) %>%
  step_unknown(all_nominal_predictors())%>%
  step_impute_mean(all_numeric_predictors()) %>%
  step_novel(all_nominal_predictors()) %>% 
  step_dummy(all_nominal_predictors()) %>% 
  step_zv(all_predictors())%>% 
  step_normalize(all_predictors()) 

elastic_net_spec <- parsnip::linear_reg(
  penalty = tune(), # penalidad lamda
  mixture = tune()) %>% # lasso o ridge 
  set_engine("glmnet")

grid_values <- grid_regular(penalty(range = c(-2,1)), levels = 50) %>%
  expand_grid(mixture = c(0, 0.25,  0.5, 0.75,  1))

set.seed(86936)
db_fold <- rsample::vfold_cv(neural_train2, v = 5)

workflow_4 <- workflow() %>%
  add_recipe(rec_4) %>%
  add_model(elastic_net_spec)

tune_res4 <- tune::tune_grid(
  workflow_4,         
  resamples = db_fold,  # Folds de validación cruzada
  grid = grid_values,        # Grilla de valores de penalización
  metrics = metrics  # metrica
)

best_penalty_4 <- select_best(tune_res4, metric = "mae")
EN_final <- finalize_workflow(workflow_4, best_penalty_4)
EN_final4_fit <- fit(EN_final, data = neural_train2)
predictiones_EN <- predict(EN_final4_fit , new_data = neural_test_real)

predictions_EN <- data.frame(
  property_id = neural_test_real$property_id,
  price  = predictiones_EN$.pred
)

write.csv(predictions_EN,"EN_MODEL4.csv", row.names = FALSE)

rm(list = setdiff(ls(), c("df","coordenadas.x.centroides","coordenadas.y.centroides",
  "tune_res4","workflow_4", "rec_4","predictions_EN"  # ← corregido: se llama predictiones_EN, no predictions_EN
  )))

################################################################################
>>>>>>> 3acc7330c9dd0ede7faff02aa2bc74ba7a14363b
=======
################################################################################

# Limpiar environment -----------------------------------------------------
rm(list = ls())
gc() 
cat('\014')
install.packages("rstudioapi")
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

# Cargar paquetes ---------------------------------------------------------
library(pacman)
# Cargar las librerías listadas e instalarlas en caso de ser necesario
p_load(tidyverse, # Manipular dataframes
       stringi, # Manipular cadenas de texto
       rio, # Importar datos fácilmente
       sf, # Leer/escribir/manipular datos espaciales
       tidymodels, # entrenamiento de modelos
       spatialsample, # Muestreo espacial para modelos de aprendizaje automático
       rsample, # Resamplear los datos
       dplyr,
       parsnip, # elastic net
       dials, # elastic net tunning
       recipies, # Recetas
       workflows)   # Crear worklows RN

# Cargar los datos --------------------------------------------------------
load("Datos_limpios.RData")

################################################################################
# GRADIENT BOOSTING


################################################################################
# ELASTIC NET ------------------------------------------------------------------
neural_cols <- c("grupo","price","surface_imputado","surface_covered","surface","bathrooms",
            "rooms","bedrooms","codigo_zona_estrato","estudio","estrato",
            "distancia_love_hotel","estrato","distancia_bicycle_rental","distancia_funeral_hall",
            "distancia_bicycle_parking","distancia_toilets","distancia_college")
neural <- df %>%  select(all_of(neural_cols)) %>%  filter(grupo == "train") 
neural$codigo_zona_estrato <- as.factor(neural$codigo_zona_estrato)
neural$estrato <- as.factor(neural$estrato)

coords <- st_coordinates(neural)
neural$long <- coords[,1]
neural$lat <- coords[,2]
neural <- st_drop_geometry(neural)

split <- initial_split(neural, prop = 0.8) #Split
neural_train <- training(split) # Training
neural_test  <- testing(split) ; rm(neural,split) # Test


# A. --- Creamos recetas
# Primera receta 
rec_1 <- recipe(price ~ surface_imputado + bathrooms + rooms + bedrooms + codigo_zona_estrato + estudio + distancia_love_hotel, data = neural_train) %>%
  step_unknown(all_nominal_predictors())%>%
  step_impute_mean(all_numeric_predictors()) %>%
  step_impute_mode(all_nominal_predictors()) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_zv(all_predictors()) %>%
  step_normalize(all_predictors())

# Segunda receta  
rec_2 <- recipe(price ~ long + lat + surface_covered + rooms + bedrooms, data = neural_train) %>%
  step_unknown(all_nominal_predictors())%>%
  step_impute_mean(all_numeric_predictors()) %>%
  step_dummy(all_nominal_predictors()) %>% 
  step_poly(long,lat, degree = 2) %>%
  step_novel(all_nominal_predictors()) %>% 
  step_zv(all_predictors()) %>% 
  step_normalize(all_predictors())

# Tercera receta  
rec_3 <- recipe(price ~ surface_imputado + bathrooms + rooms + estrato, data = neural_train) %>%
  step_unknown(all_nominal_predictors()) %>%
  step_novel(all_nominal_predictors()) %>%
  step_impute_mean(all_numeric_predictors()) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_zv(all_predictors()) %>%
  step_normalize(all_numeric_predictors())

# Cuarta receta  (TODO)
rec_4 <- recipe(price ~ . , data = neural_train) %>%
  step_unknown(all_nominal_predictors())%>%
  step_impute_mean(all_numeric_predictors()) %>%
  step_novel(all_nominal_predictors()) %>% 
  step_dummy(all_nominal_predictors()) %>% 
  step_zv(all_predictors())%>% 
  step_normalize(all_predictors()) 

# B. --- Tuneamos la búsqueda de los parámetros de la red
elastic_net_spec <- parsnip::linear_reg(
  penalty = tune(), # penalidad lamda
  mixture = tune()) %>% # lasso o ridge 
  set_engine("glmnet")

grid_values <- grid_regular(penalty(range = c(-2,1)), levels = 50) %>%
  expand_grid(mixture = c(0, 0.25,  0.5, 0.75,  1))

set.seed(86936)
db_fold <- rsample::vfold_cv(neural_train, v = 5)

# C. --- Definir workflows
# Definimos un flujo de trabajo (workflow)
# Iniciar un flujo de trabajo utilizando 'workflow()'

workflow_1 <- workflow() %>% 
  add_recipe(rec_1) %>%   
  add_model(elastic_net_spec)  

workflow_2 <- workflow() %>%
  add_recipe(rec_2) %>%
  add_model(elastic_net_spec)

workflow_3 <- workflow() %>%
  add_recipe(rec_3) %>%
  add_model(elastic_net_spec)

workflow_4 <- workflow() %>%
  add_recipe(rec_4) %>%
  add_model(elastic_net_spec)

# D. --- FIT AND PREDICT 
metrics <- metric_set(rmse, rsq, mae)
set.seed(86936)
tune_res1 <- tune::tune_grid(
  workflow_1,         # El flujo de trabajo que contiene: receta y especificación del modelo
  resamples = db_fold,  # Folds de validación cruzada
  grid = grid_values,        # Grilla de valores de penalización
  metrics = metrics  # metrica
)

tune_res2 <- tune::tune_grid(
  workflow_2,         
  resamples = db_fold,  # Folds de validación cruzada
  grid = grid_values,        # Grilla de valores de penalización
  metrics = metrics  # metrica
)

tune_res3 <- tune::tune_grid(
  workflow_3,         
  resamples = db_fold,  # Folds de validación cruzada
  grid = grid_values,        # Grilla de valores de penalización
  metrics = metrics  # metrica
)


tune_res4 <- tune::tune_grid(
  workflow_4,         
  resamples = db_fold,  # Folds de validación cruzada
  grid = grid_values,        # Grilla de valores de penalización
  metrics = metrics  # metrica
)

# E. ---- SELECCIONAR EL MODELO 
best_penalty_1 <- select_best(tune_res1, metric = "mae")
best_penalty_2 <- select_best(tune_res2, metric = "mae")
best_penalty_3 <- select_best(tune_res3, metric = "mae")
best_penalty_4 <- select_best(tune_res4, metric = "mae")


# F. ---- FINALIZAR LOS WORKFLOWS
EN_final1 <- finalize_workflow(workflow_1, best_penalty_1)
EN_final2 <- finalize_workflow(workflow_2, best_penalty_2)
EN_final3 <- finalize_workflow(workflow_3, best_penalty_3)
EN_final4 <- finalize_workflow(workflow_4, best_penalty_4)


# G. ---- AJUSTAR MODELOS
EN_final1_fit <- fit(EN_final1, data = neural_train)
EN_final2_fit <- fit(EN_final2, data = neural_train)
EN_final3_fit <- fit(EN_final3, data = neural_train)
EN_final4_fit <- fit(EN_final4, data = neural_train)

# H. ---- PREDICCIONES SOBRE TEST
predictiones_1 <- predict(EN_final1_fit , new_data = neural_test)
predictiones_2 <- predict(EN_final2_fit , new_data = neural_test)
predictiones_3 <- predict(EN_final3_fit , new_data = neural_test)
predictiones_4 <- predict(EN_final4_fit , new_data = neural_test)


# I. ---- COMPARAR LOS PERFORMANCE
mae_1 <- neural_test %>%
  bind_cols(predictiones_1) %>%
  yardstick::mae(truth = price, estimate = .pred)

resultados <- c(mae_1[[".estimate"]])

mae_2<- neural_test %>%
  bind_cols(predictiones_2) %>%
  yardstick::mae(truth = price, estimate = .pred)
resultados <- c(mae_2[[".estimate"]])


mae_3<- neural_test %>%
  bind_cols(predictiones_3) %>%
  yardstick::mae(truth = price, estimate = .pred)
resultados <- c(mae_3[[".estimate"]])

mae_4<- neural_test %>%
  bind_cols(predictiones_4) %>%
  yardstick::mae(truth = price, estimate = .pred)
resultados <- c(mae_4[[".estimate"]])

resultados <- c(
  modelo_1 = mae_1[[".estimate"]],
  modelo_2 = mae_2[[".estimate"]],
  modelo_3 = mae_3[[".estimate"]],
  modelo_4 = mae_4[[".estimate"]]
)

resultados_df <- tibble(
  modelo = paste0("modelo_", 1:4),
  mae = c(mae_1[[".estimate"]], mae_2[[".estimate"]], mae_3[[".estimate"]], mae_4[[".estimate"]])
)

resultados_df %>% arrange(mae) # Cómo que el más chimbita es el modelo 4 :) 


# J. ---- AHORA SÍ PA ENTREGAR:
neural_cols <- c("property_id", "grupo", "price", "surface_imputado", "surface_covered",
                 "surface", "bathrooms", "rooms", "bedrooms", "codigo_zona_estrato",
                 "estudio", "estrato", "distancia_love_hotel", "distancia_bicycle_rental",
                 "distancia_funeral_hall", "distancia_bicycle_parking",
                 "distancia_toilets", "distancia_college")

neural  <- df %>%  select(all_of(neural_cols))
neural$codigo_zona_estrato <- as.factor(neural$codigo_zona_estrato)
neural$estrato <- as.factor(neural$estrato)
coords <- st_coordinates(neural)
neural$long <- coords[,1]
neural$lat <- coords[,2]
neural <- st_drop_geometry(neural)

neural_train2 <- neural %>%  select(all_of(neural_cols)) %>%  filter(grupo == "train") 
neural_test_real <- neural %>%  select(all_of(neural_cols)) %>%  filter(grupo == "test"); rm(neural) 
neural_train2 <- neural_train2 %>% select(-property_id)


rec_4 <- recipe(price ~ . , data = neural_train2) %>%
  step_unknown(all_nominal_predictors())%>%
  step_impute_mean(all_numeric_predictors()) %>%
  step_novel(all_nominal_predictors()) %>% 
  step_dummy(all_nominal_predictors()) %>% 
  step_zv(all_predictors())%>% 
  step_normalize(all_predictors()) 

elastic_net_spec <- parsnip::linear_reg(
  penalty = tune(), # penalidad lamda
  mixture = tune()) %>% # lasso o ridge 
  set_engine("glmnet")

grid_values <- grid_regular(penalty(range = c(-2,1)), levels = 50) %>%
  expand_grid(mixture = c(0, 0.25,  0.5, 0.75,  1))

set.seed(86936)
db_fold <- rsample::vfold_cv(neural_train2, v = 5)

workflow_4 <- workflow() %>%
  add_recipe(rec_4) %>%
  add_model(elastic_net_spec)

tune_res4 <- tune::tune_grid(
  workflow_4,         
  resamples = db_fold,  # Folds de validación cruzada
  grid = grid_values,        # Grilla de valores de penalización
  metrics = metrics  # metrica
)

best_penalty_4 <- select_best(tune_res4, metric = "mae")
EN_final <- finalize_workflow(workflow_4, best_penalty_4)
EN_final4_fit <- fit(EN_final, data = neural_train2)
predictiones_EN <- predict(EN_final4_fit , new_data = neural_test_real)

predictions_EN <- data.frame(
  property_id = neural_test_real$property_id,
  price  = predictiones_EN$.pred
)

write.csv(predictions_EN,"EN_MODEL4.csv", row.names = FALSE)

rm(list = setdiff(ls(), c("df","coordenadas.x.centroides","coordenadas.y.centroides",
  "tune_res4","workflow_4", "rec_4","predictions_EN"  # ← corregido: se llama predictiones_EN, no predictions_EN
  )))

################################################################################
>>>>>>> 3acc7330c9dd0ede7faff02aa2bc74ba7a14363b
# NEURAL NETWORK ---------------------------------------------------------------