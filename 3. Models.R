################################################################################
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
       recipes, # Recetas
       workflows,# Crear worklows RN
       metrics,# Evaluación de metricas para Ml
       tidymodels,#modelos
       randomforest,
       ranger,
       rlang,
       tune,
       adabag,
       caret,
       gmb,
       xgboost,
       spatialsample,
       sf)
install.packages("keras")
library(reticulate)
reticulate::virtualenv_create("r-reticulate", python = install_python())
library(keras)
install_keras(envname = "r-reticulate")

load("Datos_limpios.RData")
colnames(df)
# LINEAR REGRESSION  -----------------------------------------------------------


# ELASTIC NET -----------------------------------------------------------------


# CARTS ------------------------------------------------------------------------


# RANDOM FOREST-----------------------------------------------------------------
vars<- c("price", "year", "surface_imputado", "rooms",
                     "bedrooms", "bathrooms", "property_type",
                     "cocina_lujo", "cocina_estandar", "parqueadero", "terraza",
                     "sala_comedor", "patio_lavanderia", "walkin_closet", "estudio",
                     "closet", "saloncomunal_recepcion", "seguridad", "piso",
                     "lujos", "remodelado",  "distancia_commercial",
                     "distancia_bank", "distancia_bus_station",
                     "distancia_cafe",  "distancia_college",
                     "distancia_hospital", "distancia_marketplace","codigo_barrio","valor_comercial")

RF_data  <- df %>% filter(grupo == "train") %>%  select(all_of(vars)) 
RF_data <- RF_data|> st_drop_geometry()

# A. Receta
rec_rf <- recipe(price ~ ., data = RF_data) %>%
  step_unknown(all_nominal_predictors()) %>%
  step_impute_mean(all_numeric_predictors()) %>%
  step_novel(all_nominal_predictors()) %>%
  step_zv(all_predictors()) 

prepped_rf <- prep(rec_rf, training = RF_data)
RF_data<- bake(prepped_rf, new_data = RF_data)

set.seed(123)
split_rf <- initial_split(RF_data, prop = 0.8)
RF_train <- training(split_rf)
RF_test <- testing(split_rf)

# B. Entrenar y evaluar cada modelo
formulas <- list(
  modelo1 = price ~ surface_imputado + bathrooms + rooms + bedrooms + estudio + parqueadero + distancia_college,
  modelo2 = price ~ surface_imputado + bathrooms + bedrooms + parqueadero + distancia_hospital + distancia_marketplace,
  modelo3 = price ~ surface_imputado + bathrooms + bedrooms + parqueadero + distancia_hospital + distancia_marketplace + codigo_barrio +valor_comercial,
  modelo4 = price ~ surface_imputado + rooms + estudio + walkin_closet + cocina_lujo + distancia_cafe,
  modelo5 = price ~ surface_imputado + bathrooms + patio_lavanderia + saloncomunal_recepcion + distancia_bank + distancia_bus_station,
  modelo6= price ~ surface_imputado + rooms + bathrooms + property_type + piso + codigo_barrio + valor_comercial + walkin_closet + lujos
)

modelos_rf <- list()
resultados <- data.frame(modelo = character(), MAE = numeric(), stringsAsFactors = FALSE)
for (nombre in names(formulas)) {
  rf_fit <- ranger(
    formula = formulas[[nombre]],
    data = RF_train,
    num.trees = 500,
    mtry = floor(sqrt(ncol(RF_train))),
    min.node.size = 5,
    importance = "impurity"
  )
  
  modelos_rf[[nombre]] <- rf_fit
  
  pred <- predict(rf_fit, data = RF_test)$predictions
  
  mae_val <- mean(abs(RF_test$price - pred))
  
  resultados <- rbind(resultados, data.frame(modelo = nombre, MAE = mae_val))
}


resultados <- resultados[order(resultados$MAE), ]
print(resultados)

#Tunear el mejor modelo
grid_rf <- expand.grid(
  mtry = c(3, 5, 7, floor(sqrt(ncol(RF_train)))),
  splitrule = c("variance", "extratrees"),  # Include splitrule!
  min.node.size = c(1, 5, 10)
)

ctrl_rf <- trainControl(
  method = "cv",
  number = 5,
  verboseIter = TRUE
)

rf_tuned <- train(
  price ~ surface_imputado + bathrooms + bedrooms + parqueadero + distancia_hospital + distancia_marketplace + codigo_barrio + valor_comercial,
  data = RF_train,
  method = "ranger",
  trControl = ctrl_rf,
  tuneGrid = grid_rf,
  importance = "impurity"
)


best_params <- rf_tuned$bestTune
print(best_params)


rf_final <- ranger(
  formula = price ~ surface_imputado + bathrooms + bedrooms + parqueadero + distancia_hospital + distancia_marketplace + codigo_barrio + valor_comercial,
  data = RF_train,
  num.trees = 500,
  mtry = best_params$mtry,
  min.node.size = best_params$min.node.size,
  importance = "impurity"
)

#C. Predicción
df_test <- df %>% filter(grupo == "test")
df_test_preprocesado <- bake(prepped_rf, new_data = df_test)

predicciones_RF <- predict(rf_final, data = df_test_preprocesado)$predictions

predicciones_RF <- data.frame(
  property_id = df_test$property_id,
  price = predicciones_RF
)

write.csv(predicciones_RF, "RF_Tuned.csv", row.names = FALSE)

# XGBOOST-----------------------------------------------------------------


# GRADIENT BOOSTING ------------------------------------------------------------
#A. PREPROCESAMIENTO
gb_data <- df|> st_drop_geometry()

gb_train <- gb_data %>% filter(grupo == "train") %>%
  select(price, surface_imputado, bathrooms, rooms, 
         bedrooms, estudio, parqueadero, distancia_college, 
         distancia_hospital, distancia_marketplace, piso, 
         codigo_barrio, valor_comercial, property_type, 
         walkin_closet, lujos)
gb_train <- gb_train %>% na.omit()

class(gb_train)

rec_gb <- recipe(price ~ ., data = gb_train) |>
  step_normalize(all_numeric_predictors()) |>  
  step_dummy(all_nominal_predictors()) |>  
  step_novel(all_nominal_predictors()) |> 
  step_zv(all_predictors())

prepped_gb <- prep(rec_gb, training = gb_train)
gb_train <- bake(prepped_gb, new_data = gb_train)
colnames(gb_train)
#B. Modelos
formulas <- list(
  modelo1 = price ~ surface_imputado + bathrooms + rooms + bedrooms + parqueadero_Sí + distancia_college,
  modelo2 = price ~ surface_imputado + bathrooms + bedrooms + parqueadero_Sí + distancia_hospital + distancia_marketplace + codigo_barrio + valor_comercial,
  modelo3 = price ~ surface_imputado + rooms + bathrooms + property_type_Casa + piso + codigo_barrio + valor_comercial + walkin_closet_Sí + lujos_Sí
)
grid_gbm <- expand.grid(
  interaction.depth = c(3, 5, 7),  
  n.trees = c(100, 300, 500),  
  shrinkage = c(0.01, 0.1, 0.3),  
  n.minobsinnode = c(5, 10, 20)  
)
#B. EVALUACIÓN DE MODELOS
gb_train_sf <- st_as_sf(
  gb_train,
  coords = c("lon", "lat"), 
  crs = 4326
)

set.seed(88)
block_folds <- spatial_block_cv(gb_train_sf, v = 5)  # 5 bloques espaciales

resultados_gbm <- data.frame(modelo = character(), Fold = integer(), MAE = numeric(), stringsAsFactors = FALSE)

for (nombre in names(formulas)) {
  for (i in seq_along(block_folds$splits)) {
    split <- block_folds$splits[[i]]
    
    train_data <- analysis(split)  
    test_data <- assessment(split)  
    
    gbm_cv <- train(
      formulas[[nombre]],
      data = train_data,
      method = "gbm",
      metric = "MAE",
      verbose = FALSE
    )
    
    predicciones <- predict(gbm_cv, newdata = test_data)
    mae_val <- mean(abs(predicciones - test_data$price))
    
    resultados_gbm <- rbind(resultados_gbm, data.frame(modelo = nombre, Fold = i, MAE = mae_val))
  }
}


resultados_gbm <- resultados_gbm[order(resultados_gbm$MAE), ]
print(resultados_gbm)

mejor_modelo <- resultados_gbm %>%
  group_by(modelo) %>%
  summarise(MAE_promedio = mean(MAE)) %>%
  arrange(MAE_promedio)

print(mejor_modelo)

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
# NEURAL NETWORK ---------------------------------------------------------------
#A. Preparación de datos
neural_data <- df %>%
  filter(grupo == "train") %>%
  select(price, surface_imputado, bathrooms, bedrooms,
         distancia_hospital, distancia_marketplace, codigo_barrio, valor_comercial)
neural_data <- neural_data %>% na.omit()

X <- neural_data %>% select(-price)
X <- X %>% st_drop_geometry()
y <- neural_data$price
X_scaled <- scale(X)
y_scaled <- scale(y)[,1]

X_matrix <- as.matrix(X_scaled)
y_vector <- as.numeric(y_scaled)


set.seed(2)
train_indices <- sample(1:nrow(X_matrix), 0.8 * nrow(X_matrix))

X_train <- X_matrix[train_indices, ]
X_val <- X_matrix[-train_indices, ]
y_train <- y_vector[train_indices]
y_val <- y_vector[-train_indices]

cat("Datos de entrenamiento:", nrow(X_train), "filas\n")
cat("Datos de validación:", nrow(X_val), "filas\n")

# B. CONSTRUCCIÓN DE LA RED NEURONAL
model <- keras_model_sequential() %>%
  layer_dense(units = 128, activation = "relu", input_shape = ncol(X_train)) %>%
  layer_dropout(rate = 0.3) %>% 
  layer_dense(units = 64, activation = "relu") %>%
  layer_dropout(rate = 0.2) %>%
  layer_dense(units = 32, activation = "relu") %>%
  layer_dense(units = 1, activation = "linear")  # Capa de salida para regresión
summary(model)

model %>% compile(
  optimizer = optimizer_adam(learning_rate = 0.001),
  loss = "mse",  # Error cuadrático medio para regresión
  metrics = c("mae")  # Error absoluto medio
)

early_stopping <- callback_early_stopping(
  monitor = "val_loss",
  patience = 15,
  restore_best_weights = TRUE
)

reduce_lr <- callback_reduce_lr_on_plateau(
  monitor = "val_loss",
  factor = 0.5,
  patience = 10,
  min_lr = 0.00001
)

history <- model %>% fit(
  X_train, y_train,
  epochs = 100,
  batch_size = 32,
  validation_data = list(X_val, y_val),
  callbacks = list(early_stopping, reduce_lr),
  verbose = 1
)

predictions_scaled <- model %>% predict(X_val)

mean_price <- attr(scale(neural_data$price), "scaled:center")
sd_price <- attr(scale(neural_data$price), "scaled:scale")

predictions <- predictions_scaled * sd_price + mean_price
actual_values <- y_val * sd_price + mean_price

mae <- mean(abs(predictions - actual_values))
cat("MAE:", round(mae, 2), "\n")


#C. Predicción
test_data <- df %>%
  filter(grupo == "test") %>%
  select(property_id, surface_imputado, bathrooms, bedrooms,
         distancia_hospital, distancia_marketplace, codigo_barrio, valor_comercial) %>%
  na.omit()

test_data <- test_data %>% st_drop_geometry()

X_test_matrix <- as.matrix(test_data[, -1])  
X_test_scaled <- scale(X_test_matrix, center = attr(X_scaled, "scaled:center"),
                       scale = attr(X_scaled, "scaled:scale"))

predictions_scaled <- model %>% predict(X_test_scaled)
predictions <- predictions_scaled * sd_price + mean_price
submission <- data.frame(property_id = test_data$property_id, price = predictions)
write.csv(submission, "RED.csv", row.names = FALSE)
