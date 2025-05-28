library(pacman) 

# Cargar las librerías listadas e instalarlas en caso de ser necesario
p_load(tidyverse, # Manipular dataframes
       gridExtra, # Para graficar en grid
       rio, # Import data easily
       plotly, # Gráficos interactivos
       leaflet, # Mapas interactivos
       tmaptools, # geocode_OSM()
       sf, # Leer/escribir/manipular datos espaciales
       osmdata, # Get OSM's data 
       spatialsample,
       openxlsx,
       tidymodels) #para modelos de ML

install.packages("vip")
library(vip)

# 1. Carga de los datos ----

load("~/Taller-3/Datos_limpios.RData")

rm(geometria.osm, coordenadas.x.centroides, coordenadas.y.centroides)

df <- df |> 
  mutate(color = case_when(grupo == "train" ~ "#2A9D8F", 
                           grupo == "test" ~ "#F4A261"))

leaflet() |> 
  addTiles() |> 
  setView(lng = mean(raw$lon), lat = mean(raw$lat), zoom = 12.1) |> 
  addCircles(lng = raw$lon, lat = raw$lat, color = raw$color)

df <- as.data.frame(df)

train = df |> 
  filter(grupo == "train")

test = df |> 
  filter(grupo == "test")

# Entrenamiento ----- 

my_tree <- decision_tree(cost_complexity = tune(), 
                         tree_depth = tune(),
                         min_n = tune()) |> 
  set_engine("rpart") |> 
  set_mode("regression")

grid_values <-  grid_regular(
  cost_complexity(range = c(-4, -1)),  # log10 escala, ej. 1e-4 a 1e-1
  tree_depth(range = c(2, 10)),
  min_n(range = c(5, 20)),
  levels = 4                           # número de puntos por parámetro
)

# Primera receta
rec_1 <- recipe(price ~ surface_imputado + bedrooms + property_type + distancia_commercial, data = train) |> 
  step_dummy(all_nominal_predictors()) |>  # crea dummies para las variables categóricas
  step_novel(all_nominal_predictors()) |>   # para las clases no antes vistas en el train. 
  step_zv(all_predictors()) |>   #  elimina predictores con varianza cero (constantes)
  step_normalize(all_predictors())  # normaliza los predictores. 

# Segunda receta 

rec_2 <- recipe(price ~ surface_imputado + rooms + bathrooms + property_type + piso + codigo_barrio + valor_comercial + walkin_closet + lujos, data = as.data.frame(train)) |>
  step_dummy(all_nominal_predictors()) |> 
  step_novel(all_nominal_predictors()) |> 
  step_zv(all_predictors()) |>   
  step_normalize(all_predictors())


workflow_1 <- workflow() |> 
  # Agregar la receta de preprocesamiento de datos. En este caso la receta 1
  add_recipe(rec_1) |>
  # Agregar la especificación del modelo de regresión Elastic Net
  add_model(my_tree)

## Lo mismo con la receta rec_2 

workflow_2 <- workflow() |>
  add_recipe(rec_2) |>
  add_model(my_tree)

train_sf <- st_as_sf(train)

# Validación ----

set.seed(86936)
block_folds <- spatial_block_cv(train_sf, v = 5)
block_folds

autoplot(block_folds)

set.seed(86936)
tune_res1 <- tune_grid(
  workflow_1,         # El flujo de trabajo que contiene: receta y especificación del modelo
  resamples = block_folds,  # Folds de validación cruzada espacial
  grid = grid_values,        # Grilla de valores de penalización
  metrics = metric_set(mae)  # metrica
)

collect_metrics(tune_res1)

best_tune_res1 <- select_best(tune_res1, metric = "mae")

set.seed(86936)

tune_res2 <- tune_grid(
  workflow_2,         # El flujo de trabajo que contiene: receta y especificación del modelo
  resamples = block_folds,  # Folds de validación cruzada
  grid = grid_values,        # Grilla de valores de penalización
  metrics = metric_set(mae)  # metrica
)

collect_metrics(tune_res2) |> arrange(mean)

best_tune_res2 <- select_best(tune_res2, metric = "mae")

# Finalizar el flujo de trabajo 'workflow' con el mejor valor de parametros
res1_final <- finalize_workflow(workflow_1, best_tune_res1)

# Ajustar el modelo  utilizando los datos de entrenamiento
res2_final <- finalize_workflow(workflow_2, best_tune_res2)

CART_final1_fit <- fit(res1_final, data = train)
CART_final2_fit <- fit(res2_final, data = train)

CART_0.0001_10_20 <- CART_final1_fit  |> 
  augment(new_data = test) |> 
  select(property_id, .pred) |> 
  rename("price" = ".pred")

CART_0.0001_10_15 <- CART_final2_fit  |> 
  augment(new_data = test) |> 
  select(property_id, .pred) |> 
  rename("price" = ".pred")

write_csv(CART_0.0001_10_20, file = "CART_0_0001_10_20.csv")
write_csv(CART_0.0001_10_15, file = "CART_0_0001_10_15.csv")


## XGBoost ----- 

rec_xgb <- recipe(price ~ surface_imputado + rooms + bathrooms + 
                    property_type + piso + codigo_localidad + valor_comercial + walkin_closet + lujos, data = train) |> 
  step_normalize(all_numeric_predictors()) |> 
  step_dummy(all_nominal_predictors()) |> 
  step_novel(all_nominal_predictors()) |> 
  step_zv(all_predictors())
  
my_prep <- prep(rec_xgb, training = train)
train_pr <- bake(my_prep, new_data = NULL)

my_xgb <- boost_tree(
  trees = tune(),
  tree_depth = tune(),
  learn_rate = tune(),
  loss_reduction = tune(),  
  mtry = tune()
) |> 
  set_engine("xgboost") |>
  set_mode("regression")

grid_xgb <- grid_space_filling(
  trees(range = c(200, 1000)),
  tree_depth(range = c(3, 10)),
  learn_rate(range = c(0.01, 0.3)),
  loss_reduction(range = c(0, 10)),
  mtry(range = c(2, 10)),
  size = 20  # número de combinaciones a probar
)

workflow_xgb <- workflow() |>
  add_recipe(rec_xgb) |>
  add_model(my_xgb)

set.seed(86936)
tune_xgb <- tune_grid(
  workflow_xgb,
  grid = grid_xgb,
  resamples = block_folds,
  metrics = metric_set(mae),
  control = control_grid(
    verbose = TRUE,          # muestra progreso en consola
    save_pred = TRUE,        # guarda predicciones de CV (opcional)
    save_workflow = TRUE     # guarda los workflows usados (opcional)
  )
)

best_tune_xgb <- select_best(tune_xgb, metric = "mae")

collect_metrics(tune_xgb) |> arrange(mean)

# Finalizar el flujo de trabajo 'workflow' con el mejor valor de parametros
res_xgb <- finalize_workflow(workflow_xgb, best_tune_xgb)

train[33989,149] = 13

xgb_fit <- fit(res_xgb, data = train)

XGB_model <- xgb_fit  |> 
  augment(new_data = test) |> 
  select(property_id, .pred) |> 
  rename("price" = ".pred")

write_csv(XGB_model, file = "XGB_4_747_3_1_06.csv")





