```markdown
# Pacotes -----------------------------------------------------------------
  
  library(tidyverse)
  library(lubridate)    
  library(data.table)
  library(Amelia)
  library(imbalance)
  library(ROSE)
  library(randomForest)
  library(caret)
  library(C50)
  library(caTools)
  library(lightgbm)
  library(modelr)
  library(broom)
  library(e1071)
  library(rpart)
  library(rpart.plot)
  library(gmodels)
  library(ROCR)
  library(pROC)
  library(class)
  library(xgboost)
  library(Matrix)
  library(ElemStatLearn)
  library(class)
  library(DiagrammeR)
  options(stringoAsFacotr = FALSE)
  
  # Pasta de trabalho -------------------------------------------------------
  
  setwd("SuaPastaDeTrabalhoAqui")
  
  # Variáveis ---------------------------------------------------------------
  
  # Train dataset
  # ip: ip address of click.
  # app: app id for marketing.
  # device: device type id of user mobile phone (e.g., iphone 6 plus, iphone 7, huawei mate 7, etc.)
  # os: os version id of user mobile phone
  # channel: channel id of mobile ad publisher
  # click_time: timestamp of click (UTC)
  # attributed_time: if user download the app for after clicking an ad, this is the time of the app download
  # is_attributed: the target that is to be predicted, indicating the app was downloaded
  # Note that ip, app, device, os, and channel are encoded.
  
  # Test dataset is similar, with the following differences:
  # click_id: reference for making predictions
  # is_attributed: not included
  
  # Dataset -----------------------------------------------------------------
  # Abrir dataset
  df <- read_csv("train_sample.csv")
  
  # Imprimir o início do dataset
  head(df)
  
  # Verificar as dimensões do dataset
  dim(df)
  
  # Visualizar o dataset
  View(df)
  
  # Resumo estatístico do dataset
  summary(df)
  
  # Avaliar dados NA --------------------------------------------------------
  sapply(df, function(x) sum(is.na(x)))
  
  # Avaliar dados NA com figura
  missmap(df, main = "Valores NA")
  
  # Novo dataset com var categórica -----------------------------------------
  # Transformar os dados em categóricos
  # Adicionar um dado com o dia da semana
  # Excluir a coluna de data do download e do click pois não acrescentam info
  df1 <- df %>% 
    mutate(app = as.factor(app), 
           device = as.factor(device), 
           os = as.factor(os), 
           channel = as.factor(channel), 
           is_attributed = as.factor(is_attributed),
           click_time = as_datetime(click_time),
           wday = as.factor(wday(click_time))) %>% 
    select(-c(attributed_time, click_time))
  
  # Resumo estatístico do novo dataset
  summary(df1)
  
  # Ver a dimensão do novo dataset
  dim(df1)
  
  # Visualizar o novo dataset
  View(df1)
  
  # Verificar os valores NA no novo dataset
  missmap(df1, main = "Valores NA")
  
  # Visualizar variáveis ---------------------------------------------------
  # Plotar as var categóricas em ordem
  categoricas <- c("app", "device", "os", "channel", "is_attributed", "wday")
  
  categoricas_ordem <- function(x) {
    x <- x %>%
      mutate(app = fct_infreq(app)) %>%
      mutate(app = fct_lump(app, n = 10)) %>%
      mutate(device = fct_infreq(device)) %>%
      mutate(device = fct_lump(device, n = 10)) %>%
      mutate(os = fct_infreq(os)) %>%
      mutate(os = fct_lump(os, n = 10)) %>%
      mutate(channel = fct_infreq(channel)) %>%
      mutate(channel = fct_lump(channel, n = 10)) %>%
      mutate(is_attributed = fct_infreq(is_attributed)) %>%
      mutate(is_attributed = fct_lump(is_attributed, n = 10)) %>% 
      mutate(wday = fct_infreq(wday)) %>%
      mutate(wday = fct_lump(wday, n = 5))
  }
  
  temp <- categoricas_ordem(df1)
  
  graficos_cat <- function(x) {
    temp %>%
      ggplot() +
      geom_bar(aes_string(x = x))
  }
  
  lapply(categoricas, graficos_cat)
  
  # Spliting o dataset em treino e teste -------------------------------------
  indice <- createDataPartition(df1$is_attributed, p = 0.70, list = FALSE)
  df_treino <- df1[indice,]
  df_teste <- df1[-indice,]
  
  # Coletar os labels da variável target
  df_treino_labels <- df_treino$is_attributed
  df_teste_labels <- df_teste$is_attributed
  
  # Verificar a dimensão dos datasets de treino e teste
  dim(df_treino)
  dim(df_teste)
  
  # Balanceamento do target -------------------------------------------------
  # Verificar o balanceamento da variável target
  imbalanceRatio(as.data.frame(df_treino), classAttr = "is_attributed")
  
  # Alternativa para verificar o balancemaento da variável target
  prop.table(table(df_treino$is_attributed)) * 100 
  
  # Aplicar o balancemaento
  df_treino_bal <- ROSE(is_attributed ~ ., 
                          data = df_treino, seed = 1, p= 0.5, 
                          hmult.majo=1, hmult.mino=1, 
                          subset=options("subset")$subset, 
                          na.action=options("na.action")$na.action)$data
  
  # Verificando o balancemaneto final da variável target
  prop.table(table(df_treino_bal$is_attributed)) * 100
  
  # Verificar a dimensão dos dados
  dim(df_treino_bal)
  dim(df_teste)

# Modelos -------------------------------------------------------------
  # Treinar modelo de árvore de decisão
  modelo1 <- rpart(is_attributed ~ .,
                  data = df_treino_bal,
                  control = rpart.control(cp = 0.3))
  
  # Plotar árvore de decisão
  rpart.plot(modelo1)
  
  #  Imprimir o cp do modelo
  printcp(modelo1)
  
  # Plotar gráfico do cp
  plotcp(modelo1)
  
  # Imprimir resumo do modelo
  summary(modelo1)
  
  # Previsões do modelo de classificação
  pred1 <- predict(modelo1, df_teste, type = "class")
  
  # Matriz de confusão do modelo
  matrix1 <- confusionMatrix(pred1, df_teste$is_attributed, positive = "1")
  matrix1
  c(matrix1$overall["Accuracy"],
    matrix1$byClass["Sensitivity"],
    matrix1$byClass["Specificity"],
    matrix1$byClass["Prevalence"])
  
  # Previsões baseado em proabilidade de pertencer a classe
  prob1 <- predict(modelo1, df_teste, type = "prob")
  
  # Imprimir a curva roc e imprimir o auc
  r <- multiclass.roc(df_teste$is_attributed, prob1[,2], percent = TRUE)
  roc <- r[['rocs']]
  r1 <- roc[[1]]
  
  # Plotar o gráfico roc
  plot.roc(r1,
           print.auc=TRUE,
           auc.polygon=TRUE,
           grid=c(0.1, 0.2),
           grid.col=c("green", "red"),
           max.auc.polygon=TRUE,
           auc.polygon.col="lightblue",
           print.thres=TRUE,
           main= 'ROC Curve')
  
  # AUC
  p <- predict(modelo1, df_teste, type = "vector")
  pr <- prediction(p, df_teste$is_attributed)
  prf <- performance(pr, measure = "tpr", x.measure = "fpr")
  plot(prf)
  auc <- performance(pr, measure = "auc")
  auc <- auc@y.values[[1]]
  auc
  
  # Média de erro
  error1 <- mean(df_teste$is_attributed != pred1)
  error1
 
# XGBoost  ---------------------------------------------------------
  # Modelo XGBoost
  # Transformar as var dos dados de treino em numéricas
  str(df_treino_bal)
  df_treino_bal$ip = as.numeric((df_treino_bal$ip))
  df_treino_bal$channel = as.numeric((df_treino_bal$channel))
  
  # Tranformar as var fatorias em matriz esparsa
  trainm <- sparse.model.matrix(is_attributed ~ - ip - channel + app + device + os + wday, 
                                data = df_treino_bal)
  head(trainm)
  
  # Label de var do target
  train_label <- as.numeric(as.character(df_treino_bal$is_attributed))
  
  # Matriz para treino do modelo
  train_matrix <- xgb.DMatrix(data = as.matrix(trainm), 
                              label = train_label)
  str(df_teste)
  
  # Transformar as var dos dados de teste em numéricas
  df_teste$ip = as.numeric((df_teste$ip))
  df_teste$channel = as.numeric((df_teste$channel))
  
  testm <- sparse.model.matrix(is_attributed ~ - ip - channel + app + device + os + wday
                               , data = df_teste)
  head(testm)
  
  # Label de var do target dos dados teste
  test_label <- as.numeric(as.character(df_teste$is_attributed))
  
  # Matriz para teste
  test_matrix <- xgb.DMatrix(data = as.matrix(testm), label = test_label)
  
  # Modelo
  nc <- length(unique(test_label))
  xgb_params <- list("objective" = "multi:softprob",
                     "eval_metric" = "mlogloss",
                     "num_class" = nc)
  watchlist <- list(train = train_matrix, test = test_matrix)
  modelo2 <- xgb.train(params = xgb_params,
                         data = train_matrix,
                         nrounds = 300,
                         watchlist = watchlist,
                         eta = 0.3,
                         max.depth = 5,
                         gamma = 1,
                         subsample = 1,
                         colsample_bytree = 1)
  
  # Modelo em df
  modelo2 <- data.frame(modelo2$evaluation_log)
  
  # Plotar gráfico do modelo treinado
  plot(modelo2$iter, modelo2$train_mlogloss, col = 'blue')
  lines(modelo2$iter, modelo2$test_mlogloss, col = 'red')
  
  # Valor minimo da taxa de aprendizagem
  minimo <- min(modelo2$test_mlogloss)
  modelo2[modelo2$test_mlogloss == minimo,]
  
  # Modelo com melhor eta
  modelo2 <- xgb.train(params = xgb_params,
                         data = train_matrix,
                         nrounds = 2,
                         watchlist = watchlist,
                         eta = minimo,
                         max.depth = 3,
                         gamma = 0,
                         subsample = 1,
                         colsample_bytree = 1)
  
  # Var importntes para o modelo
  imp <- xgb.importance(colnames(train_matrix), model = modelo2)
  print(imp)
  xgb.plot.importance(imp)
  
  # Dump
  xgb.dump(modelo2, with_stats = T)
  
  # Plot do modelo
  xgb.plot.tree(model = modelo2)
  
  # Previsão do modelo
  pred2 <- predict(modelo2, newdata = test_matrix)
  
  # Ajustando a matriz
  pred2 <- matrix(pred2, nrow = nc, ncol = length(pred2)/nc) %>%
    t() %>%
    data.frame() %>%
    mutate(label = test_label, max_prob = max.col(., "last")-1)
  
  # MAtriz de confusão do modelo
  matrix2 <- confusionMatrix(as.factor(pred2$max_prob), as.factor(pred2$label), 
                             positive = "1")
  matrix2
  c(matrix2$overall["Accuracy"],
    matrix2$byClass["Sensitivity"],
    matrix2$byClass["Specificity"],
    matrix2$byClass["Prevalence"])

# Naive Bayes -------------------------------------------------------------
  # Treinar modelo Naive Bayes
  # suavização laplace
  modelo3 <- naiveBayes(is_attributed ~ ., data = df_treino_bal, laplace = 2)
  summary(modelo3)
  
  # Previsão do modelo
  pred3 <- predict(modelo3, df_teste)
  
  # MAtriz de confusão
  matrix3 <- confusionMatrix(df_teste$is_attributed, pred3)
  matrix3
  
  c(matrix3$overall["Accuracy"],
    matrix3$byClass["Sensitivity"],
    matrix3$byClass["Specificity"],
    matrix3$byClass["Prevalence"])
  
  # MAtriz de confusão 2
  CrossTable(pred3, 
             df_teste$is_attributed,
             prop.chisq = FALSE, 
             prop.t = FALSE, 
             prop.r = FALSE,
             dnn = c('Previsto', 'Observado'))
  
  # Previsões baseado em proabilidade de pertencer a classe
  prob3 <- predict(modelo3, df_teste, type = "raw")
  
  # Imprimir a curva roc e imprimir o auc
  r3 <- multiclass.roc(df_teste$is_attributed, prob3[,2], percent = TRUE)
  roc3 <- r3[['rocs']]
  r3 <- roc3[[1]]
  
  # Plotar o gráfico roc
  plot.roc(r3,
           print.auc=TRUE,
           auc.polygon=TRUE,
           grid=c(0.1, 0.2),
           grid.col=c("green", "red"),
           max.auc.polygon=TRUE,
           auc.polygon.col="lightblue",
           print.thres=TRUE,
           main= 'ROC Curve')
  
  # Média de erro
  error3 <- mean(df_teste$is_attributed != pred3)
  error3
  
# KNN ---------------------------------------------------------------------
  # Transformar as variáveis em númerica
  list_num <- c("ip", "app", "device", "os", "channel", "wday")
  
  for (i in list_num){
    df_treino_bal[[i]] <- as.numeric(df_treino_bal[[i]])
  }
  
  for (i in list_num){
    df_teste[[i]] <- as.numeric(df_teste[[i]])
  }
  
  # Normalizar as variáveis
  scale.features <- function(df, variables){
    for (variable in variables){
      df[[variable]] <- scale(df[[variable]], center = T, scale = T)
    }
    return(df)
  }
  
  dados_treino_scaled <- scale.features(df_treino_bal, list_num)
  dados_teste_scaled <- scale.features(df_teste, list_num)
  
  # Escolher o valor de k
  previsoes = NULL
  perc.erro = NULL
  
  for(i in 1:20){
    set.seed(1)
    previsoes = knn(train = dados_treino_scaled[-c(6)], 
                    test = dados_teste_scaled[-c(6)], 
                    cl = dados_treino_scaled$is_attributed, k = i)
    perc.erro[i] = mean(dados_teste_scaled$is_attributed != previsoes)
  }
    perc.erro
  
  # Plotar grafico do valor de k
  k.values <- 1:20
  error.df <- data.frame(perc.erro,k.values)
  ggplot(error.df,aes(x = k.values, y = perc.erro)) + geom_point()+ geom_line(lty = "dotted", color = 'red')
  
  # Treinar o modelo
  ctrl <- trainControl(method = "repeatedcv", 
                       repeats = 3)
  
  # Criação do modelo
  modelo4 <- train(is_attributed ~ ., 
                   data = dados_treino_scaled, 
                   trControl = ctrl, 
                   tuneLength = 20)
  
  modelo4
  
  summary(modelo4)
  
  # Plotar o modelo
  plot(modelo4)
  
  # Previsão do modelo
  pred4 <- predict(modelo4, dados_teste_scaled[-c(6)])
  
  # MAtriz de confusão
  matrix4 <- confusionMatrix(pred4, df_teste$is_attributed)
  matrix4
  
  c(matrix4$overall["Accuracy"],
    matrix4$byClass["Sensitivity"],
    matrix4$byClass["Specificity"],
    matrix4$byClass["Prevalence"])
  
  # Média de erro
  error4 <- mean(df_teste$is_attributed != pred4)
  error4

# Random Forest -----------------------------------------------------------
  # Treinar o modelo
  modelo5 <- randomForest(is_attributed ~ ., data = df_treino_bal, proximity=FALSE) 
  
  # Visualizar o resultado do modelo
  modelo5
  
  # Previsão do modelo
  pred5 <- predict(modelo5, df_teste)
  
  # MAtrix de confusão
  matrix5 <- confusionMatrix(pred5, df_teste$is_attributed)
  matrix5
  
  c(matrix5$overall["Accuracy"],
    matrix5$byClass["Sensitivity"],
    matrix5$byClass["Specificity"],
    matrix5$byClass["Prevalence"])

  # Média de erro
  error5 <- mean(df_teste$is_attributed != pred5)
  error5
  
# Escolha do melhor modelo ------------------------------------------------
     
  Accuracy <- c(matrix1$overall["Accuracy"], matrix2$overall["Accuracy"], 
                matrix3$overall["Accuracy"], matrix4$overall["Accuracy"], 
                matrix5$overall["Accuracy"])
  
  Sensitivity <- c(matrix1$byClass["Sensitivity"], matrix2$byClass["Sensitivity"], 
                   matrix3$byClass["Sensitivity"], matrix4$byClass["Sensitivity"], 
                   matrix5$byClass["Sensitivity"])
  
  Specificity <- c(matrix1$byClass["Specificity"], matrix2$byClass["Specificity"], 
                   matrix3$byClass["Specificity"], matrix4$byClass["Specificity"], 
                   matrix5$byClass["Specificity"])

  Prevalence <- c(matrix1$byClass["Prevalence"], matrix2$byClass["Prevalence"], 
                  matrix3$byClass["Prevalence"], matrix4$byClass["Prevalence"], 
                  matrix5$byClass["Prevalence"])
  
  Errors <- c(error1, "NA", error3, error4, error5) 
  
  tabela_final <- data.frame(Accuracy, Sensitivity, Specificity, Prevalence, Errors,
                             row.names = c("rpart", "xgboost", "naive byes",
                                           "knn", "randomForest"))
  
  tabela_final
  
  # Minha escolha final baseado na sensitividade (razão de true positives)
  # seria o modelo naive bayes
  ```
