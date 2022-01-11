options(stringoAsFacotr = FALSE)
library(tidyr)
library(data.table)
library(ggplot2)
library(scales)
library(treemap)
library(readr)
library(dplyr)
library(xgboost)
library(stringr)
library(tm)
library(SnowballC)
library(skmeans)
library(Metrics)
library(caret)

train <- fread("D/DataSience/CientistaDeDados/BigDataAnalytics_R_Azure/ProjetoscomFeedback/Projeto02/train.csv")

test <- fread("D/DataSience/CientistaDeDados/BigDataAnalytics_R_Azure/ProjetoscomFeedback/Projeto02/test.csv")

town <- fread("D/DataSience/CientistaDeDados/BigDataAnalytics_R_Azure/ProjetoscomFeedback/Projeto02/town_state.csv")

client <- read_csv("D/DataSience/CientistaDeDados/BigDataAnalytics_R_Azure/ProjetoscomFeedback/Projeto02/cliente_tabla.csv")

product <- read_csv("D/DataSience/CientistaDeDados/BigDataAnalytics_R_Azure/ProjetoscomFeedback/Projeto02/producto_tabla.csv")

# Semana — Week number (From Thursday to Wednesday)
# Agencia_ID — Sales Depot ID
# Canal_ID — Sales Channel ID
# Ruta_SAK — Route ID (Several routes = Sales Depot)
# Cliente_ID — Client ID
# NombreCliente — Client name
# Producto_ID — Product ID
# NombreProducto — Product Name
# Venta_uni_hoy — Sales unit this week (integer)
# Venta_hoy — Sales this week (unit: pesos)
# Dev_uni_proxima — Returns unit next week (integer)
# Dev_proxima — Returns next week (unit: pesos)
# Demanda_uni_equil — Adjusted Demand (integer) (This is the target you will predict
                                               
# Quantas são as demandas feitas por dia da semana?
# Há diferença das demandas realizadas de acordo com o dia da semana?
# Avaliar apenas 10% do dataset de treino
# Resposta: Nao há diferenças significativas de demanda entre os dias da semana
ggplot(train %>% sample_frac(0.1)) +
  geom_histogram(aes(Semana), color="black", fill="red", alpha=0.5)+
  scale_x_continuous(breaks=1:10, name="Dia da semana")+
  scale_y_continuous(name="Unidades")

# Existe diferença de damanda (unidades de produtos) entre os depósitos de vendas?
# Organizar df agrupado pelos depósitos de vendas
# E com a soma das unidades demandada e soma do valor demandado
# O mesmo com as unidades devolvidas e valor devolvido
# Somar o número de unidade demandado na próxima semana
# Criar coluna com o valor líquido da venda (valor demandado - valor devolvido)
# E criar coluna com a taxa de retorno (unidades devolvidas dividido pelas unidades demandadas somado das unidades devolvidas)
agencias <- train %>%
  group_by(Agencia_ID) %>%
  summarise(Units = sum(Venta_uni_hoy),
            Pesos = sum(Venta_hoy),
            Return_Units = sum(Dev_uni_proxima),
            Return_Pesos = sum(Dev_proxima),
            Net = sum(Demanda_uni_equil)) %>%
  mutate(Net_Pesos = Pesos - Return_Pesos,
         Return_Rate = Return_Units / (Units+Return_Units)) %>%
  arrange(desc(Units)) %>%
  inner_join(town, by = "Agencia_ID")

# Histograma demonstrando a demanda nas agências em unidades por semana
# Resposta: Maioria da demanda nas agências é abaixo de 200k de unidades por semana
# Resposta: MAis de 40 agências demandam o mesmo número de unidades por semana
ggplot(agencias, aes(x=Units/7))+
  geom_histogram(fill="red", color="black", alpha=0.5, binwidth=10000)+
  scale_x_continuous(name="Unidade por semana", labels=function(x)paste(x/1000, "k"))+
  scale_y_continuous(name="Depósitos")

# Qual a taxa de retorno de produtos das unidades em relação ao número de unidade de produtos demandadas
# Será avaliado apenas nos 100 depósitos com mais unidades
# O tamanho do quadrado indica a quantidade de unidades de produtos
# A cor indica a taxa de retorno de unidades de produtos demandados
# Resposta: Não há correlação entre número de unidade e taxa de retorno de unidades
treemap(agencias[1:100, ], 
        index=c("Agencia_ID"), vSize="Units", vColor="Return_Rate", 
        palette=c("#FFFFFF","#FFFFFF","#FF0000"),
        type="value", title.legend="Taxa de retorno (%)", 
        title="Top 100 depósitos")

# Coletar os 30 e 100 depósitos com mais itens demandados
top30agencias <- agencias$Agencia_ID[1:30]
top100agencias <- agencias$Agencia_ID[1:100]

# Qual o número de unidades de produtos demandados por cada depósito em cada dia da semana
# Organizar o df agrupando por depósito e por dia da semana
# Manter o mesmo racional de colunas somadas
agencias.history <- train %>%
  group_by(Agencia_ID, Semana) %>%
  summarise(Units = sum(Venta_uni_hoy),
            Pesos = sum(Venta_hoy),
            Return_Units = sum(Dev_uni_proxima),
            Return_Pesos = sum(Dev_proxima),
            Net = sum(Demanda_uni_equil)) %>%
  mutate(Net_Pesos = Pesos - Return_Pesos,
         Avg_Pesos = Pesos / Units,
         Return_Rate = Return_Units / (Units+Return_Units)) %>%
  arrange(Agencia_ID, Semana) %>%
  inner_join(town, by="Agencia_ID")

# Quantas unidades cada depósito é demandado em cada dia da semana
# A cor será baseada na taxa de unidades devolvidas
# Em geom_bar o stat identity permanece o valor como está no df (é possível fazer soma)
# Facet wrap cria quadrantes para o número de agências
ggplot(agencias.history %>% filter(Agencia_ID %in% top30agencias))+
  geom_bar(aes(x=Semana, y=Units, fill=Return_Rate), stat="identity", color="black")+
  facet_wrap(~Agencia_ID)+
  scale_x_continuous((name="Dia da semana"))+
  scale_y_continuous(name="Unidades", labels=function(x)paste(x/1000, "k"))+
  scale_fill_gradient(name="Taxa de\nretorno %", low="white", high="red")+
  ggtitle("Top 30 depósitos")+
  theme_bw()

# Qual o número de unidades demandada por estado em cada dia da semana
# Organizar df por estado e semana
# Manter o mesmo racional de colunas somadas
states <- agencias.history %>%
  group_by(State, Semana) %>%
  summarise(Units = sum(Units),
            Pesos = sum(Pesos),
            Return_Units = sum(Return_Units),
            Return_Pesos = sum(Return_Pesos),
            Net = sum(Net)) %>%
  mutate(Avg_Pesos = Pesos / Units,
         Return_Rate = Return_Units / (Units+Return_Units)) %>%
  arrange(desc(Units))

# Quantas unidades cada estado é demandado em cada dia da semana
# A cor será baseada na taxa de unidades devolvidas
ggplot(states)+
  geom_bar(aes(x=Semana, y=Units, fill=Return_Rate), stat="identity", color="black")+
  scale_x_continuous(name="Dia da semana")+
  scale_y_continuous(name="Unidades", labels=function(x)paste(x/1e6, "m"))+
  scale_fill_gradient(name="Taxa de\nretorno %", low="white", high="red")+
  facet_wrap(~State)+
  ggtitle("Estados")+
  theme_bw()

# Qual canal de venda teve mais demanda de unidades
# Organizar df agrupando com canal e por semana
canals <- train %>%
  group_by(Canal_ID, Semana) %>%
  summarise(Units = sum(Venta_uni_hoy),
            Pesos = sum(Venta_hoy),
            Return_Units = sum(Dev_uni_proxima),
            Return_Pesos = sum(Dev_proxima),
            Net = sum(Demanda_uni_equil)) %>%
  mutate(Net_Pesos = Pesos - Return_Pesos,
         Avg_Pesos = Pesos / Units,
         Return_Rate = Return_Units / (Units+Return_Units)) %>%
  arrange(desc(Units))

# Qual o canal teve mais demanda de unidades
# Resposta: Canal 1 é o mais importante
treemap(canals, index=c("Canal_ID"), vSize="Units", type="index", 
        title="Top canais")

# Qual a demanda de unidades por dia de semana em cada canal
# Resposta: O canal 1 tem maior unidade demandada
ggplot(canals)+
  geom_bar(aes(x=Semana, y=Units, fill=Return_Rate), stat="identity", color="black")+
  scale_x_continuous(name="Dia da semana")+
  scale_y_continuous(name="Unidades", labels=function(x)paste(x/1e6, "m"))+
  scale_fill_gradient(name="Taxa de\nretorno %", low="white", high="red")+
  facet_wrap(~Canal_ID, scale="free")+
  theme_bw()

# Quantos canais distintos cada depósito utiliza para realizar as demandas?
# Organizar df agrupando por depósito e avaliando número distinto de canais
agencias.canals <- train %>%
  group_by(Agencia_ID) %>%
  summarise(n_canals = n_distinct(Canal_ID))

# Resposta: Mais de 350 lojas utilizam um único canal para as demandas
ggplot(agencias.canals)+
  geom_histogram(aes(x=n_canals), fill="red", color="black", alpha=0.5, binwidth=0.5)+
  scale_x_continuous(name="Número de canais", breaks=1:5)+
  scale_y_continuous(name="Número de depósitos")+
  theme(axis.text.x=element_text(hjust=1))+
  theme_bw()

# Quais rotas entregam mais unidades
# Organizar df agrupando por rota e determianndo quantidade de depositos distintas
# Quantidade de cliente distinto
# E número de unidades vendidas e a demanda da próxima semana
routes <- train %>% group_by(Ruta_SAK) %>%
  summarise(n_Agencias = n_distinct(Agencia_ID),
            n_Clients = n_distinct(Cliente_ID),
            Units=sum(Venta_uni_hoy),
            Return_Units = sum(Dev_uni_proxima)) %>%
  mutate(Return_Rate = Return_Units / (Units+Return_Units)) %>%
  arrange(desc(Units))

# Resposta: Maioria das rotas entregam < 100 k unidades por semana
ggplot(routes, aes(x=Units/7))+
  geom_histogram(fill="red", color="black", alpha=0.5, binwidth=5000)+
  scale_x_continuous(name="Unidades / Semana", labels=function(x)paste(x/1000, "k"))+
  scale_y_continuous(name="Rotas")+
  theme_bw()

# Qual o cliente com maior demanda?
# Organizar df agrupando por cliente e soma de unidade e valor da demanda
# Inner com os nomes dos clientes baseado id do cliente
sales <- train %>%
  group_by(Cliente_ID) %>%
  summarise(Units = sum(Venta_uni_hoy),
            Pesos = sum(Venta_hoy),
            Return_Units = sum(Dev_uni_proxima),
            Return_Pesos = sum(Dev_proxima),
            Net = sum(Demanda_uni_equil)) %>%
  mutate(Return_Rate = Return_Units / (Units+Return_Units),
         Avg_Pesos = Pesos / Units) %>%
  mutate(Net_Pesos = Pesos - Return_Pesos) %>%
  inner_join(client, by="Cliente_ID") %>%
  arrange(desc(Pesos))

# Gráfico com o cliente com maior demanda em unidade e menor taxa de retorno
# Limite dos 100 maiores clientes baseado no valor
# Resposta: Puebla remission é o cliente com maior demanda em unidades e considerável taxa de retorno
treemap(sales[1:100, ], 
        index=c("NombreCliente"), vSize="Units", vColor="Return_Rate", 
        palette=c("#FFFFFF","#FFFFFF","#FF0000"),
        type="value", title.legend="Taxa de retorno %", title="Top 100 clientes")

# Com quantos depósitos distintos cada cliente trabalha?
# Organizar df agrupando por cliente e por depósitos distintos
agencias.by.client <- train %>%
  group_by(Cliente_ID) %>%
  summarise(n_agencias = n_distinct(Agencia_ID)) %>%
  inner_join(client, by="Cliente_ID")

# Plotar tabela com quantos depósitos os clientes trabalham
# Respostar: 844113 clientes trabalham com um único depósito
table(agencias.by.client$n_agencias)

# Quem são os clientes que trabalham com os depósitos de números 5, 9 e 62 ?
# Resposta: desayunos, puebla remission e comercializadora la puerta del sol
# O maior cliente utiliza mais depósitos (faz sentido)
agencias.by.client %>% filter(n_agencias %in% c(5, 9, 62))

# Com quantos canais distintos cada cliente interage
clients.canals <- train %>%
  group_by(Cliente_ID) %>%
  summarise(n_canals = n_distinct(Canal_ID))

# Plotar tabela com quantos canais os clientes interagem
# Resposta: 874022 clientes interagem apenas com 1 canal
table(clients.canals$n_canals)

# Quantos canais são utilizados para interagir com clientes
clients.agencies.canals <- train %>%
  group_by(Cliente_ID, Agencia_ID) %>%
  summarise(n_canals = n_distinct(Canal_ID))

# Plotar tabela com quantos canais os depósitos utilizam para interagir com os clientes
# Resposta: 922108 lojas e clientes interagem pelo mesmo canal
table(clients.agencies.canals$n_canals)

# Quantas rotas diferentes os clientes recebem os pedidos?
# Organizar df agrupando pelo id dos clientes e as rotas distintas das entregas
clients.routes <- train %>%
  group_by(Cliente_ID) %>%
  summarise(n_routes = n_distinct(Ruta_SAK))

# Plotar gráfico quantas rotas tem maior número de clientes
# Resposta: Maioria dos clientes tem < 10 rotas de entregas
ggplot(clients.routes)+
  geom_histogram(aes(x=n_routes), fill="red", color="black", alpha=0.5, binwidth=1)+
  scale_x_continuous(name="Número de rotas")+
  scale_y_continuous(name="Número de clientes", labels=function(x)paste(x/1000, "k"))+
  theme_bw()

# Quais os produtos com maior demanda
# Organizar df agrupando pelo produto e soma de unidade, valor e retorno
# Inner com o nome do produto baseado no id
products <- train %>% group_by(Producto_ID) %>%
  summarise(Units = sum(Venta_uni_hoy),
            Pesos = sum(Venta_hoy),
            Return_Units = sum(Dev_uni_proxima),
            Return_Pesos = sum(Dev_proxima),
            Net = sum(Demanda_uni_equil)) %>%
  mutate(Avg_Pesos = Pesos / Units,
         Return_Rate = Return_Units / (Units+Return_Units)) %>%
  filter(!is.nan(Avg_Pesos)) %>%
  inner_join(product, by="Producto_ID") %>%
  arrange(desc(Units))

# Transformar o nome dos produtos em categórica
products$NombreProducto <- factor(as.character(products$NombreProducto), levels=products$NombreProducto)

# Plotar gráfico com o produto de maior demanda
# A cor demosntra a taxa de retorno dos produtos
# Resposta: Nito 1p 62g e o item de maior demanda e bolsa mini rock tem alta taxa de devolução
treemap(products[1:100, ], 
        index=c("NombreProducto"), vSize="Units", vColor="Return_Rate", 
        palette=c("#FFFFFF","#FFFFFF","#FF0000"),
        type="value", title.legend="Taxa de retorno %", title="Top 100 produtos")

# Qual o valor médio da maioria dos produtos?
# Resposta: Maioria dos item tem valor médio proximo de 10
ggplot(products, aes(x=Avg_Pesos))+
  geom_histogram(aes(y=..density..), fill="gray", color="black", alpha=0.5)+
  geom_density(fill="red", alpha=0.5)+
  scale_x_continuous(name="Média do preço dos produtos", lim=c(0, 50))+
  scale_y_continuous(name="Densidade", labels=percent)+
  theme_bw()

# Selecionar os top 100 produtos
top100products <- products$Producto_ID[1:100]

# Qual a quantidade de produtos cada depósito trabalha
# Organizar df
agencias.products <- train %>% group_by(Agencia_ID) %>%
  summarise(n_products = n_distinct(Producto_ID))

# Resposta: A maioria dos depositos trabalham com entre 100 e 200 produtos
ggplot(agencias.products)+
  geom_histogram(aes(x=n_products), fill="red", color="black", alpha=0.5, binwidth=10)+
  scale_x_continuous(name="Número de produtos")+
  scale_y_continuous(name="Número de depósitos")+
  theme_bw()

# Quantos produtos são vendidos por quais canais
# Organizar df por id de produto e canais distintos
canals.products <- train %>% group_by(Producto_ID) %>%
  summarise(n_canals = n_distinct(Canal_ID))

# Resposta: Maioria dos produtos são vendidos por ao menos 2 canais distintos
ggplot(canals.products)+
  geom_histogram(aes(x=n_canals), fill="red", color="black", alpha=0.5, binwidth=1)+
  scale_x_continuous(name="Número de canais", breaks=1:10, lim=c(1, 10))+
  scale_y_continuous(name="Número de produtos")+
  theme_bw()

# Quantas rotas distintas são utilizadas nas entregas dos produtos?
# Organizar df agrupando por id de produtos e rotas distintas
routes.products <- train %>% group_by(Producto_ID) %>%
  summarise(n_routes = n_distinct(Ruta_SAK))

# Resposta: Maioria dos produtos são entregues com < 100 rotas distintas
ggplot(routes.products)+
  geom_histogram(aes(x=n_routes), fill="red", color="black", alpha=0.5, binwidth=10)+
  scale_x_continuous(name="Número de rotas")+
  scale_y_continuous(name="Número de produtos")+
  theme_bw()

# Quantos produtos utilizam muitas rotas para a entrega
# Organizar df agrupando por rotas e produtos distintos
routes.products <- train %>% group_by(Ruta_SAK) %>%
  summarise(n_products = n_distinct(Producto_ID))

# Resposta: Poucos produtos (< 50) utilizam muitas rotas para ser entregues
ggplot(routes.products)+
  geom_histogram(aes(x=n_products), fill="red", color="black", alpha=0.5, binwidth=10)+
  scale_x_continuous(name="Número de produtos")+
  scale_y_continuous(name="Número de rotas")+
  theme_bw()

# Quantos produtos cada cliente demanda?
# Organizar df agrupando por cliente e produtos distintos
# Inner do nome dos clientes baseado no id do cliente
products.by.client <- train %>%
  group_by(Cliente_ID) %>%
  summarise(n_products = n_distinct(Producto_ID)) %>%
  inner_join(client, by="Cliente_ID")

# Resposta Maioria dos clientes (aproximadamente 50k) demandam poucos produtos distintos (<50)
ggplot(products.by.client)+
  geom_histogram(aes(x=n_products), fill="red", color="black", alpha=0.5, binwidth=2)+
  scale_x_continuous(name="Número de produtos por clientes", lim=c(0, 150))+
  scale_y_continuous(name="Número de clientes", labels=function(x)paste(x/1000, "k"))+
  theme_bw()

################### XGBOOST ########################

# Selecionar dataset para treino e teste
train <- fread('D/DataSience/CientistaDeDados/BigDataAnalytics_R_Azure/ProjetoscomFeedback/Projeto02/train.csv',
            select = c("Semana",'Cliente_ID', 'Producto_ID', 'Agencia_ID', 'Ruta_SAK', 'Demanda_uni_equil'))
test <- fread('D/DataSience/CientistaDeDados/BigDataAnalytics_R_Azure/ProjetoscomFeedback/Projeto02/test.csv',
           select = c("Semana",'id','Cliente_ID', 'Producto_ID', 'Agencia_ID', 'Ruta_SAK'))

# Selecionar a partir da semana 4 para ter dados da semana anterior das semanas
# 5, 6, 7 e 8
# E utilizar a semana 9 como teste do treino
train <- train[Semana>4,]

# Verificar a proporção de dados por semana no dataset
prop.table(table(train$Semana))
prop.table(table(test$Semana))

# Atribuir todo o dataset com id igual a 0 para identificar df de treino
train$id <- 0

# Colocar o nome da coluna alvo com target e eliminar a coluna com nome antigo
train$target <- train$Demanda_uni_equil
train$Demanda_uni_equil <- NULL

# Coluna tst com valor igual 0 - distinguir os dados de treino e teste
train$tst <- 0

# Coluna target igual 0 no df teste 
test$target <- 0

# Coluna tst com valor igual 1 - distinguir os dados de treino e teste
test$tst <- 1

# Unir os datasets de treino e teste
data <- rbind(train, test)
rm(test)  
rm(train)

# Criar feature de uma semana anterior com os valores da variável target
# Primeiro criar um df com a semana + 1 parar criar o valor target na semana seguinte
data1 <- data[,.(Semana=Semana+1,Cliente_ID,Producto_ID,target)]
# Merge dos dados (neste caso média do target) com a semana posterior
# Agrupado com semana, cliente e produto
data <- merge(data,
           data1[Semana>6,.(targetl1=mean(target)), 
                      by=.(Semana,Cliente_ID,Producto_ID)],
           all.x=T, 
           by=c("Semana","Cliente_ID","Producto_ID"))

# Deletar o data1
rm(data1)

# Selecionar a partir da semana 4 que quando temos a média do target da semana anterior
data <- data[Semana>5,]

# Criar features com frequencias para algumas variáveis
# Número de de linhas por cada grupo, nteste caso por agencia e semana
nAgencia_ID <- data[,.(nAgencia_ID=.N),by=.(Agencia_ID,Semana)]
# Média de linhas por cada grupo, neste caso por agencia
nAgencia_ID <- nAgencia_ID[,.(nAgencia_ID=mean(nAgencia_ID,na.rm=T)),by=Agencia_ID]
data <- merge(data,nAgencia_ID,by='Agencia_ID',all.x=T)
rm(nAgencia_ID)

# Número de de linhas por cada grupo, nteste caso por rota e semana
nRuta_SAK <- data[,.(nRuta_SAK=.N),by=.(Ruta_SAK,Semana)]
# Média de linhas por cada grupo, neste caso por rota
nRuta_SAK <- nRuta_SAK[,.(nRuta_SAK=mean(nRuta_SAK,na.rm=T)),by=Ruta_SAK]
data <- merge(data,nRuta_SAK,by='Ruta_SAK',all.x=T)
rm(nRuta_SAK)

# Número de de linhas por cada grupo, nteste caso por cliente e semana
nCliente_ID <- data[,.(nCliente_ID=.N),by=.(Cliente_ID,Semana)]
# Média de linhas por cada grupo, neste caso por cliente
nCliente_ID <- nCliente_ID[,.(nCliente_ID=mean(nCliente_ID,na.rm=T)),by=Cliente_ID]
data <- merge(data,nCliente_ID,by='Cliente_ID',all.x=T)
rm(nCliente_ID)

# Número de de linhas por cada grupo, nteste caso por produto e semana
nProducto_ID <- data[,.(nProducto_ID=.N),by=.(Producto_ID,Semana)]
# Média de linhas por cada grupo, neste caso por produto
nProducto_ID <- nProducto_ID[,.(nProducto_ID=mean(nProducto_ID,na.rm=T)),by=Producto_ID]
data <- merge(data,nProducto_ID,by='Producto_ID',all.x=T)
rm(nProducto_ID)

# Separar os df em treino e teste com toda a engenharia de atributos
data_train=data[tst==0,]
data_test=data[tst==1,]
rm(data)

# Preparando os dados de treino e de teste dentro do data_train
# Será treinado nas semanas 7 e 8 e avaliado na semana 9
data_train_test <- data_train[Semana == 9]
data_train <- data_train[Semana<9,]

# Coletar os nomes das features menos id, target e tst que não devem ser avaliadas
features=names(data_train)[!(names(data_train) %in% c('id',"target",'tst'))] 

# Pegar índice para amostragem do data_train
# 100000 indices selecionados aleatoriamente
wltst <- sample(nrow(data_train),100000)  

# Transformar na matrix do xgb
dval <- xgb.DMatrix(data=data.matrix(data_train[wltst,features,with=FALSE]),
                  label=data.matrix(data_train[wltst,target]),missing=NA)
watchlist <- list(dval=dval)

# Treinar o modelo de regressão linear
clf <- xgb.train(params=list(  objective="reg:linear", 
                               booster = "gbtree",
                               eta=0.1, 
                               max_depth=10, 
                               subsample=0.85,
                               colsample_bytree=0.7) ,
                 data = xgb.DMatrix(data=data.matrix(data_train[-wltst,features,with=FALSE]),
                                    label=data.matrix(data_train[-wltst,target]),missing=NA), 
                 nrounds = 75, 
                 verbose = 1,
                 print_every_n=5,
                 early_stopping_rounds    = 10,
                 watchlist           = watchlist,
                 maximize            = FALSE,
                 eval_metric='rmse'
)

# Visualizando os detalhes do modelo
summary(clf)

# Fazendo previsões com a semana 9 e analisando o resultado
# Importante pois na semana 9 temos os valores reais
previsoes <- predict(clf, xgb.DMatrix(data.matrix(data_train_test[,features,with=FALSE]), missing=NA), type = "response")

# Df com os valores reais e as previsões do modelo
df_previsoes <- data.frame('previsao' = previsoes, 'real' = data_train_test$target)

# Estatística
postResample(df_previsoes$previsao, df_previsoes$real)
# RMSE mostra qnts unidades de erros temos no modelo ( no caso de unidades de produtos)
# R squared mostra o quanto a regressão explica a variável dependente a partir das variáveis independentes
# MAE demonstra a diferença absoluta de erros do modelo 

# Adicionando uma coluna com os resíduos do modelo
df_previsoes$residual <- df_previsoes$real - df_previsoes$previsao

# Selecionando parte das previsões para plotar o gráfico de residuos
prev <- df_previsoes %>% 
  sample_frac(0.01)

# Plotar gráfico dos resíduos
plot(prev$residual, ylab = "residuals", main = "extreme gradient boosting")

# Plotar gráfico dos valores reais vs resíduos
plot(prev$real,
     prev$previsao,
     xlab = "actual",
     ylab = "predicted",
     main = "extreme gradient boosting")
abline(lm(prev$previsao ~ prev$rea))

# Visualizar os variáveis independentes mais importantes do modelo
importance_matrix <- xgb.importance(model = clf)
xgb.plot.importance(importance_matrix, xlab = "Feature Importance")

# TEntativa de analisar o dataset da semana 10 e 11 que não temos o valor real
# Vida real (dados de teste)
# Selecionando o dataset da semana 10
data_test1 <- data_test[Semana==10,]

# Predição da Semana 10 (dados de teste)
pred <- predict(clf, xgb.DMatrix(data.matrix(data_test1[,features,with=FALSE]),missing=NA))

# Dataframe com o id da semana 10 e as previsẽs da semana 10
res <- exp(round(pred,5))-1
results <- data.frame(id=data_test1$id,Demanda_uni_equil=res)

# Preparando o dataset de teste da semana 11, agrupando por cliente e produto e inserindo o valor do target da semana anterior
data_test_lag1 <- data_test1[,.(Cliente_ID,Producto_ID)]

# Inserindo o valor da predição no dataset
data_test_lag1$targetl1 <- res

# Agrupando por cliente e produto
data_test_lag1 <- data_test_lag1[,.(targetl1=mean(targetl1)), by=.(Cliente_ID,Producto_ID)]

# Selecionando dataset de treino da Semana 11
data_test2 <- data_test[Semana==11,]

# Eliminando a coluna targetl1
data_test2[,targetl1:=NULL]

# Agrupando o dataset da semana 11 com os dados da predição
data_test2 <- merge(data_test2,data_test_lag1,all.x=T,by=c('Cliente_ID','Producto_ID'))

# Predição da semana 11
pred <- predict(clf, xgb.DMatrix(data.matrix(data_test2[,features,with=FALSE]),missing=NA))

# Dataframe com o id da semana 11 e as previsẽs da semana 11
res <- exp(round(pred,5))-1
res.df <- data.frame(id=data_test2$id,Demanda_uni_equil=res)

# Uninando os dataframes com o id das semana 10 e 11 mais as previsẽs dass semanas 10 e 11
# Aqui poderia enviar para o kaggle
results <- rbind(results, res.df)
