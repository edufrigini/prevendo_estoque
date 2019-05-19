# DSA - DATA SCIENCE ACADEMY 
# FORMACAO CIENTISTA DE DADOS
# LIGUAGEM R COM AZURE MACHINE LEARNING
#
# PROJETO 2, Prevendo Demanda de Estoque com Base em Vendas 
# ALUNO: EDUARDO FRIGINI DE JESUS 
# 
# Goal: Maximize sales and minimize returns of bakery goods

# Data fields
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
# Demanda_uni_equil — Adjusted Demand (integer) (This is the target you will predict)

setwd("C:/FCD/BigDataRAzure/Projeto2")
getwd()

install.packages("gmodels") 
install.packages("psych")
install.packages("rmarkdown")
library(rmarkdown)


# carregando as bibliotecas, se nao estiver instalada, instalar install.packages("nome do pacote")
library(data.table) # para usar a fread
library("gmodels") # para usar o CrossTable
library(psych) # para usar o pairs.panels
library(lattice) # graficos de correlacao
require(ggplot2)
library(randomForest)


library(DMwR)
library(dplyr)
library(tidyr)
library("ROCR")
library(caret)
library(lattice)
library(corrplot)
library(corrgram)

##  Carregando os dados na memoria
# Usando o arquivo train.csv para treinar o modelo para producao
dados_originais <- fread("train.csv", sep = ",", header = TRUE, stringsAsFactors = TRUE)
dados  <- dados_originais[sample(1:nrow(dados_originais), 1000, replace = F)]

head(dados)
str(dados)
View(dados)

## Tratando os dados
## Convertendo as variáveis para o tipo fator (categórica)
to.factors <- function(df, variables){
  for (variable in variables){
    df[[variable]] <- as.factor(df[[variable]])
  }
  return(df)
}
# Variáveis do tipo fator
# nao converti as outras pq eram mts categorias e a floresta randomica nao processa mts categorias
colunas_F <- c("Semana", "Canal_ID")
dados <- to.factors(df = dados, variables = colunas_F)

# corrigir caso hajam dados NA
dados <- na.omit(dados)
str(dados)
head(dados)



## Analise exploratoria dos dados
summary(dados$Demanda_uni_equil)
# Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
# 0.000   2.000   3.000   7.018   6.000 230.000 

mean(dados$Demanda_uni_equil) # 7.018
median(dados$Demanda_uni_equil) # 3
quantile(dados$Demanda_uni_equil) # 0%  25%  50%  75% 100% 
                                  # 0    2    3    6  230 
quantile(dados$Demanda_uni_equil, probs = c(0.01, 0.95)) # 1% = 0.00 e 95% = 23.05
quantile(dados$Demanda_uni_equil, seq(from = 0, to = 1, by = 0.10))
IQR(dados$Demanda_uni_equil) # diferenca entre Q3 e Q1 = 4
range(dados$Demanda_uni_equil) # 230
diff(range(dados$Demanda_uni_equil))
sd(dados$Demanda_uni_equil) # 13.74
var(dados$Demanda_uni_equil) # 188.9666


# A variavel alvo esta com muitos outlyers, dai vou restringir ao valores menores que 0.95 quartil
quartil_90 <- quantile(dados$Demanda_uni_equil, probs = 0.90)
class(quartil_90)
quartil_90[[1]]
dados_sem_ouliers <- dados[Demanda_uni_equil<=quartil_90[[1]],]
plot(dados_sem_ouliers$Demanda_uni_equil)

sd(dados_sem_ouliers$Demanda_uni_equil) #  4.35
var(dados_sem_ouliers$Demanda_uni_equil) # 19

## OS DADOS DO TARGET NAO SEGUEM UMA DISTRIBUICAO NORMAL


## Explorando os dados graficamente
plot(dados_sem_ouliers$Semana)
hist(dados_sem_ouliers$Ruta_SAK)

plot(dados_sem_ouliers$Demanda_uni_equil) # target e esta com outliers
hist(dados_sem_ouliers$Demanda_uni_equil) # dados concentrados no zero
boxplot(dados_sem_ouliers$Demanda_uni_equil)

plot(dados_sem_ouliers$Agencia_ID)
plot(dados_sem_ouliers$Canal_ID) # Predominancia do canal 1

# Explorando os dados
# variaveis numericas
cols <- c("Venta_uni_hoy", "Venta_hoy", "Dev_uni_proxima", "Dev_proxima", "Demanda_uni_equil") 
cor(dados[, cols, with=FALSE])
pairs.panels(dados[, cols, with=FALSE])

# Correlacao forte com Venta_Roy e Venta_Uni_Roy
#### OBSERVACOES
#### Venta_uni_hoy e Venta_hoy sao colineares
#### Dev_uni_proxima e Dev Proxima sao colineares



######################################################################
## Apenas para confirmar a correlacao com Venta_hoy e Venta_uni_hoy ##
######################################################################

# Vetor com os métodos de correlação
metodos <- c("pearson", "spearman")

# Aplicando os métodos de correlação com a função cor()
cors <- lapply(metodos, function(method) 
  (cor(dados[, cols, with=FALSE], method = method)))

head(cors)

# Preparando o plot
plot.cors <- function(x, labs){
  diag(x) <- 0.0 
  plot( levelplot(x, 
                  main = paste("Plot de Correlação usando Método", labs),
                  scales = list(x = list(rot = 90), cex = 1.0)) )
}

# Mapa de Correlação
Map(plot.cors, cors, metodos)


#########################################################
## Sem necessidade desse grafico, apenas para confirmar
#########################################################


str(dados_sem_ouliers)
# Demanda x potenciais variáveis preditoras
labels <- list("Boxplots - Demanda por Semana",
               "Boxplots - Demanda por Agencia",
               "Boxplots - Demanda por Canal",
               "Boxplots - Demanda por Rota",
               "Boxplots - Demanda Cliente")

xAxis <- list("Semana", "Agencia_ID", "Canal_ID", "Ruta_SAK", "Cliente_ID")

# Função para criar os boxplots
plot.boxes  <- function(X, label){ 
  ggplot(dados, aes_string(x = X, y = "Demanda_uni_equil", group = X)) + 
    geom_boxplot( ) + 
    ggtitle(label) +
    theme(text = element_text(size = 18)) 
}

Map(plot.boxes, xAxis, labels)

str(dados_sem_ouliers)

# Avalidando a importância de todas as variaveis
modelo <- randomForest(Demanda_uni_equil ~ . , 
                       data = dados_sem_ouliers, 
                       ntree = 100, 
                       nodesize = 10,
                       importance = TRUE)
# Plotando as variáveis por grau de importância
varImpPlot(modelo)


# Correlacao forte com Venta_Roy e Venta_Uni_Roy
#### OBSERVACOES
#### Venta_uni_hoy e Venta_hoy sao colineares
#### Dev_uni_proxima e Dev_Proxima sao colineares

# Removendo variáveis colineares
modelo <- randomForest(Demanda_uni_equil ~ . - Venta_hoy
                       - Dev_proxima, 
                       data = dados_sem_ouliers, 
                       ntree = 100, 
                       nodesize = 10,
                       importance = TRUE)
varImpPlot(modelo)

modelo$importance

# Gravando o resultado
df_saida <- dados_sem_ouliers[, c("Demanda_uni_equil", (modelo$importance))]
df_saida

# Removendo as variaveis menos importantes ou colineares
dados_ok <- dados_sem_ouliers
dados_ok$Dev_proxima <- NULL
dados_ok$Venta_hoy <- NULL
dados_ok$Cliente_ID <- NULL
dados_ok$Agencia_ID <- NULL
str(dados_ok)

# Gerando dados de treino e de teste
sample <- sample.int(n = nrow(dados_ok), size = floor(.7*nrow(dados_sem_ouliers)), replace = F)
treino <- dados_ok[sample, ]
teste  <- dados_ok[-sample, ]

# Verificando o numero de linhas
nrow(treino)
nrow(teste)

# Treinando o modelo linear (usando os dados de treino)
modelo_lm <- lm(Demanda_uni_equil ~ ., data = treino)
modelo_lm

# Prevendo demanda de produtos
previsao1 <- predict(modelo_lm)
View(previsao1)
plot(treino$Demanda_uni_equil, previsao1)

previsao2 <- predict(modelo_lm, teste)
View(previsao2)
plot(teste$Demanda_uni_equil, previsao2)


# Avaliando a Performance do Modelo
summary(modelo_lm)

# Adjusted R-squared:  0.9923


