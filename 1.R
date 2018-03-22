library('GGally')       # графики совместного разброса переменных
library('lmtest')       # тесты остатков регрессионных моделей
library('FNN')          # алгоритм kNN
library('ISLR')

data("Auto")
?Auto

head(Auto)
str(Auto)

Auto$cylinders <- as.factor(Auto$cylinders)

# константы
my.seed <- 100
train.percent <- 0.85

# обучающая выборка и тестовая выборка
set.seed(my.seed)
inTrain <- sample(seq_along(Auto$mpg), 
                  nrow(Auto) * train.percent)
df.train <- Auto[inTrain, c(1:5)]
df.test <- Auto[-inTrain, c(1:5)]

# описательные статистики по переменным
summary(df.train)

# совместный график разброса переменных
ggp <- ggpairs(df.train)
print(ggp, progress = F)

# цвета по фактору cylinders
ggp <- ggpairs(df.train, 
               mapping = ggplot2::aes(color = cylinders))
print(ggp, progress = F)

# Модели
model.1 <- lm(mpg ~ weight + displacement + horsepower + cylinders,
              data = df.train)
summary(model.1)

# исключаем displacement
model.2 <- lm(mpg ~ weight + horsepower + cylinders,
              data = df.train)
summary(model.2)

# Исключаем cylinders
model.3 <- lm(mpg ~ weight + horsepower,
              data = df.train)
summary(model.3)

#Проверка остатков

# тест Бройша-Пагана
bptest(model.3)

# статистика Дарбина-Уотсона
dwtest(model.3)

# графики остатков
par(mar = c(4.5, 4.5, 2, 1))
par(mfrow = c(1, 3))
plot(model.3, 1)
plot(model.3, 4)
plot(model.3, 5)
par(mfrow = c(1, 1))

# Сравнение с kNN
# фактические значения y на тестовой выборке
y.fact <- Auto[-inTrain, 1]
y.model.lm <- predict(model.3, df.test)
MSE.lm <- sum((y.model.lm - y.fact)^2) / length(y.model.lm)

# kNN требует на вход только числовые переменные
df.train.num <- as.data.frame(apply(df.train, 2, as.numeric))
df.test.num <- as.data.frame(apply(df.test, 2, as.numeric))

for (i in 2:50){
  model.knn <- knn.reg(train = df.train.num[, !(colnames(df.train.num) %in% 'mpg')], 
                       y = df.train.num[, 'mpg'], 
                       test = df.test.num[, -1], k = i)
  y.model.knn <- model.knn$pred
  if (i == 2){
    MSE.knn <- sum((y.model.knn - y.fact)^2) / length(y.model.knn)
  } else {
    MSE.knn <- c(MSE.knn, 
                 sum((y.model.knn - y.fact)^2) / length(y.model.knn))
  }
}

# график
par(mar = c(4.5, 4.5, 1, 1))
plot(2:50, MSE.knn, type = 'b', col = 'darkgreen',
     xlab = 'значение k', ylab = 'MSE на тестовой выборке')
lines(2:50, rep(MSE.lm, 49), lwd = 2, col = grey(0.2), lty = 2)
legend('bottomright', lty = c(1, 2), pch = c(1, NA), 
       col = c('darkgreen', grey(0.2)), 
       legend = c('k ближайших соседа', 'регрессия (все факторы)'), 
       lwd = rep(2, 2))

