# BigData_pySPARK_AM
Проект от Яндекс. Мастерская


## Описание проекта
В распоряжении около 10 млн записей о поездках такси в Чикаго за 2022 и 2023 год. Необходимо построить ML модель на Spark предсказания количества заказов на следующий час для каждой общественной зоны Чикаго с применен PySpark на локальном кластере из Docker контейнеров
 
Исходники:

- Данные за 2022 год: https://data.cityofchicago.org/Transportation/Taxi-Trips-2022/npd7-ywjz
- Данные за 2023 год: https://data.cityofchicago.org/Transportation/Taxi-Trips-2023/e55j-2ewb


## Навыки и инструменты
- **pyspark**
   - import SparkConf, SparkContext
   - from sql ( SparkConf, SparkContext, functions, Window)
   - from ml (
     ml.feature import VectorAssembler, StandardScaler, \
     ml.evaluation import RegressionEvaluator, \
     ml.regression import LinearRegression, RandomForestRegressor, GBTRegressor \
  )
- **python**
- **pandas**
- **matplotlib**
- **numpy**
- **seaborn**
- **re** 
- **math**
- **os**
- **tqdm** 
- **folium**
- **scikit-learn**
    - from sklearn.model_selection import (
    **GridSearchCV**,
    **train_test_split**,
    **cross_val_score**
) 
    - from sklearn.preprocessing import **StandardScaler**
    - from sklearn.dummy import **DummyRegressor**
    - from sklearn.linear_model import **LinearRegression**
    - from sklearn.metrics import **mean_absolute_error**


## Вывод

 1. **Предобработка данных**


- столбцы trip_start_timestamp и trip_end_timestamp перевели в тип данных timestamp. Также изучили начало и конец временного диапазона представленного за 2022 и 2023 года.

Временной диапазон данных за 2022 год: \
- min(trip_start_timestamp) - 2022-01-01 00:00:00 
- max(trip_start_timestamp) - 2022-12-31 23:45:00 
- min(trip_end_timestamp) -  2022-01-01 00:00:00 
- max(trip_end_timestamp) - 2023-01-01 16:00:00 


Временной диапазон данных за 2023 год \
- min(trip_start_timestamp) - 2023-01-01 00:00:00 
- max(trip_start_timestamp) - 2023-08-01 00:00:00 
- min(trip_end_timestamp) - 2023-01-01 00:00:00 
- max(trip_end_timestamp) - 2023-08-01 13:15:00 

  
- удалили столбцы: pickup_centroid_latitude, pickup_centroid_latitude, pickup_centroid_longitude, pickup_centroid_location, dropoff_centroid_latitude, dropoff_centroid_longitude, dropoff_centroid_location, trip_id, pickup_census_tract, dropoff_census_tract, taxi_id
- пустые знаечния в pickup_community_area и dropoff_community_area заполнили нулями.
- удалили строки в столбцах которых присуствуеют пропуски: extras, trip_total, tips, tolls, fare, trip_seconds, trip_miles
- В датасет за 2022 переименовали компании:
  - Blue Ribbon Taxi Association и Blue Ribbon Taxi Association Inc.
  - Taxicab Insurance Agency Llc и Taxicab Insurance Agency, LLC
  - KOAM Taxi Association и Koam Taxi Association
- В датасет за 2023 переименовали компании:
  - Choice Taxi Association и Choice Taxi Association Inc
  - Blue Ribbon Taxi Association и Blue Ribbon Taxi Association Inc.
  - Taxicab Insurance Agency Llc и Taxicab Insurance Agency, LLC
  - Taxi Affiliation Services Llc - Yell и Taxi Affiliation Services
- В датасет за 2022 выполнили фильтрацию и оставить следующие значения:
  -  trip_seconds < 5000
  -  trip_miles < 200
  -  fare < 150
  -  tips < 20
  -  tolls < 100
  -  extras < 35
  -  trip_total < 150
- В датасет за 2023 выполнили фильтрацию и оставили следующие значения:
  -  trip_seconds < 6000
  -  trip_miles < 50
  -  fare < 150
  -  tips < 20
  -  tolls < 100
  -  extras < 50
  -  trip_total < 150
- после фильтрации удалили столбцы fare, tips, tolls, extras, так как trip_total будет нести в себе всю информацию об данных столбцах
- объединили taxi_2022_sdf и taxi_2023_sdf в один общий датасет


 2. **EDA (исследовательский анализ данных)**

Было выполнено исслдеование данных:

Топ-10 районов по заказам:
 - 1.Near North Side - (Central) 
 - 2.O'Hare - (Far North Side) 
 - 3.(The) Loop[11] - (Central) 
 - 4.Near West Side - (West Side) 
 - 5.Unknown - (Unknown) 
 - 6.Near South Side - (Central) 
 - 7.Lake View - (North side) 
 - 8.Garfield Ridge - (Southwest Side) 
 - 9.Lincoln Park - (North side) 
 - 10.Uptown - (Far North Side)
   
Районы располагаются побольшей части у аэропорта и центра (возле берега озера)

Топ-10 компаний по заказам:
 - Taxi Affiliation Services
 - Flash Cab	
 - Sun Taxi
 - City Service
 - Taxicab Insurance Agency, LLC	
 - Chicago Independents	
 - Medallion Leasin	
 - Globe Taxi	
 - 5 Star Taxi	
 - Blue Ribbon Taxi Association	
 - Star North Taxi Management Llc	
 - Choice Taxi Association	
 - Top Cab Affiliation	
 - U Taxicab	
 - 24 Seven Taxi

Компании работают во всех регионах

Топ-4 способа оплаты в такси:
 - Credit Card	
 - Cash	
 - Mobile	
 - Prcard	

Было выполнена частичная дополнительная предобработка данных, к примеру, избавились от строк, где ест ькомпании с общим за весь периодом кол-вом заказов меньше 100. А также избавились от строк, где регион поасадки был 0.

В даннном пункте были проанализированы предоставленные данные во времени.

- было замеченно, среднее значение спроса к услугам такси растет со временем.Ряд не станционарный.
- наблюдается сезонность дня.
- даты не влияют на спрос. влияет день недели/месяц/время/квартал. было замечено, что очень мальенький спрос каждый понедельник.
- к выходным срос спадает
- временные периода спроса ориентировочно: 12-19ч. Максимум приходится на 16-18ч.

 3. **Выделение признаков**

 признаки 'avg_trip_seconds', 'avg_trip_miles_avg', 'avg_trip_total', 
                 'hour_sin','hour_cos', 'day_of_week_sin', 'day_of_week_cos',
                 'month_sin', 'month_cos','quarter_sin', 'quarter_cos',
                 'rolling_mean_2', 
                 'lag_1h',
                 'lag_2h',
                 'lag_3h',
                 'lag_4h',
                 'lag_5h',
                 'lag_6h',
                 'lag_23h',
                 'lag_24h',
                 'lag_1w'

целевой признак =  'count_taxi_orders'


С помощью написанной функции make_features создаются признаки: в ней находятся среднии значения по стоимости поездки, расстоянию, времени поездки для каждого региона на каждый час. Также формируются столбцы с временной информация и затем переводятся с помощью тригонометрических формул из категоральных значений в числовые. Вычисляется скользящее среднее и формируются серия lag.

Затем с помощью taxi_data_split разбиваем датасет в соотношении 60/30/10 без перемешивания данных.

И потом данные с помощью функции taxi_scaled масштабируются.

 4. **Обучение моделей**
    
Для предсказаний были использованы три модели из билиотеки MLlibrary Spark: LinearRegression, GBTRegressor и RandomForestRegressor.

 5.  **Выбор лучшей модели**

Модель LinearRegression \
Среднее значение МАЕ по регионам на валидационной выборке:  2.402668764308234 \
Медианное значение МАЕ по регионам на валидационной выборке:  0.920516684209353 \

Модель RandomForestRegressor \
Среднее значение МАЕ по регионам на валидационной выборке:  2.227369849167553 \
Медианное значение МАЕ по регионам на валидационной выборке:  0.8399444770289052 \

Модель GBTRegressor \
Среднее значение МАЕ по регионам на валидационной выборке:  2.166565314679981 \
Медианное значение МАЕ по регионам на валидационной выборке:  0.8197810175121473 \

Лучше всех себя показал GBTRegressor

 6. **Тестирование лучшей модели**
    
Лучшая модель \
Среднее значение МАЕ по регионам на тестовой выборке:  2.4296289879506565 \
Медианное значение МАЕ по регионам на тестовой выборке:  0.7812887687072926 \

Погрешность в 2 заказа на каждый регион в час нас устраивает.

Улучшить качество предсказания моделей можно поработав с признакми: 
- расширить диапазон скользящей средней
- сделать больше сдвигов lag
- ввести перебор гиперпараметров




