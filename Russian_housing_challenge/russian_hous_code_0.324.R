library(data.table)
library(Matrix)
library(xgboost)
library(caret)
library(dplyr)

# Load CSV files

data.train <- read.csv(file.choose())
data.test <- read.csv(file.choose())
data.macro = read.csv(file.choose())

data.train<-as.data.table(data.train)

data.macro<-as.data.table(data.macro)

# Transform target variable, so that we can use RMSE in XGBoost
data.train[,price_doc:=log1p(as.integer(price_doc))]

data.train[is.na(data.train)] = -1
data.test[is.na(data.test)] = -1
# Combine data.tables
data <- rbind(data.train, data.test,fill=TRUE)

data<-as.data.table(data)

#clean full_sq and life_sq. sometime full_sq is smaller than life_sq
data[ ,life_sq:=ifelse(is.na(life_sq),full_sq,life_sq)]
data[ ,full_sq:=ifelse(life_sq>full_sq,life_sq,full_sq)];

#build_year
data[ ,build_year:=ifelse((build_year >1690 & build_year<2020),build_year,'NA')]
data[ ,build_year:=as.integer(build_year)]

#num_rooms
data[ ,num_room:=ifelse(num_room==0,'NA',num_room)]
data[ ,num_room:=as.integer(num_room)]

#state
data[ ,state := ifelse(state==33,3,state)]
# Convert characters to factors/numeric
cat("Feature engineering")
data[,":="(thermal_power_plant_raion=ifelse(thermal_power_plant_raion=="no",0,1)
           ,incineration_raion=ifelse(incineration_raion=="no",0,1)
           ,oil_chemistry_raion=ifelse(oil_chemistry_raion=="no",0,1)
           ,radiation_raion=ifelse(radiation_raion=="no",0,1)
           ,railroad_terminal_raion=ifelse(railroad_terminal_raion=="no",0,1)
           ,big_market_raion=ifelse(big_market_raion=="no",0,1)
           ,nuclear_reactor_raion=ifelse(nuclear_reactor_raion=="no",0,1)
           ,detention_facility_raion=ifelse(detention_facility_raion=="no",0,1)
           ,culture_objects_top_25=ifelse(culture_objects_top_25=="no",0,1)
           ,water_1line=ifelse(water_1line=="no",0,1)
           ,big_road1_1line=ifelse(big_road1_1line=="no",0,1)
           ,railroad_1line=ifelse(railroad_1line=="no",0,1)
)]

# Date features
data[,timestamp := as.Date(timestamp)]
data.macro[,timestamp := as.Date(timestamp)]
data[,":="(#date_yday=yday(timestamp)
   date_month=month(timestamp)
  ,date_year=year(timestamp)
  ,date_week=week(timestamp)
  ,date_mday=mday(timestamp)
  ,date_wday=wday(timestamp)
)]

# Count NA's
data[,na_count := rowSums(is.na(data))]

# Some relative features
data[,":="(rel_floor = floor/max_floor
           ,diff_floor = max_floor-floor
           ,rel_kitchen_sq = kitch_sq/full_sq
           ,rel_life_sq = life_sq/full_sq
           ,rel_kitchen_life = kitch_sq/life_sq
           ,rel_sq_per_floor = full_sq/floor
           ,diff_life_sq = full_sq-life_sq
           ,building_age = date_year - build_year
)]

#macro data
das<-as.data.frame(data.macro)
data.macro1 <- das[ , names(das) %in% c("timestamp" ,"balance_trade"
                                                            ,"balance_trade_growth"
                                                            ,"eurrub"
                                                            ,"average_provision_of_build_contract"
                                                            ,"micex_rgbi_tr"
                                                            ,"micex_cbi_tr"
                                                            ,"deposits_rate"
                                                            ,"mortgage_value"
                                                            ,"mortgage_rate"
                                                            ,"income_per_cap"
                                                            ,"rent_price_4.room_bus"
                                                            ,"museum_visitis_per_100_cap"
                                                            ,"apartment_build")]

names(data.macro1)
str(data.macro1)
#merge macro data
data1<-data
data<-merge(x = data,y = data.macro1,by = 'timestamp',all.x = TRUE)
str(data)

features = colnames(data)
for (f in features){
  if( (class(data[[f]]) == "character") || (class(data[[f]]) == "factor"))
  {
    levels = unique(data[[f]])
    data[[f]] = as.numeric(factor(data[[f]], level = levels))
  }
}

data[, material:=as.factor(material)]
data[, build_year := as.factor(build_year)]
data[, state:=as.factor(state)]
data[, product_type:=as.factor(product_type)]
data[, sub_area:=as.factor(sub_area)]
data[, ecology:=as.factor(ecology)]


# one-hot-encoding features
data = as.data.frame(data)
ohe_feats = c('material', 'build_year', 'state', 'product_type', 'sub_area', 'ecology')
dummies = dummyVars(~ material + build_year + state + product_type + sub_area + ecology , data = data)
df_all_ohe <- as.data.frame(predict(dummies, newdata = data))
df_all_combined <- cbind(data[,-c(which(colnames(data) %in% ohe_feats))],df_all_ohe)
data = as.data.table(df_all_combined)

data2<-data

data[,c("date_year", "timestamp", "young_male", "school_education_centers_top_20_raion", "0_17_female", "railroad_1line", "7_14_female", "0_17_all", "children_school",
        "16_29_male", "mosque_count_3000", "female_f", "church_count_1000", "railroad_terminal_raion",
        "mosque_count_5000", "big_road1_1line", "mosque_count_1000", "7_14_male", "0_6_female", "oil_chemistry_raion",
        "young_all", "0_17_male", "ID_bus_terminal", "university_top_20_raion", "mosque_count_500","ID_big_road1",
        "ID_railroad_terminal", "ID_railroad_station_walk", "ID_big_road2", "ID_metro", "ID_railroad_station_avto",
        "0_13_all", "mosque_count_2000", "work_male", "16_29_all", "young_female", "work_female", "0_13_female",
        "ekder_female", "7_14_all", "big_church_count_500",
        "leisure_count_500", "cafe_sum_1500_max_price_avg", "leisure_count_2000",
        "office_count_500", "male_f", "nuclear_reactor_raion", "0_6_male", "church_count_500", "build_count_before_1920",
        "thermal_power_plant_raion", "cafe_count_2000_na_price", "cafe_count_500_price_high",
        "market_count_2000", "museum_visitis_per_100_cap", "trc_count_500", "market_count_1000", "work_all", "additional_education_raion",
        "build_count_slag", "leisure_count_1000", "0_13_male", "office_raion",
        "raion_build_count_with_builddate_info", "market_count_3000", "ekder_all", "trc_count_1000", "build_count_1946-1970",
        "office_count_1500", "cafe_count_1500_na_price", "big_church_count_5000", "big_church_count_1000", "build_count_foam",
        "church_count_1500", "church_count_3000", "leisure_count_1500",
        "16_29_female", "build_count_after_1995", "cafe_avg_price_1500", "office_sqm_1000", "cafe_avg_price_5000", "cafe_avg_price_2000",
        "big_church_count_1500", "full_all", "cafe_sum_5000_min_price_avg",
        "office_sqm_2000", "church_count_5000","0_6_all", "detention_facility_raion", "cafe_avg_price_3000")
     :=NULL]



varnames <- setdiff(colnames(data), c("id","price_doc"))

cat("Create sparse matrix")
# To sparse matrix
train_sparse <- Matrix(as.matrix(sapply(data[price_doc > -1,varnames,with=FALSE],as.numeric)), sparse=TRUE)
test_sparse <- Matrix(as.matrix(sapply(data[is.na(price_doc),varnames,with=FALSE],as.numeric)), sparse=TRUE)
y_train <- data[!is.na(price_doc),price_doc]
test_ids <- data[is.na(price_doc),id]
dtrain <- xgb.DMatrix(data=train_sparse, label=y_train)
dtest <- xgb.DMatrix(data=test_sparse);
gc()

# Params for xgboost
param <- list(objective="reg:linear",
              eval_metric = "rmse",
              booster = "gbtree",
              eta = .05,
              gamma = 1,
              max_depth = 4,
              min_child_weight = 1,
              subsample = .7,
              colsample_bytree = .7
)


set.seed(555)
rounds = 501
mpreds = data.table(id=test_ids)

for(random.seed.num in 1:10) {
  print(paste("[", random.seed.num , "] training xgboost begin ",sep=""," : ",Sys.time()))
  set.seed(random.seed.num)
  xgb_model <- xgb.train(data = dtrain,
                         params = param,
                         watchlist = list(train = dtrain),
                         nrounds = rounds,
                         verbose = 1,
                         print.every.n = 5
                         # missing='NAN'
  )

  vpreds = predict(xgb_model,dtest)
  mpreds = cbind(mpreds, vpreds)
  colnames(mpreds)[random.seed.num+1] = paste("pred_seed_", random.seed.num, sep="")
}


mpreds_2 = mpreds[, id:= NULL]
mpreds_2 = mpreds_2[, price_doc := rowMeans(.SD)]
mpreds_2[, ':='(id = data.test$id, timestamp=data.test$timestamp)]

submission = data.table(id=test_ids, price_doc=exp(mpreds_2$price_doc)-1)

write.table(submission, "sberbank_submission_v03.csv", sep=",", dec=".", quote=FALSE, row.names=FALSE)


data.train1 = data.train[price_doc>1000000 & price_doc<111111112,]
nrow(data.train)


