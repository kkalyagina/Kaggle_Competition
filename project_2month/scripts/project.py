#Import dependencies
import pandas as pd
import numpy as np
from datetime import datetime, date
import warnings
from modules.eda_methods import barplot
from modules.eda_methods import merge_table
from modules.missing_value import get_mean_na_part
from modules.missing_value import plot_missing_values_heatmap
from modules.missing_value import iterative
from modules.model import model_0
from modules.model import ses
from modules.model import gen_par
from modules.model import sarima

#Download dataset
sales = pd.read_csv("./competitive-data-science-predict-future-sales/sales_train.csv")
items = pd.read_csv("./competitive-data-science-predict-future-sales/items.csv")
categories = pd.read_csv("./competitive-data-science-predict-future-sales/item_categories.csv")
shops = pd.read_csv("./competitive-data-science-predict-future-sales/shops.csv")
test = pd.read_csv("./competitive-data-science-predict-future-sales/test.csv")

#Create special table for building categories plot
cat_graph=categories.copy(deep=True)
cat_graph['item_category']=cat_graph['item_category_name'].str.split('-')
cat_graph['category']=cat_graph['item_category'].apply(lambda x: x[0].strip())
cat_graph['item'] = cat_graph['item_category'].apply(lambda x: x[1].strip() if len(x) > 1 else x[0].strip())
cat_graph.drop(['item_category_name','item_category'], inplace=True, axis=1)

plot_categories=plot(barplot,
                    'Count of items in different categories', 
                    'item_category_id', 
                    'category', 
                    cat_graph, 
                    np.sum, 
                    'lightblue', 
                    'Count of items', 
                    'Categories')

#Merge train, test and other tables
train=merge_table(sales, items, "item_id", "item_category_id", "item_id")
test=merge_table(test, items, "item_id", "item_category_id", "item_id")

#Format column of date
train["date"] = pd.to_datetime(train["date"], format="%d.%m.%Y")

test["date_block_num"] = 34

total_table_day=plot(lineplot, 
                    'Item quantity per day'
                    'date', 
                    "item_cnt_day", 
                    train, 
                    np.sum,
                    'blue',
                    'Date', 
                    'Count of items')

train['year_month'] = train['date'].astype('datetime64[M]')

total_table_month=plot(lineplot, 
                      'Item quantity per day', 
                      'year_month', 
                      "item_cnt_day", 
                      train, 
                      np.sum,
                      'blue',
                      'Date', 
                      'Count of items')

plot_shop=plot(barplot,
               "Count of items in each shops",
               "shop_id", 
               "item_cnt_day", 
               train, 
               np.sum, 
               'g',
               'shop_id',
               'Count of items')

#Detalization on shop/category/item
train_super_id_sales_items = train[['shop_id','item_category_id', 'item_id', 'year_month', 'item_cnt_day']].groupby(['shop_id','item_category_id', 'item_id', 'year_month']).sum()

train_super_id_sales_items = train_super_id_sales_items.unstack('year_month', fill_value=np.nan)

train_super_id_sales_items.columns = train_super_id_sales_items.columns.droplevel()

get_mean_na_part(train_super_id_sales_items)

#Detalization on shop/category
train_super_id_sales_category = train[['shop_id','item_category_id', 'year_month', 'item_cnt_day']].groupby(['shop_id','item_category_id', 'year_month']).sum()

train_super_id_sales_category = train_super_id_sales_category.unstack('year_month', fill_value=np.nan)

train_super_id_sales_category.columns = train_super_id_sales_category.columns.droplevel()

get_mean_na_part(train_super_id_sales_category)

#Detalization on shop
train_super_id_sales_shop = train[['shop_id', 'year_month', 'item_cnt_day']].groupby(['shop_id', 'year_month']).sum()

train_super_id_sales_shop = train_super_id_sales_shop.unstack('year_month', fill_value=np.nan)

train_super_id_sales_shop.columns = train_super_id_sales_shop.columns.droplevel()

get_mean_na_part(train_super_id_sales_shop)

#Plots of detalization on shops
plot_shop_1=plot(lineplot, 
                'Shop_1', 
                'year_month', 
                train_super_id_sales_shop.T[2], 
                train_super_id_sales_shop.T[2], 
                np.sum,
                'blue'
                'Date', 
                'Count of items per month')

plot_shop_2=plot(lineplot, 
                'Shop_2', 
                'year_month', 
                train_super_id_sales_shop.T[3], 
                train_super_id_sales_shop.T[3], 
                np.sum,
                'blue'
                'Date', 
                'Count of items per month')

plot_shop_3=plot(lineplot, 
                'Shop_3', 
                'year_month', 
                train_super_id_sales_shop.T[4], 
                train_super_id_sales_shop.T[4], 
                np.sum,
                'blue'
                'Date', 
                'Count of items per month')

plot_shop_4=plot(lineplot, 
                'Shop_4', 
                'year_month', 
                train_super_id_sales_shop.T[5], 
                train_super_id_sales_shop.T[5], 
                np.sum,
                'blue'
                'Date', 
                'Count of items per month')

#Plots of detalization on shop/category
plot_category_1=plot(lineplot, 
                'Category_1', 
                'year_month', 
                train_super_id_sales_category[59][25], 
                train_super_id_sales_category[59][25], 
                np.sum,
                'blue'
                'Date', 
                'Count of items per month')

plot_category_2=plot(lineplot, 
                'Category_2', 
                'year_month', 
                train_super_id_sales_category[45][25], 
                train_super_id_sales_category[45][25], 
                np.sum,
                'blue'
                'Date', 
                'Count of items per month')

plot_category_3=plot(lineplot, 
                'Category_3', 
                'year_month', 
                train_super_id_sales_category[59][83], 
                train_super_id_sales_category[59][83], 
                np.sum,
                'blue'
                'Date', 
                'Count of items per month')

plot_category_4=plot(lineplot, 
                'Category_4', 
                'year_month', 
                train_super_id_sales_category[25][25], 
                train_super_id_sales_category[25][25], 
                np.sum,
                'blue'
                'Date', 
                'Count of items per month')

#Plots of detalization on shop/category/item
plot_item_1=plot(lineplot, 
                'Item_1', 
                'year_month', 
                train_super_id_sales_items.T[59][83][22087], 
                train_super_id_sales_items.T[59][83][22087], 
                np.sum,
                'blue'
                'Date', 
                'Count of items per month')

plot_item_2=plot(lineplot, 
                'Item_2', 
                'year_month', 
                train_super_id_sales_items.T[18][79][17717], 
                train_super_id_sales_items.T[18][79][17717], 
                np.sum,
                'blue'
                'Date', 
                'Count of items per month')

plot_item_3=plot(lineplot, 
                'Item_3', 
                'year_month', 
                train_super_id_sales_items.T[28][2][5572], 
                train_super_id_sales_items.T[28][2][5572], 
                np.sum,
                'blue'
                'Date', 
                'Count of items per month')

plot_item_4=plot(lineplot, 
                'Item_4', 
                'year_month', 
                train_super_id_sales_items.T[15][2][5643], 
                train_super_id_sales_items.T[15][2][5643], 
                np.sum,
                'blue'
                'Date', 
                'Count of items per month')

#Plot of shop missing value
plot_missing_values_heatmap(train_super_id_sales_shop)

#Filling in NaN

#Option 1
train_super_id_sales_shop=train_super_id_sales_shop.fillna(round(train_super_id_sales_shop.mean(axis=0)))

#Option 2
train_super_id_sales_items = train_super_id_sales_items.reset_index(level=0, drop=True)
train_super_id_sales_items = train_super_id_sales_items.groupby(['item_category_id']).sum() / train_super_id_sales_items.groupby(['item_category_id']).count()
train_super_id_sales_items = train_super_id_sales_items.fillna(method='ffill').fillna(method='bfill')

train_super_id_sales_category_FILLED = train_super_id_sales_category.fillna(train_super_id_sales_items)

train_super_id_sales_shop_category_FILLED=train_super_id_sales_category_FILLED.reset_index().drop(columns=['item_category_id']).groupby(['shop_id']).sum()

#Option 3
train_super_id_sales_shop_iterative=iterative(train_super_id_sales_shop.T)

#Model 0

#Option 1
m0_option_1=model_0(train_super_id_sales_shop.T)
sum(mae_cv)/len(mae_cv), sum(rmse_cv)/len(rmse_cv)

#Option 2
m0_option_2=model_0(train_super_id_sales_shop_category_FILLED.T)
sum(mae_cv)/len(mae_cv), sum(rmse_cv)/len(rmse_cv)

#Option 3
m0_option_3=model_0(train_super_id_sales_shop_iterative)
sum(mae_cv)/len(mae_cv), sum(rmse_cv)/len(rmse_cv)

#SimpleExpSmoothing

#Option 1
ses_option_1=ses(train_super_id_sales_shop.T)
sum(mae_cv)/len(mae_cv), sum(rmse_cv)/len(rmse_cv)

#Option 2
ses_option_2=ses(train_super_id_sales_shop_category_FILLED.T)
sum(mae_cv)/len(mae_cv), sum(rmse_cv)/len(rmse_cv)

#Option 3
ses_option_3=ses(train_super_id_sales_shop_iterative)
sum(mae_cv)/len(mae_cv), sum(rmse_cv)/len(rmse_cv)

#SARIMA
warnings.filterwarnings("ignore")

#Generation of combinations of seasonal parameters p, q and q
par_option_1=gen_par(train_super_id_sales_shop.T)
par_option_2=gen_par(train_super_id_sales_shop_category_FILLED.T)
par_option_3=gen_par(train_super_id_sales_shop_iterative)

sarima_option_1=sarima(train_super_id_sales_shop.T, (0, 1, 0), (1, 0, 0, 12))
sum(mae_cv)/len(mae_cv), sum(rmse_cv)/len(rmse_cv)
# results.plot_diagnostics(figsize=(15, 12))
# plt.show()

sarima_option_2=sarima(train_super_id_sales_shop_category_FILLED.T, (0, 1, 0), (0, 1, 0, 12))
sum(mae_cv)/len(mae_cv), sum(rmse_cv)/len(rmse_cv)
# results.plot_diagnostics(figsize=(15, 12))
# plt.show()

sarima_option_3=sarima(train_super_id_sales_shop_iterative, (1, 0, 1), (1, 0, 0, 12))
sum(mae_cv)/len(mae_cv), sum(rmse_cv)/len(rmse_cv)

#print(results.summary())

# results.plot_diagnostics(figsize=(15, 12))
# plt.show()
