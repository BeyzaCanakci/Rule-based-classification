#Rule -based classification
#Business Problem
#A gaming company wants to use some customer features to create level-based customer definitions (personas), and based on these new personas, create segments. Then, using these segments, they want to predict the average revenue potential of a new customer.

#Example: They want to determine how much on average a 25-year-old male user from Turkey who uses iOS would bring to the company.

#Dataset Story
#The persona.csv dataset contains the prices of products sold by an international gaming company and some demographic information about the users who purchased them.
#The dataset consists of records generated for each sales transaction, meaning it’s not unique — the same customer (with the same demographic characteristics) may appear multiple times.

#Variables:

#PRICE: Amount spent by the customer

#SOURCE: Device type the customer uses

#SEX: Gender of the customer

#COUNTRY: Country of the customer

#AGE: Age of the customer


#| PRICE | SOURCE  | SEX  | COUNTRY | AGE |
#| ----- | ------- | ---- | ------- | --- |
#| 39    | android | male | bra     | 17  |
#| 39    | android | male | bra     | 17  |
#| 49    | android | male | bra     | 17  |
#| 29    | android | male | tur     | 17  |
#| 49    | android | male | tur     | 17  |

#| customers\_level\_based      | PRICE   | SEGMENT |
#| ---------------------------- | ------- | ------- |
#| BRA\_ANDROID\_FEMALE\_0\_18  | 1139.80 | A       |
#| BRA\_ANDROID\_FEMALE\_19\_23 | 1070.60 | A       |
#| BRA\_ANDROID\_FEMALE\_24\_30 | 508.14  | A       |
#| BRA\_ANDROID\_FEMALE\_31\_40 | 233.16  | C       |
#| BRA\_ANDROID\_FEMALE\_41\_66 | 236.66  | C       |


import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# Load data
df = pd.read_csv("/Users/beyzacanakci/Desktop/miuul/Kural_Tabanlı_Sınıflandırma/persona.csv")

# question: How many unique SOURCE?
df["SOURCE"].nunique()
df["SOURCE"].unique()

# question: How many unique PRICE?

df["PRICE"].nunique()

# question: How many sales have been realized from countries?

df["COUNTRY"].value_counts()

# Question : What is mean of PRICE depend on the countries? 


mean_prices = df.groupby("COUNTRY")["PRICE"].mean().reset_index()
print(mean_prices.to_string(float_format='{:,.2f}'.format))


# Question: What is mean of PRICE depend on SOURCE?
print(df.groupby(by=['SOURCE'])["PRICE"].mean())

# Question 10: What is mean of PRICE group by COUNTRY-SOURCE?

df.groupby(['COUNTRY','SOURCE'])["PRICE"].mean()


# AIM 2 & 3: What is average PRICE groupby COUNTRY, SOURCE, SEX, AGE.  Group and sort
agg_df = df.groupby(["COUNTRY", "SOURCE", "SEX", "AGE"]).agg({"PRICE": "mean"}).sort_values("PRICE", ascending=False).reset_index()
agg_df

############################################################
# Task 4: Turn the names in the Index to the variable name.
############################################################
# All variables except PrICE in the output of the third problem are index names.
# Convert these names to variable names.
# Hint: reset_index ()
# agg_df.reset_index (inplace = true)

agg_df = agg_df.reset_index()
print(agg_df.to_string(float_format='{:,.2f}'.format))

# Task 5: Turn the AGE variable to categorical variables and add to AGG_DF.
############################################################
# Convert the Age numerical variable to categorical variables.
# Create the ranges as you think they will be convincing.
# For example: '0_18', '19_23', '24_30', '31_40', '41_70'

bins = [0, 18, 23, 30, 40, agg_df["AGE"].max()]
labels = ['0_18', '19_23', '24_30', '31_40', f'41_{agg_df["AGE"].max()}']
agg_df["age_cat"] = pd.cut(agg_df["AGE"], bins, labels=labels)
agg_df
# AIM 6: Create level-based customers
agg_df['customers_level_based'] = (
    agg_df['COUNTRY'].str.upper() + "_" +
    agg_df['SOURCE'].str.upper() + "_" +
    agg_df['SEX'].str.upper() + "_" +
    agg_df['age_cat'].astype(str)
)

# Ensure unique values and average PRICE per persona
agg_df = agg_df.groupby("customers_level_based").agg({"PRICE": "mean"}).reset_index()

# TASK 7: Segment customers by PRICE
agg_df["SEGMENT"] = pd.qcut(agg_df["PRICE"], 4, labels=["D", "C", "B", "A"])

# Describe segments
print(agg_df.groupby("SEGMENT").agg({"PRICE": ["mean", "max", "min", "count"]}))

# TASK 8: Predict for new customers
# Example 1: 33 years old, ANDROID, Turkish, Female
new_customer = "TUR_ANDROID_FEMALE_31_40"
result = agg_df[agg_df["customers_level_based"] == new_customer]
print(f"Customer: {new_customer} => Segment: {result['SEGMENT'].values[0]}, Expected Revenue: {result['PRICE'].values[0]:.2f}")

# Example 2: 35 years old, IOS, French, Female
new_customer2 = "FRA_IOS_FEMALE_31_40"
result2 = agg_df[agg_df["customers_level_based"] == new_customer2]
print(f"Customer: {new_customer2} => Segment: {result2['SEGMENT'].values[0]}, Expected Revenue: {result2['PRICE'].values[0]:.2f}")
