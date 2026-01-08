

Import Libraries
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

"""Load Dataset"""

events = pd.read_csv("events.csv")
item_properties = pd.read_csv("item_properties.csv")
category_tree = pd.read_csv("category_tree.csv")



print("Events Data Info:")
print(events.info())
print("\nSample Events Data:")
print(events.head())

# Remove duplicate rows
events.drop_duplicates(inplace=True)

# Remove rows with missing user or item IDs
events.dropna(subset=['visitorid', 'itemid'], inplace=True)

# Convert timestamp to datetime format
events['timestamp'] = pd.to_datetime(events['timestamp'], unit='ms')


# Remove users with very few interactions (noise reduction)

user_interaction_count = events['visitorid'].value_counts()
active_users = user_interaction_count[user_interaction_count > 5].index
events = events[events['visitorid'].isin(active_users)]


# Assign weights to interaction types

interaction_weights = {
    'view': 1,
    'addtocart': 2,
    'transaction': 3
}

events['interaction_score'] = events['event'].map(interaction_weights)

# Remove rows with unknown interaction types (if any)
events.dropna(subset=['interaction_score'], inplace=True)


# Aggregate interactions per user-item pair

interaction_df = events.groupby(
    ['visitorid', 'itemid']
)['interaction_score'].sum().reset_index()

print("\nUser-Item Interaction Table:")
print(interaction_df.head())


# Build User–Item Interaction Matrix


user_item_matrix = interaction_df.pivot_table(
    index='visitorid',
    columns='itemid',
    values='interaction_score',
    fill_value=0
)

print("\nUser-Item Interaction Matrix Shape:")
print(user_item_matrix.shape)


# Encode User and Item IDs

user_encoder = LabelEncoder()
item_encoder = LabelEncoder()

interaction_df['user_id'] = user_encoder.fit_transform(interaction_df['visitorid'])
interaction_df['item_id'] = item_encoder.fit_transform(interaction_df['itemid'])

# Save Cleaned & Processed Data


events.to_csv("cleaned_events.csv", index=False)
interaction_df.to_csv("user_item_interactions.csv", index=False)
user_item_matrix.to_csv("user_item_matrix.csv")
