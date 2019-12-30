import numpy as np
import pandas as pd

from src.utils.util import format_time


def stat_brand(data: pd.DataFrame) -> list:
    return data['牌号'].unique()


def stat_criterion(data: pd.DataFrame, round_length=3) -> dict:
    criterion = {}
    for brand in data['牌号'].unique():
        item_brand = data[data['牌号'] == brand]
        criterion[brand] = np.round(item_brand['出口水分设定值'].mean(), round_length)
    return criterion


def stat_brand_batch(data: pd.DataFrame) -> dict:
    data_per_brand = {}
    for brand in data['牌号'].unique():
        item_brand = data[data['牌号'] == brand]
        split_data_per_batch = []

        for batch in item_brand['批次'].unique():
            item_batch = item_brand[item_brand['批次'] == batch]

            item_batch['时间'] = item_batch['时间'].map(lambda x: format_time(x))
            item_batch = item_batch.sort_values(by=['时间'], ascending=True)

            split_data_per_batch.append(item_batch)

        data_per_brand[brand] = split_data_per_batch

    return data_per_brand
