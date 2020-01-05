from datetime import datetime
from typing import Tuple

import pandas as pd
import numpy as np
from scipy.stats import skew
from scipy.stats import kurtosis

from src.utils.util import format_time

columns = [
    '时间', '批次', '牌号', '设备状态',
    '入口水分', '出口水分', '出口水分设定值', '烘前叶丝流量累积量',
    '热风速度设定值', '热风速度实际值',
    '烘前叶丝流量设定值', '烘前叶丝流量',
    '筒壁1区温度设定值', '筒壁1区温度实际值',
    '筒壁2区温度实际值', '筒壁2区温度设定值', '罩压力'
]

feature_column = ['出口水分差值', '热风速度实际值', '入口水分', '罩压力', '筒壁1区温度实际值', '筒壁1区温度设定值', '筒壁2区温度实际值', '筒壁2区温度设定值']
label_column = ['筒壁1区温度设定值', '筒壁2区温度设定值']

STABLE_WINDOWS_SIZE = 10  # 稳态的时长
SPLIT_NUM = 10  # 特征选取分割区间的数量（需要被FEATURE_RANGE整除）
TIME_IN_ROLLER = 70  # 烟丝在一个滚筒的时间
MODEL_CRITERION = 0.05  # 模型标准，工艺标准为0.5
FEATURE_RANGE = 70  # 特征选取的区间范围
LABEL_RANGE = 10  # Label选取的区间范围
SETTING_LAG = 20  # 设定值和实际值的时延
REACTION_LAG = 10  # 实际值调整后，水分变化的时延

MODEL_TRANSITION_CRITERION = 0.1
TRANSITION_FEATURE_RANGE = 20  # Transition 特征选取的区间范围
TRANSITION_SPLIT_NUM = 4  # Transition 特征选取分割区间的数量
STABLE_UNAVAILABLE = 120  # 出口水分不可用阶段
TRANSITION_SIZE = 400  # 定义 Transition 的长度

MODEL_HEAD_CRITERION = 0.25

VERBOSE = True


def read_data(filename: str) -> pd.DataFrame:
    """
    read data from local disk or cloud
    :param filename: filename to read
    :return: DataFrame
    """
    original = pd.read_csv(filename, encoding='gbk')
    return original[columns]


def data_clean(data: pd.DataFrame) -> pd.DataFrame:
    """
    clean dirty data
    :param data: original data
    :return: cleaned data
    """

    # 1. Drop nan
    data = data.dropna()

    # 2. 牌号莫名奇妙的存储时间，drop这些行
    index = data[[isinstance(item, str) and item.startswith('2019') for item in data['牌号']]].index
    data = data.drop(index, axis=0)

    # 3. 烘前叶丝流量 == 0，表示设备没有运行
    index = data[data['烘前叶丝流量'] == 0].index
    data = data.drop(index, axis=0)

    # 4. 只考虑 '准备', '启动', '生产', '收尾' 四个状态的数据
    data = data[data['设备状态'].isin(['准备', '启动', '生产', '收尾'])]

    # 5. 过滤批次为 '000'的数据
    data = data[data['批次'] != '000']

    # 6. 计算出口水分差值
    data['出口水分差值'] = data['出口水分'] - data['出口水分设定值']
    return data


def split_data_by_brand(data: pd.DataFrame) -> Tuple[dict, dict]:
    """
    split the continuous time series data into each brand and batch (牌号和批次)
    :param data: the continuous time series data
    :return: data after split, data_per_brand[i][j] means i-th brand and j-th batch
    """

    data_per_brand = {}
    criterion = {}

    for brand in data['牌号'].unique():
        item_brand = data[data['牌号'] == brand]
        data_per_batch = []

        for batch in item_brand['批次'].unique():
            item_batch = item_brand[item_brand['批次'] == batch]

            item_batch['时间'] = item_batch['时间'].map(lambda x: format_time(x))
            item_batch = item_batch.sort_values(by=['时间'], ascending=True)

            data_per_batch.append(item_batch)

        criterion[brand] = item_brand['出口水分设定值'].mean()
        data_per_brand[brand] = data_per_batch

    return data_per_brand, criterion


def calc_feature(item_: pd.DataFrame, feature_end: int, feature_range: int, split_num: int) -> np.array:
    """
    calc feature for each sample data
    :param item_: sample data
    :param feature_end: the end time to calc feature
    :param feature_range: feature calc range
    :param split_num: how many splits after splitting
    :return: feature array
    """
    feature_start = feature_end - feature_range

    feature_slice = item_[feature_column].iloc[feature_start: feature_end].values

    # shape = (SPLIT_NUM, FEATURE_RANGE / SPLIT_NUM, FEATURE_NUM)
    feature_slice = np.array(np.vsplit(feature_slice, split_num))

    # shape = (*, SPLIT_NUM, FEATURE_NUM)
    feature = np.concatenate([
        np.mean(feature_slice, axis=1).ravel(),
        np.std(feature_slice, axis=1).ravel(),
        calc_integral(feature_slice).ravel(),
        skew(feature_slice, axis=1).ravel(),
        kurtosis(feature_slice, axis=1).ravel(),
    ])

    return feature.ravel()


def rolling_window(data_: np.array, window: int) -> np.array:
    """
    slide window from start to end of 1D array,
    generate a 2D array which shape = (len(data_) - 1, window)
    :param data_: 1D array
    :param window: slide window size
    :return: 2D array
    """
    shape = data_.shape[:-1] + (data_.shape[-1] - window + 1, window)
    strides = data_.strides + (data_.strides[-1],)
    return np.lib.stride_tricks.as_strided(data_, shape=shape, strides=strides)


def calc_integral(data_: np.array) -> np.array:
    """
    calc integral
    :param data_: shape = (SPLIT_NUM, FEATURE_RANGE / SPLIT_NUM, FEATURE_NUM)
    :return shape = (SPLIT_NUM, FEATURE_NUM), each value is the integral
    """
    if data_.shape[0] <= 1:
        return 0
    sum_ = np.sum(data_, axis=1)
    return sum_ - (data_[:, 0, :] + data_[:, data_.shape[1] - 1, :]) / 2


def calc_label(item_: pd.DataFrame, start: int, end: int) -> np.array:
    """
    calc label for each sample
    :param item_: sample data
    :param start: the start time to calc label
    :param end: the end time to calc label
    :return: a array with exactly 2 number: temperature of region 1 and temperature of region 2
    """
    return np.mean(item_[label_column].iloc[start: end].values, axis=0)


def calc_current(item_: pd.DataFrame, start: int) -> np.array:
    """
    calc current value for evaluation use
    :param item_: sample data
    :param start: the start time for region 2 to calc current value
    :return: a array with exactly 2 number: current temperature of region 1 and current temperature of region 2
    """
    return item_[label_column].iloc[start].values


def calc_delta(item_: pd.DataFrame, start: int, end: int) -> np.array:
    label_ = calc_label(item_, start, end)
    delta = label_ - item_[label_column].iloc[start]
    return delta.values


def generate_brand_transition_training_data(item_brand, brand_index, setting) -> Tuple[np.array, np.array, np.array, np.array]:
    """
    generate training data and label for one brand in 'transition' stage
    this method is time consuming
    :param item_brand: the brand data to generate
    :param brand_index: brand index
    :param setting: the setting value
    :return:
        brand_train_data: all training data for this brand, shape=(N, M),
            which N denotes the number of training data, M denotes the number of feature
        brand_train_label: all training label for this brand, shape=(N, 2)
        brand_delta: delta info
        brand_mapping: mapping info
    """
    brand_train_data = []
    brand_train_label = []
    brand_delta = []
    brand_mapping = []

    for batch_index, item_batch in enumerate(item_brand):
        item_batch = item_batch.iloc[:TRANSITION_SIZE, :]
        item_batch = item_batch.reset_index(drop=True)
        length = len(item_batch)
        humidity = item_batch['出口水分'].values

        stable_index = np.abs(humidity - setting) < MODEL_TRANSITION_CRITERION
        # No stable area in this batch
        if np.sum(stable_index) == 0 or len(stable_index) < STABLE_WINDOWS_SIZE:
            continue
        stable_index = np.sum(rolling_window(stable_index, STABLE_WINDOWS_SIZE), axis=1)
        stable_index = np.where(stable_index == STABLE_WINDOWS_SIZE)[0]

        range_start = STABLE_UNAVAILABLE + TRANSITION_FEATURE_RANGE
        range_end = length - STABLE_WINDOWS_SIZE

        for stable_start in stable_index:
            if stable_start < range_start or stable_start >= range_end:
                continue
            adjust_end = stable_start - REACTION_LAG - SETTING_LAG
            adjust_start = adjust_end - LABEL_RANGE

            # store feature
            brand_train_data.append(
                calc_feature(
                    item_batch,
                    adjust_start,
                    TRANSITION_FEATURE_RANGE,
                    TRANSITION_SPLIT_NUM
                )
            )

            # store label
            brand_train_label.append(
                calc_label(
                    item_batch,
                    adjust_start,
                    adjust_end
                )
            )

            # store mapping info
            brand_mapping.append([
                brand_index,
                batch_index,
                adjust_start,
                adjust_end,
                stable_start
            ])

            # store delta value
            brand_delta.append(
                calc_delta(
                    item_batch,
                    adjust_start,
                    adjust_end,
                )
            )

    brand_train_data = np.array(brand_train_data)
    brand_train_label = np.array(brand_train_label)
    brand_mapping = np.array(brand_mapping)
    brand_delta = np.array(brand_delta)

    return brand_train_data, brand_train_label, brand_delta, brand_mapping,


def generate_brand_produce_training_data(item_brand, brand_index, setting) -> Tuple[np.array, np.array, np.array, np.array]:
    """
    generate training data and label for one brand in 'produce' stage
    this method is time consuming
    :param item_brand: the brand data to generate
    :param brand_index: brand index
    :param setting: the setting value
    :return:
        brand_train_data: all training data for this brand, shape=(N, M),
            which N denotes the number of training data, M denotes the number of feature
        brand_train_label: all training label for this brand, shape=(N, 2)
        brand_delta: delta info
        brand_mapping: mapping info
    """
    brand_train_data = []
    brand_train_label = []
    brand_delta = []
    brand_mapping = []

    for batch_index, item_batch in enumerate(item_brand):
        item_batch = item_batch.iloc[TRANSITION_SIZE:, :]
        item_batch = item_batch.reset_index(drop=True)
        length = len(item_batch)
        humidity = item_batch['出口水分'].values

        stable_index = np.abs(humidity - setting) < MODEL_CRITERION
        # No stable area in this batch
        if np.sum(stable_index) == 0 or len(stable_index) < STABLE_WINDOWS_SIZE:
            continue
        stable_index = np.sum(rolling_window(stable_index, STABLE_WINDOWS_SIZE), axis=1)
        stable_index = np.where(stable_index == STABLE_WINDOWS_SIZE)[0]

        range_start = REACTION_LAG + SETTING_LAG + LABEL_RANGE + FEATURE_RANGE
        range_end = length - STABLE_WINDOWS_SIZE

        for stable_start in stable_index:
            if stable_start < range_start or stable_start >= range_end:
                continue
            adjust_end = stable_start - REACTION_LAG - SETTING_LAG
            adjust_start = adjust_end - LABEL_RANGE

            # store feature
            brand_train_data.append(
                calc_feature(
                    item_batch,
                    adjust_start,
                    FEATURE_RANGE,
                    SPLIT_NUM
                )
            )

            # store label
            brand_train_label.append(
                calc_label(
                    item_batch,
                    adjust_start,
                    adjust_end,
                )
            )

            # store mapping info
            brand_mapping.append([
                brand_index,
                batch_index,
                adjust_start,
                adjust_end,
                stable_start
            ])

            # store delta value
            brand_delta.append(
                calc_delta(
                    item_batch,
                    adjust_start,
                    adjust_end
                )
            )

    brand_train_data = np.array(brand_train_data)
    brand_train_label = np.array(brand_train_label)
    brand_mapping = np.array(brand_mapping)
    brand_delta = np.array(brand_delta)

    return brand_train_data, brand_train_label, brand_delta, brand_mapping,


def generate_head_dict(data_per_brand: dict, criterion: dict, round_: bool = True) -> Tuple[dict, dict]:
    """
    generate 2 dict represent init value and stable value for each brand
    :param data_per_brand: all brand data
    :param criterion: criterion for each brand
    :param round_: whether round result
    :return:
        init_per_brand: init value for each brand
        stable_per_brand: stable value for each brand
    """
    init_value = {}
    stable_value = {}
    total_lag = SETTING_LAG + REACTION_LAG
    for brand in data_per_brand['牌号'].unique():
        item_brand = data_per_brand[data_per_brand['牌号'] == brand]

        init_per_batch = []
        stable_per_batch = []

        for batch in item_brand['批次'].unique():
            item_batch = item_brand[item_brand['批次'] == batch]
            if len(item_batch) <= 5:
                continue
            item_batch['时间'] = item_batch['时间'].map(lambda x: format_time(x))
            item_batch = item_batch.sort_values(by=['时间'], ascending=True)

            # calc init value
            init_per_batch.append([item_batch['筒壁1区温度设定值'].iloc[0], item_batch['筒壁2区温度设定值'].iloc[0]])

            # calc stable value
            humidity = item_batch['出口水分']
            end = len(item_batch) - 1
            start = len(item_batch) - 1
            for i in range(len(item_batch) - 1, 1, -1):
                if np.abs(humidity.iloc[i] - criterion[brand]) <= MODEL_HEAD_CRITERION:
                    start -= 1
                else:
                    break
            if start == end or start < total_lag:
                continue

            stable_one = np.mean(item_batch['筒壁1区温度设定值'].iloc[start - total_lag: end - total_lag])
            stable_two = np.mean(item_batch['筒壁2区温度设定值'].iloc[start - total_lag: end - total_lag])
            stable_per_batch.append([stable_one, stable_two])

        init_value[brand] = init_per_batch
        stable_value[brand] = stable_per_batch

    init_per_brand = {}
    stable_per_brand = {}

    for index, item in enumerate(init_value):
        init_per_brand[item] = np.round(np.mean(init_value[item], axis=0), 2) if round_ else np.mean(init_value[item], axis=0)

    for index, item in enumerate(stable_value):
        stable_per_brand[item] = np.round(np.mean(stable_value[item], axis=0), 2) if round_ else np.mean(stable_value[item], axis=0)

    return init_per_brand, stable_per_brand


def generate_all_training_data(data_per_brand: dict, criterion: dict, stage: str) -> Tuple[np.array, np.array, np.array, np.array]:
    """
    generate training data and label for all brand
    :param data_per_brand: all brand data
    :param criterion: criterion for each brand
    :param stage: including 'transition', 'produce'
    :return:
        train_data: all training data, shape=(N, M),
            which N denotes the number of training data, M denotes the number of feature
        train_label: all training label, shape=(N, 2)
        delta: delta info
        mapping: mapping info
    """
    train_data_list = []
    train_label_list = []
    delta_list = []
    mapping_list = []
    if stage == 'produce':
        for brand_index, brand in enumerate(data_per_brand):
            start = datetime.now()

            brand_train_data, brand_train_label, brand_delta, brand_mapping = generate_brand_produce_training_data(
                data_per_brand[brand],
                brand_index,
                criterion[brand]
            )
            train_data_list.append(brand_train_data)
            train_label_list.append(brand_train_label)
            delta_list.append(brand_delta)
            mapping_list.append(brand_mapping)

            if VERBOSE:
                print('{}: len={}, time={}s'.format(brand, len(brand_train_data), (datetime.now() - start).seconds))
        return concatenate(train_data_list), concatenate(train_label_list), concatenate(delta_list), concatenate(mapping_list)

    elif stage == 'transition':
        for brand_index, brand in enumerate(data_per_brand):
            start = datetime.now()

            brand_train_data, brand_train_label, brand_delta, brand_mapping = generate_brand_transition_training_data(
                data_per_brand[brand],
                brand_index,
                criterion[brand]
            )
            train_data_list.append(brand_train_data)
            train_label_list.append(brand_train_label)
            delta_list.append(brand_delta)
            mapping_list.append(brand_mapping)

            if VERBOSE:
                print('{}: len={}, time={}s'.format(brand, len(brand_train_data), (datetime.now() - start).seconds))
        return concatenate(train_data_list), concatenate(train_label_list), concatenate(delta_list), concatenate(mapping_list)
    else:
        raise Exception('stage must in [transition, produce], now is ' + str(stage))


def concatenate(data_: list) -> np.array:
    """
    concatenate list with item of different length
    """
    result = data_[0]
    for i in range(1, len(data_)):
        if len(data_[i]) is not 0:
            result = np.concatenate([result, data_[i]], axis=0)
    return result


def predict_head(data_per_batch: pd.DataFrame, init_per_brand: dict, stable_per_brand: dict) -> np.array:
    """
    predict in head stage
    :param data_per_batch: one batch data
    :param init_per_brand: init value for each brand
    :param stable_per_brand: stable value for each brand
    :return: predicted value in head stage
    """
    range_ = STABLE_UNAVAILABLE + TRANSITION_FEATURE_RANGE
    number = data_per_batch['牌号'].iloc[0]
    pred_head_one = []
    pred_head_two = []
    for i in range(range_):
        if i < 16:
            pred_head_one.append(init_per_brand[number])
        else:
            pred_head_one.append(stable_per_brand[number][0])

        if i < 58:
            pred_head_two.append(init_per_brand[number])
        else:
            pred_head_two.append(stable_per_brand[number][1])

    pred_head_one = np.round(pred_head_one, 3)
    pred_head_two = np.round(pred_head_two, 3)
    return np.array([pred_head_one, pred_head_two]).T


def generate_validation_data(data_per_batch: pd.DataFrame, stage: str) -> Tuple[np.array, np.array]:
    """
    generate real test data for one batch data at each time t
    :param data_per_batch: one batch data
    :param stage: including 'transition', 'produce'
    :return:
        final_X_test: shape=(N, M), which N denotes the number of training data, M denotes the number of feature
        final_mapping: mapping info
    """

    final_X_test = []
    final_mapping = []

    if stage == 'transition':
        data_per_batch = data_per_batch.iloc[STABLE_UNAVAILABLE: TRANSITION_SIZE]
        length = len(data_per_batch)
        for item_index in range(TRANSITION_FEATURE_RANGE, length, 1):
            adjust_start = item_index

            final_X_test.append(
                calc_feature(
                    data_per_batch,
                    adjust_start,
                    TRANSITION_FEATURE_RANGE,
                    TRANSITION_SPLIT_NUM
                )
            )
            final_mapping.append(adjust_start)
        return np.array(final_X_test), np.array(final_mapping)
    elif stage == 'produce':
        data_per_batch = data_per_batch.iloc[TRANSITION_SIZE - FEATURE_RANGE:]
        length = len(data_per_batch)
        for item_index in range(FEATURE_RANGE, length, 1):
            adjust_start = item_index

            final_X_test.append(
                calc_feature(
                    data_per_batch,
                    adjust_start,
                    FEATURE_RANGE,
                    SPLIT_NUM
                )
            )
            final_mapping.append(adjust_start)
        return np.array(final_X_test), np.array(final_mapping)
    else:
        raise Exception('stage must in [transition, produce], now is ' + str(stage))
