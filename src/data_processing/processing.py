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

feature_column = ['出口水分差值', '热风速度实际值', '入口水分', '罩压力']
feature_column_1 = ['筒壁1区温度实际值', '筒壁1区温度设定值'] + feature_column
feature_column_2 = ['筒壁2区温度实际值', '筒壁2区温度设定值'] + feature_column
label_column_1 = ['筒壁1区温度设定值']
label_column_2 = ['筒壁2区温度设定值']

STABLE_WINDOWS_SIZE = 10  # 稳态的时长
SPLIT_NUM = 10  # 特征选取分割区间的数量（需要被FEATURE_RANGE整除）
TIME_IN_ROLLER = 70  # 烟丝在一个滚筒的时间
MODEL_CRITERION = 0.05  # 模型标准，工艺标准为0.5
FEATURE_RANGE = 80  # 特征选取的区间范围
LABEL_RANGE = 20  # Label选取的区间范围
SETTING_LAG = 20  # 设定值和实际值的时延
REACTION_LAG = 10  # 实际值调整后，水分变化的时延

MODEL_TRANSITION_CRITERION = 0.1
TRANSITION_FEATURE_RANGE = 20  # Transition 特征选取的区间范围
TRANSITION_LABEL_RANGE = 10  # Transition Label 选取的区间范围
TRANSITION_SPLIT_NUM = 4  # Transition 特征选取分割区间的数量
STABLE_UNAVAILABLE = 200  # 出口水分不可用阶段
TRANSITION_SIZE = 400  # 定义 Transition 的长度

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


def calc_feature(item_: pd.DataFrame, feature_end_1: int, feature_end_2: int, feature_range: int, split_num: int) -> np.array:
    """
    calc feature for each sample data
    :param item_: sample data
    :param feature_end_1: the end time for region 1 to calc feature
    :param feature_end_2: the end time for region 2 to calc feature
    :param feature_range: feature calc range
    :param split_num: how many splits after splitting
    :return: feature array
    """
    feature_start_1 = feature_end_1 - feature_range
    feature_start_2 = feature_end_2 - feature_range

    feature_slice_1 = item_[feature_column_1].iloc[feature_start_1: feature_end_1].values
    feature_slice_2 = item_[feature_column_2].iloc[feature_start_2: feature_end_2].values

    # shape = (SPLIT_NUM, FEATURE_RANGE / SPLIT_NUM, FEATURE_NUM)
    feature_slice_1 = np.array(np.vsplit(feature_slice_1, split_num))
    feature_slice_2 = np.array(np.vsplit(feature_slice_2, split_num))

    # shape = (*, SPLIT_NUM, FEATURE_NUM)
    feature = np.concatenate([
        np.mean(feature_slice_1, axis=1).ravel(),
        np.std(feature_slice_1, axis=1).ravel(),
        calc_integral(feature_slice_1).ravel(),
        skew(feature_slice_1, axis=1).ravel(),
        kurtosis(feature_slice_1, axis=1).ravel(),
        np.mean(feature_slice_2, axis=1).ravel(),
        np.std(feature_slice_2, axis=1).ravel(),
        calc_integral(feature_slice_2).ravel(),
        skew(feature_slice_2, axis=1).ravel(),
        kurtosis(feature_slice_2, axis=1).ravel()
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


def calc_label(item_: pd.DataFrame, start_1: int, end_1: int, start_2: int, end_2: int) -> np.array:
    """
    calc label for each sample
    :param item_: sample data
    :param start_1: the start time for region 1 to calc label
    :param end_1: the end time for region 1 to calc label
    :param start_2: the start time for region 2 to calc label
    :param end_2: the end time for region 2 to calc label
    :return: a array with exactly 2 number: temperature of region 1 and temperature of region 2
    """
    mean_1 = np.mean(item_[label_column_1].iloc[start_1: end_1].values)
    mean_2 = np.mean(item_[label_column_2].iloc[start_2: end_2].values)
    return np.array([mean_1, mean_2])


def calc_current(item_: pd.DataFrame, start_1: int, start_2: int) -> np.array:
    """
    calc current value for evaluation use
    :param item_: sample data
    :param start_1: the start time for region 1 to calc current value
    :param start_2: the start time for region 2 to calc current value
    :return:  a array with exactly 2 number: current temperature of region 1 and current temperature of region 2
    """
    return np.array([item_[label_column_1].iloc[start_1].values[0], item_[label_column_2].iloc[start_2].values[0]])


def calc_delta(item_: pd.DataFrame, start_1: int, end_1: int, start_2: int, end_2: int) -> np.array:
    label_ = calc_label(item_, start_1, end_1, start_2, end_2)
    delta_1 = label_[0] - item_[label_column_1].iloc[start_1]
    delta_2 = label_[1] - item_[label_column_2].iloc[start_2]
    return np.array([delta_1, delta_2]).ravel()


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

        for stable_start in stable_index:
            if stable_start < STABLE_UNAVAILABLE + TRANSITION_FEATURE_RANGE or stable_start >= length - STABLE_WINDOWS_SIZE:
                continue
            adjust_end_2 = stable_start - REACTION_LAG - SETTING_LAG
            adjust_start_2 = adjust_end_2 - LABEL_RANGE

            adjust_end_1 = adjust_end_2 - TIME_IN_ROLLER
            adjust_start_1 = adjust_start_2 - TIME_IN_ROLLER

            # store feature
            brand_train_data.append(
                calc_feature(
                    item_batch,
                    adjust_start_1,
                    adjust_start_2,
                    TRANSITION_FEATURE_RANGE,
                    TRANSITION_SPLIT_NUM
                )
            )

            # store label
            brand_train_label.append(
                calc_label(
                    item_batch,
                    adjust_start_1,
                    adjust_end_1,
                    adjust_start_2,
                    adjust_end_2
                )
            )

            # store mapping info
            brand_mapping.append([
                brand_index,
                batch_index,
                adjust_start_1,
                adjust_end_1,
                adjust_start_2,
                adjust_end_2,
                stable_start
            ])

            # store delta value
            brand_delta.append(
                calc_delta(
                    item_batch,
                    adjust_start_1,
                    adjust_end_1,
                    adjust_start_2,
                    adjust_end_2
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

        for stable_start in stable_index:
            if stable_start < REACTION_LAG + TIME_IN_ROLLER * 3 or stable_start >= length - STABLE_WINDOWS_SIZE:
                continue
            adjust_end_2 = stable_start - REACTION_LAG - SETTING_LAG
            adjust_start_2 = adjust_end_2 - LABEL_RANGE

            adjust_end_1 = adjust_end_2 - TIME_IN_ROLLER
            adjust_start_1 = adjust_start_2 - TIME_IN_ROLLER

            # store feature
            brand_train_data.append(
                calc_feature(
                    item_batch,
                    adjust_start_1,
                    adjust_start_2,
                    FEATURE_RANGE,
                    SPLIT_NUM
                )
            )

            # store label
            brand_train_label.append(
                calc_label(
                    item_batch,
                    adjust_start_1,
                    adjust_end_1,
                    adjust_start_2,
                    adjust_end_2
                )
            )

            # store mapping info
            brand_mapping.append([
                brand_index,
                batch_index,
                adjust_start_1,
                adjust_end_1,
                adjust_start_2,
                adjust_end_2,
                stable_start
            ])

            # store delta value
            brand_delta.append(
                calc_delta(
                    item_batch,
                    adjust_start_1,
                    adjust_end_1,
                    adjust_start_2,
                    adjust_end_2
                )
            )

    brand_train_data = np.array(brand_train_data)
    brand_train_label = np.array(brand_train_label)
    brand_mapping = np.array(brand_mapping)
    brand_delta = np.array(brand_delta)

    return brand_train_data, brand_train_label, brand_delta, brand_mapping,


def generate_all_training_data(data_per_brand: dict, criterion: dict, stage: str) -> Tuple[np.array, np.array, np.array, np.array]:
    """
    generate training data and label for all brand
    :param data_per_brand: all brand data
    :param criterion: criterion for each  brand
    :param stage: including 'head', 'transition', 'produce'
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
    elif stage == 'head':
        pass


def concatenate(data_: list) -> np.array:
    """
    concatenate list with item of different length
    """
    result = data_[0]
    for i in range(1, len(data_)):
        result = np.concatenate([result, data_[i]], axis=0)
    return result


def generate_test_data(data_: pd.DataFrame) -> np.array:
    min_len = TIME_IN_ROLLER + FEATURE_RANGE + LABEL_RANGE
    if len(data_) < min_len:
        return None
    data_ = data_[-min_len:]

    adjust_start_2 = min_len - LABEL_RANGE
    adjust_start_1 = adjust_start_2 - TIME_IN_ROLLER

    feature = calc_feature(data_, adjust_start_1, adjust_start_2, FEATURE_RANGE, SPLIT_NUM)
    return feature


def generate_validation_data(item: pd.DataFrame, stage: str) -> Tuple[np.array, np.array, np.array]:
    """
    TODO: add a demo to call this method
    generate real test data for one batch data at each time t
    :param item: one batch data
    :param stage: including 'head', 'transition', 'produce'
    :return:
        final_X_test: shape=(N, M), which N denotes the number of training data, M denotes the number of feature
        final_y_test: training label for evaluation, shape=(N, 2)
        final_mapping: mapping info
    """
    length = len(item)
    final_X_test = []
    final_y_test = []
    final_mapping = []

    if stage == 'transition':
        for item_index in range(TIME_IN_ROLLER + TRANSITION_FEATURE_RANGE, length, 1):
            adjust_start_2 = item_index - LABEL_RANGE
            adjust_start_1 = adjust_start_2 - TIME_IN_ROLLER

            final_X_test.append(
                calc_feature(
                    item,
                    adjust_start_1,
                    adjust_start_2,
                    TRANSITION_FEATURE_RANGE,
                    TRANSITION_SPLIT_NUM
                )
            )

            final_y_test.append(
                calc_current(
                    item,
                    adjust_start_1,
                    adjust_start_2,
                )
            )
            final_mapping.append([adjust_start_1, adjust_start_2])
    elif stage == 'produce':
        for item_index in range(TIME_IN_ROLLER + FEATURE_RANGE, length, 1):
            adjust_start_2 = item_index - LABEL_RANGE
            adjust_start_1 = adjust_start_2 - TIME_IN_ROLLER

            final_X_test.append(
                calc_feature(
                    item,
                    adjust_start_1,
                    adjust_start_2,
                    FEATURE_RANGE,
                    SPLIT_NUM
                )
            )

            final_y_test.append(
                calc_current(
                    item,
                    adjust_start_1,
                    adjust_start_2,
                )
            )
            final_mapping.append([adjust_start_1, adjust_start_2])
    elif stage == 'head':
        pass
    return np.array(final_X_test), np.array(final_y_test), np.array(final_mapping)
