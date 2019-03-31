from collections import namedtuple
import calendar

import numpy as np
import pandas as pd
import torch
import torch.utils.data as dutils


Datasets = namedtuple('Datasets', ['train', 'validation', 'test'])


def prepare_data(df:pd.DataFrame, n_stores=None):
    df = df.rename(str.lower, axis='columns')
    if n_stores is not None:
        df = select_first_stores(df, n_stores)

    weekday_columns = ['day_of_week_{}'.format(day.lower())
                       for day in calendar.day_name]
    df, weekday_lookup = reverse_onehot(df, weekday_columns, 'week_day')

    monthday_columns = ['day_of_month_{:02d}'.format(day)
                        for day in np.arange(1, 32)]
    df, monthday_lookup = reverse_onehot(df, monthday_columns, 'month_day')

    shop_lookup = create_lookup(df['organization_unit_num'].unique())
    df.loc[:, 'organization_unit_num'] = [int(shop_lookup[i])
                                          for i in df.loc[:, 'organization_unit_num']]

    columns_to_drop = (
            list(weekday_lookup) +
            list(monthday_lookup) +
            [
             'holiday_valentines_day', 'holiday_labor_day',
             'holiday_may_day', 'holiday_spring_bank_holiday',
             'holiday_late_summer_bank_holiday', 'holiday_victoria_day',
             'holiday_victoria_day', 'holiday_civic_holiday',
             'holiday_ca_thanksgiving'
            ]+
            [f'time_of_year_{day}' for day in np.arange(0, 12)]
    )
    print('drop')
    df = df.drop(columns_to_drop, 1)

    print('sort')
    return df.sort_values(['organization_unit_num', 'sales_dt'])

def select_first_stores(df, n_stores):
    store_list = df['organization_unit_num'].unique()
    idx = df['organization_unit_num'].isin(store_list)
    df = df[idx]
    df.reset_index(drop=True, inplace=True)
    return df


def reverse_onehot(df, onehot_cols, new_col):
    lookup_dict = create_lookup(onehot_cols)
    df.loc[:, new_col] = [lookup_dict[i] for i in df[onehot_cols].idxmax(1)]
    return df, lookup_dict


def create_lookup(values):
    lookup_dict = {idx: i for i, idx in enumerate(values)}
    return lookup_dict


class StoreData(dutils.Dataset):
    def __init__(self, df):
        self.df = df

    def __getitem__(self, i):
        x = torch.FloatTensor(self.df.iloc[i, 2:].values.astype('float'))
        y = torch.FloatTensor([self.df.iloc[i, 0]])
        return x, y

    def __len__(self):
        return len(self.df)


def split_dataframe(df, fractions=(.8, .1, .1)):
    train = df.sample(frac=0.9, random_state=42)
    validation = train.sample(frac=0.1, random_state=42)
    test = df.drop(train.index)
    train = train.drop(validation.index)
    return Datasets(train, validation, test)
