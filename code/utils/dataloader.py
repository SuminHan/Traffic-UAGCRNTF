import os, tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import utils.custom_scaler as custom_scaler

import datetime
from datetime import datetime, timedelta 

import pickle 
def load_graph_data(pkl_filename):
    sensor_ids, sensor_id_to_ind, adj_mx = load_pickle(pkl_filename)
    return sensor_ids, sensor_id_to_ind, adj_mx

def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data


def seq2instance(data, P, Q):
    num_step = data.shape[0]
    data_type = data.dtype
    num_sample = (num_step - P - Q + 1)
    x = np.zeros(shape = (num_sample, P, *data.shape[1:]))
    y = np.zeros(shape = (num_sample, Q, *data.shape[1:]))
    for i in range(0, num_sample):
        x[i] = data[i : i + P].astype(data_type)
        y[i] = data[i + P : i + P + Q].astype(data_type)
    return x, y
    

def load_data(args):
    metadata = dict(args=args)

    df = pd.read_hdf(f'datasets/{args.dataset}.h5')
    metadata['num_sensors'] = df.shape[1]
    timestamp = pd.DatetimeIndex(df.index); metadata['timestamp'] = timestamp
    dataset = set_all_dataset(df.values, timestamp, args, metadata)
    return dataset, metadata


def train_scaler(train_data, args):
    # it will be used to normalize (batch, num_seq, N, 1) data or denormalize from model output
    scaler = custom_scaler.ZScoreNormalization()
    scaler.fit(train_data)
    
    return scaler


def set_all_dataset_norm(data, timestamps, args, metadata):
    P, Q = args.P, args.Q
    train_ratio, val_ratio, test_ratio = args.train_ratio, args.val_ratio, args.test_ratio
    assert train_ratio + val_ratio + test_ratio == 1
    
    weekdays = timestamps.weekday
    timeofday = (timestamps.values - timestamps.values.astype("datetime64[D]")) / np.timedelta64(5, "m")

    TE = np.stack([weekdays, timeofday], -1).astype(np.int32)
    

    if args.activity_embedding:
        from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
        dr = pd.date_range(start=timestamps.strftime('%Y-%m-%d').min(), end=timestamps.strftime('%Y-%m-%d').max())
        cal = calendar()
        holidays = cal.holidays(start=dr.min(), end=dr.max())
        holidays = [h.strftime('%Y-%m-%d') for h in holidays]
        holidays = timestamps.strftime('%Y-%m-%d').isin(holidays).astype(int)

        travel_purpose = np.load('../activity_embedding/why_trip_profiling_all_gaussian.npy') 
        travel_purpose = travel_purpose / travel_purpose.std()

        assert travel_purpose.shape[0] == 12*24*7

        new_TE = []
        for i in range(TE.shape[0]):
            if holidays[i] == 0:
                timeidx = int(TE[i, 0]*12*24 + int(TE[i, 1]))
                new_TE.append(travel_purpose[timeidx])
            else:
                timeidx1 = int(5*12*24 + int(TE[i, 1]))
                timeidx2 = int(6*12*24 + int(TE[i, 1]))
                new_TE.append((travel_purpose[timeidx1] + travel_purpose[timeidx2])/2)

        TE = np.stack(new_TE, 0)
        metadata['TE_channel'] = TE.shape[1]

        
    if args.sensor_node2vec:
        # spatial embedding 
        f = open(f'../graph_generation/{args.dataset}/n2v_SE_ua.txt', mode = 'r')
        lines = f.readlines()
        temp = lines[0].split()
        N, dims = int(temp[0]), int(temp[1])
        SE = np.zeros(shape = (N, dims), dtype = np.float32)
        for line in lines[1 :]:
            temp = line.split()
            index = int(temp[0])
            SE[index] = temp[1 :]
        metadata['SEN2V'] = SE

    datasize = len(data)
    data = np.expand_dims(data, -1).astype(np.float32)

    x, y = seq2instance(data, P, Q)
    te = seq2instance(TE, P, Q)
    te = np.concatenate(te, axis = 1)

    num_samples = x.shape[0]
    num_test = round(num_samples * 0.2)
    num_train = round(num_samples * 0.7)
    num_val = num_samples - num_test - num_train
    
    trainX, trainY = x[:num_train], y[:num_train]
    # val
    valX, valY = (
        x[num_train: num_train + num_val],
        y[num_train: num_train + num_val],
    )
    # test
    testX, testY = x[-num_test:], y[-num_test:]

    metadata['scaler'] = train_scaler(trainX, args)
    trainX = metadata['scaler'].transform(trainX)
    valX = metadata['scaler'].transform(valX)
    testX = metadata['scaler'].transform(testX)
    
    trainTE = te[:num_train]
    valTE = te[num_train: num_train + num_val]
    testTE = te[-num_test:]
    
    np.save(f'prediction/{args.dataset}/ground_truth.npy', testY)
    np.save(f'prediction/{args.dataset}/testTE.npy', testTE)
    
    return trainX, trainTE, trainY, valX, valTE, valY, testX, testTE, testY

def load_data_norm(args):
    metadata = dict(args=args)
    df = pd.read_hdf(f'../dataset/{args.dataset}/{args.dataset}.h5')
    metadata['num_sensors'] = df.shape[1]

    
    graph_file_path = None
    if args.graph_type == 'legacy':
        graph_file_path = f'../graph_generation/{args.dataset}/original_adj_mx.pkl'
    elif args.graph_type == 'cooccur_dist':
        graph_file_path = f'../graph_generation/{args.dataset}/urban_activity_sim_ua.pkl'
    elif args.graph_type == 'n2v_sim':
        graph_file_path = f'../graph_generation/{args.dataset}/n2v_sim_ua.pkl'
    elif args.graph_type == 'new_dist_sim':
        graph_file_path = f'../graph_generation/{args.dataset}/new_dist_sim.pkl'
    
    if args.graph_type != 'none':
        sensor_ids, sensor_id_to_ind, adj_mx = load_graph_data(graph_file_path)
        args.graph_file = graph_file_path
        metadata['adj_mx'] = adj_mx
        print('distance_graph_loaded', np.count_nonzero(adj_mx))

    timestamp = pd.DatetimeIndex(df.index); metadata['timestamp'] = timestamp
    dataset = set_all_dataset_norm(df.values, timestamp, args, metadata)

    return dataset, metadata



