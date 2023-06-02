import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
import argparse
import time, tqdm
import mymodels, utils
from utils import *
import warnings
import keras
from keras.callbacks import TensorBoard

import logging

slack_message = ""

from datetime import datetime 

start_time = datetime.now() 



def model_define(args, metadata):
    P, Q = args.P, args.Q
    num_sensors = metadata['num_sensors']

    X  = layers.Input(shape=(P, num_sensors, metadata['scaler'].output_channel), dtype=tf.float32)

    if args.activity_embedding:
        TE  = layers.Input(shape=(P+Q, metadata['TE_channel']), dtype=tf.float32)
    else:
        TE  = layers.Input(shape=(P+Q, 2), dtype=tf.float32)
    
    tmp_model = mymodels.str_to_class(args.model_name)(args, metadata)
    Y = tmp_model(X, TE)
    Y =  metadata['scaler'].inverse_transform(Y)

    
    model = keras.models.Model((X, TE), Y)
    model_name = tmp_model.model_name


    return model, model_name


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='parameter')
    parser.add_argument('--dataset', type=str, choices=['metr-la', 'pems-bay', 'pemsd7'], default=f'pemsd7')
    parser.add_argument('--model_name', type=str, default=f'MyUAGCRN')
    parser.add_argument('--graph_type', type=str, choices=['legacy', 'cooccur_dist', 'coocur', 'n2vsim', 'new_dist_sim', 'none'], default='none')
    parser.add_argument('--restore_model', action='store_true')
    parser.add_argument('--train_again', action='store_true')
    parser.add_argument('--activity_embedding', action='store_true')
    parser.add_argument('--timestamp_embedding', action='store_true')
    parser.add_argument('--sensor_embedding', action='store_true')
    parser.add_argument('--sensor_node2vec', action='store_true')
    parser.add_argument('--P', type=int, default=12) # history sequence
    parser.add_argument('--Q', type=int, default=12) # prediction sequence
    parser.add_argument('--D', type=int, default=64) # hidden dimension
    parser.add_argument('--K', type=int, default=8) # TF: number of attention heads if used
    parser.add_argument('--d', type=int, default=8) # TF: number of attention hidden dimension (D=K*d)
    parser.add_argument('--L', type=int, default=3) # TF: stack of layers
    parser.add_argument('--K_diffusion', type=int, default=3) # DCGRU: number of diffusion step if use diffusion
    parser.add_argument('--filter_type', type=str, default='dual_random_walk') # DCGRU: number of diffusion step if use diffusion
    
    parser.add_argument('--train_ratio', type=float, default=0.7)
    parser.add_argument('--val_ratio', type=float, default=0.1)
    parser.add_argument('--test_ratio', type=float, default=0.2)

    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max_epoch', type=int, default=100)
    parser.add_argument('--optimizer', type=str, default=f'adam')
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--patience_stop', type=int, default=5)
    parser.add_argument('--patience_lr', type=int, default=2)
    
    args = parser.parse_args()
    if not os.path.isdir('prediction'):
        os.mkdir('prediction')
    if not os.path.isdir(f'prediction/{args.dataset}'):
        os.mkdir(f'prediction/{args.dataset}')

    if args.dataset == 'pemsd7':
        assert args.Q == 9
    else:
        assert args.Q == 12

    dataset, metadata = utils.load_data_norm(args)
    trainX, trainTE, trainY, valX, valTE, valY, testX, testTE, testY = dataset
    print('trainX:', trainX.shape,  'trainTE:', trainTE.shape,  'trainY:', trainY.shape)
    print('valX:', valX.shape,      'valTE:', valTE.shape,      'valY:', valY.shape)
    print('testX:', testX.shape,    'testTE:', testTE.shape,    'testY:', testY.shape)

    print(args)
    slack_message += str(args) + '\n'

    # define the model  
    ae_name = ''
    assert not (args.activity_embedding and args.timestamp_embedding)
    if args.activity_embedding:
        ae_name = 'AE'
    elif args.timestamp_embedding:
        ae_name = 'TE'
    else:
        ae_name = 'none'
    
    se_name = ''
    assert not (args.sensor_embedding and args.sensor_node2vec)
    if args.sensor_embedding:
        se_name = 'SE'
    elif args.sensor_node2vec:
        se_name = 'N2V'
    else:
        se_name = 'none'

    model, model_name = model_define(args, metadata)
    model_logging_name = f'{args.dataset}_{model_name}_G-{args.graph_type}_AE-{ae_name}_SE-{se_name}'
    model_checkpoint = f'./model_checkpoint/{args.dataset}/{model_logging_name}'
    model_logs = f'./model_logs/{args.dataset}/{model_logging_name}'
    
    
    if not args.train_again and os.path.isfile(f'prediction/{args.dataset}/{model_logging_name}.npy'):
        import sys
        sys.exit()


    # Set up logging configuration
    if not os.path.isdir('./test_logs'):
        os.mkdir('./test_logs')
    if not os.path.isdir(f'./test_logs/{args.dataset}'):
        os.mkdir(f'./test_logs/{args.dataset}')
    
    if args.restore_model:
        logging.basicConfig(filename=f'./test_logs/{args.dataset}/{model_logging_name}.log', level=logging.INFO, format='%(asctime)s - %(message)s', filemode='a')
    else:
        logging.basicConfig(filename=f'./test_logs/{args.dataset}/{model_logging_name}.log', level=logging.INFO, format='%(asctime)s - %(message)s', filemode='w')

    logging.info(str(args))

    # Custom callback for logging metrics during training and testing
    class LoggingCallback(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            if logs is not None:
                # logging.info(f"Epoch {epoch + 1}: Training Loss = {logs['loss']}")
                # logging.info(str(logs))
                logging.info(f"Epoch {epoch + 1}: "+ str(logs))
            
            if (epoch+1) % 10 == 0:
                predY = self.model.predict((testX, testTE), batch_size=args.batch_size)

                for q in range(args.Q):
                    logging.info(f'Horizon {q+1:02d} (MAE/RMSE/MAPE):\t' + '\t'.join(f'{s:.4f}' for s in metric(testY[:, q, :, :], predY[:, q, :, :])))


    # if args.restore_model:
    try:
        if not args.train_again:
            model.load_weights(model_checkpoint)
            print('model restore successful')
    except:
        print('there exists no pretrained model')
        # import sys
        # sys.exit(0)
    
    if args.optimizer == 'sgd':
        optimizer = tf.keras.optimizers.SGD(learning_rate = args.learning_rate)
    elif args.optimizer == 'adam':
        optimizer = keras.optimizers.Adam(args.learning_rate)
    elif args.optimizer == 'adagrad':
        optimizer = keras.optimizers.Adagrad(args.learning_rate)

    model.compile(loss=custom_mae_loss, optimizer=optimizer)
    model.summary()

    # Define some callbacks to improve training.
    early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss", patience=args.patience_stop, min_delta=1e-6)
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=args.patience_lr)
    model_ckpt = tf.keras.callbacks.ModelCheckpoint(model_checkpoint, save_weights_only=True, \
                    save_best_only=True, monitor='val_loss', mode='min', verbose=0)
    time_callback = utils.TimeHistory()
    # tb_callback = TensorBoard(log_dir=model_logs, histogram_freq=1, write_graph=True, write_images=True)
    logging_callback = LoggingCallback()


    # # Suppress the error message
    # warnings.filterwarnings("ignore", message="MutableGraphView::SortTopologically error: detected edge(s) creating cycle(s)")

    # # Generate the model visualization (which may still contain the cycle)

    # # Reset the warning filters (so other warning messages are still displayed)
    # warnings.resetwarnings()
    
    model.fit((trainX, trainTE), trainY,
                batch_size=args.batch_size,
                epochs=args.max_epoch,
                verbose=1,
                validation_data=((valX, valTE), valY),
                callbacks=[early_stopping, model_ckpt, reduce_lr, logging_callback],
    )

    model, model_name = model_define(args, metadata)
    model.load_weights(model_checkpoint)

    predY = model.predict((testX, testTE), batch_size=args.batch_size)
    # labelY = testY
    

    labelY = np.load(f'prediction/{args.dataset}/ground_truth.npy')

    
    np.save(f'prediction/{args.dataset}/{model_logging_name}.npy', predY)

    # print(f'{args.dataset}_{model_name}_{args.graph_type}_{args.P}-{args.Q} test result 15min:', '\t'.join(f'{s:.5f}' for s in metric(labelY[:, :3, :, :], predY[:, :3, :, :])))
    # print(f'{args.dataset}_{model_name}_{args.graph_type}_{args.P}-{args.Q} test result 30min:', '\t'.join(f'{s:.5f}' for s in metric(labelY[:, :6, :, :], predY[:, :6, :, :])))
    # #print(f'{model_name}_{args.graph_type}_{args.P}-{args.Q} test result 45min:', '\t'.join(f'{s:.5f}' for s in metric(labelY[:, :9, :, :], predY[:, :9, :, :])))
    # print(f'{args.dataset}_{model_name}_{args.graph_type}_{args.P}-{args.Q} test result:', '\t'.join(f'{s:.5f}' for s in metric(labelY, predY)))


    slack_message += str(f'{model_logging_name} result') + '\n'

    logging.info(f'Final test:')
    for q in range(args.Q):
        slack_message += f'Horizon {q+1}:\t'          + '\t'.join(f'{s:.5f}' for s in metric(labelY[:, q, :, :], predY[:, q, :, :])) + '\n'
        print(f'Horizon {q+1}:\t'          + '\t'.join(f'{s:.5f}' for s in metric(labelY[:, q, :, :], predY[:, q, :, :])))
        logging.info(f'Horizon {q+1:02d} (MAE/RMSE/MAPE):\t' + '\t'.join(f'{s:.4f}' for s in metric(testY[:, q, :, :], predY[:, q, :, :])))

    time_elapsed = datetime.now() - start_time 
    slack_message += f'Time elapsed (hh:mm:ss ms) {time_elapsed}\n'
    slack_message += f'------------------------------------------\n'

    print(slack_message)
