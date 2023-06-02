with open('run_subprocess_list.sh', 'w') as fp:
    for model in ['MyUAGCRN', 'MyUAGCTransformer']:
        for activity_postfix in ['--activity_embedding']:
            for sensor_postfix in ['--sensor_embedding']:
                for graph_type in ['cooccur_dist']:
                    for dataset, Q in [('metr-la', 12), ('pems-bay', 12), ('pemsd7', 9)]:
                        command = f'python train.py --model_name={model} --dataset={dataset} --Q={Q} {activity_postfix} {sensor_postfix} --graph_type={graph_type}'
                        print(command)
                        fp.write(command + '\n')

                    
    # for model in ['MyARLSTM', 'MyTransformer']:
    #     for activity_postfix in ['--activity_embedding', '']:
    #         for sensor_postfix in ['--sensor_embedding', '']:
    #             for dataset, Q in [('metr-la', 12), ('pems-bay', 12), ('pemsd7', 9)]:
    #                 command = f'python train.py --model_name={model} --dataset={dataset} --Q={Q} {activity_postfix} {sensor_postfix}'
    #                 print(command)
    #                 fp.write(command + '\n')
                    
    # for model in ['MyUADCGRU']:
    #     for activity_postfix in ['--activity_embedding']:
    #         for sensor_postfix in ['--sensor_embedding']:
    #             for graph_type in ['cooccur_dist']:
    #                 for dataset, Q in [('metr-la', 12), ('pems-bay', 12), ('pemsd7', 9)]:
    #                     command = f'python train.py --model_name={model} --dataset={dataset} --Q={Q} {activity_postfix} {sensor_postfix} --graph_type={graph_type}'
    #                     print(command)
    #                     fp.write(command + '\n')

                    
