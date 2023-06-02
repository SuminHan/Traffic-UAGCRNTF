python train.py --model_name=MyUAGCRN --dataset=metr-la --Q=12 --activity_embedding --sensor_embedding --graph_type=cooccur_dist
python train.py --model_name=MyUAGCRN --dataset=pems-bay --Q=12 --activity_embedding --sensor_embedding --graph_type=cooccur_dist
python train.py --model_name=MyUAGCRN --dataset=pemsd7 --Q=9 --activity_embedding --sensor_embedding --graph_type=cooccur_dist
python train.py --model_name=MyUAGCTransformer --dataset=metr-la --Q=12 --activity_embedding --sensor_embedding --graph_type=cooccur_dist
python train.py --model_name=MyUAGCTransformer --dataset=pems-bay --Q=12 --activity_embedding --sensor_embedding --graph_type=cooccur_dist
python train.py --model_name=MyUAGCTransformer --dataset=pemsd7 --Q=9 --activity_embedding --sensor_embedding --graph_type=cooccur_dist
