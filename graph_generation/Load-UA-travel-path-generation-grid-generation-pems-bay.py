#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pickle 
import numpy as np
import os
import pandas as pd
import scipy.sparse as sp
from fastdtw import fastdtw
import csv
import tqdm
import matplotlib.pyplot as plt
from os.path import join as pjoin

import geopandas as gpd
from cartoframes.viz import *
from shapely.geometry import Point, LineString
import math
import folium
from shapely.geometry import LineString


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


# In[2]:


import datetime
EXECUTION_DATE = 'ua' #datetime.datetime.now().strftime('%Y%m%d')
GRAPH_TAG = 'GRID'
MILE_TO_METER = 1609.34
GRID_UNIT_METER = 2*MILE_TO_METER

# DATASET_NAME = 'metr-la'
DATASET_NAME = 'pems-bay'
# DATASET_NAME = 'pemsd7'

if not os.path.isdir(DATASET_NAME):
    os.mkdir(DATASET_NAME)


# In[3]:


if DATASET_NAME == 'metr-la':
    FILE_PATH = '../dataset/metr-la/'
    FILE_SENSOR_IDS = pjoin(FILE_PATH, 'graph_sensor_ids.txt')
    FILE_SENSOR_LOC = pjoin(f'../dataset/corrected-{DATASET_NAME}-sensorid-osm-path-uv.csv')
    FILE_ADJ_MX = pjoin(FILE_PATH, 'adj_mx.pkl')
    FILE_DATA = pjoin(FILE_PATH, 'metr-la.h5')
    
    sensor_df = pd.read_csv(FILE_SENSOR_LOC)
    _sensor_ids, _sensor_id_to_ind, adj_mx = load_graph_data(FILE_ADJ_MX)
    
    
elif DATASET_NAME == 'pems-bay':    
    FILE_PATH = '../dataset/pems-bay/'
    FILE_SENSOR_IDS = pjoin(FILE_PATH, 'graph_sensor_ids_bay.txt')
    FILE_SENSOR_LOC = pjoin(f'../dataset/corrected-{DATASET_NAME}-sensorid-osm-path-uv.csv')
    FILE_ADJ_MX = pjoin(FILE_PATH, 'adj_mx_bay.pkl')
    FILE_DATA = pjoin(FILE_PATH, 'pems-bay.h5')

    sensor_df = pd.read_csv(FILE_SENSOR_LOC)
    _sensor_ids, _sensor_id_to_ind, adj_mx = load_graph_data(FILE_ADJ_MX)
    

elif DATASET_NAME == 'pemsd7':
    FILE_PATH = '../dataset/pemsd7/'
    FILE_SENSOR_LOC = pjoin(f'../dataset/corrected-{DATASET_NAME}-sensorid-osm-path-uv.csv')
    FILE_ADJ_MX = pjoin(FILE_PATH, 'adj_mx.pkl')
    FILE_DATA = pjoin(FILE_PATH, 'pemsd7.h5')
    
    sensor_df = pd.read_csv(FILE_SENSOR_LOC)
    _sensor_ids, _sensor_id_to_ind, adj_mx = load_graph_data(FILE_ADJ_MX)
    


# In[4]:


data_df = pd.read_hdf(FILE_DATA)
sensor_ids = data_df.columns.values.astype(str)
assert np.sum(sensor_ids == _sensor_ids) == len(sensor_ids)
NUM_SENSORS = len(sensor_ids)
sensor_id_to_ind = _sensor_id_to_ind
ind_to_sensor_id = {v:k for k, v in sensor_id_to_ind.items()}

print('original distance_graph_loaded', np.count_nonzero(adj_mx))

with open(f'{DATASET_NAME}/original_adj_mx.pkl', 'wb') as f:
    pickle.dump([sensor_ids, sensor_id_to_ind, adj_mx], f, protocol=2)


# In[5]:


sensor_gdf = gpd.GeoDataFrame(
    sensor_df, geometry=gpd.points_from_xy(x=sensor_df.lng, y=sensor_df.lat)
)
sensor_gdf.crs = 'epsg:4326'
sensor_gdf_3310 = sensor_gdf.to_crs('epsg:3310')

Layer(sensor_gdf)
# # Prepare for next job

# In[6]:


id2geo = {str(sid):geo for sid, geo in zip(sensor_gdf['sid'], sensor_gdf.geometry)}


# In[7]:


from shapely.geometry import MultiPoint

multipoint = MultiPoint(sensor_gdf.geometry)
sensor_hull = multipoint.convex_hull

sensor_hull_gdf = gpd.GeoDataFrame(geometry=[sensor_hull])
sensor_hull_gdf.crs = 'epsg:4326'
sensor_hull_gdf_3310 = sensor_hull_gdf.to_crs('epsg:3310')
sensor_hull_3310 = sensor_hull_gdf_3310.iloc[0].geometry

x1, y1, x2, y2 = sensor_hull_gdf.total_bounds

sensor_center_latitude = (y1 + y2)/2 #sensor_hull.centroid.y
sensor_center_longitude = (x1 + x2)/2 #sensor_hull.centroid.x


# In[8]:


import osmnx as ox

if not os.path.isdir('osm_graph'):
    os.mkdir('osm_graph')
    
OSM_FILE_PATH = f'osm_graph/{DATASET_NAME}-drive.graphml'
    
graphs = dict()
# retrieve the street network for the location
if not os.path.isfile(OSM_FILE_PATH):
    center_point = gpd.GeoDataFrame(geometry = [Point(sensor_center_longitude, sensor_center_latitude)])
    center_point.crs = 'epsg:4326'
    center_point = center_point.to_crs('epsg:3310')
    max_distance = sensor_gdf.to_crs('epsg:3310').distance(center_point.iloc[0].geometry).max()+GRID_UNIT_METER*2
    print('max_distance:', max_distance)
    graph = ox.graph_from_point((sensor_center_latitude, sensor_center_longitude), dist=max_distance, network_type="drive")

    # save the street network to a shapefile
    ox.save_graphml(graph, filepath=OSM_FILE_PATH)
else:
    graph = ox.load_graphml(filepath=OSM_FILE_PATH)


# In[9]:


osm_nodes, osm_edges = ox.graph_to_gdfs(graph)
osm_nodes['osmidn'] = osm_nodes.index
osm_nodes['osmidstr'] = osm_nodes['osmidn'].astype(str)

osm_edges = osm_edges.reset_index()
cond = np.array([str(type(s)) for s in osm_edges['highway']]) == "<class 'str'>"
osm_edges = osm_edges[cond]


# In[10]:


osm_motorway = osm_edges[osm_edges['highway'].isin(['motorway',])]
osm_primary = osm_edges[osm_edges['highway'].isin(['motorway_link', 'primary', 'primary_link'])]
osm_secondary = osm_edges[osm_edges['highway'].isin(['secondary', 'secondary_link'])]
osm_others = osm_edges[~osm_edges['highway'].isin(['motorway','residential',
                                                  'motorway_link', 'primary', 'primary_link',
                                                  'secondary', 'secondary_link'])]
# Layer(osm_primary)


# # OSM path to sensor matching

# In[11]:


sensor_gdf.crs = 'epsg:4326'
osm_motorway.crs = 'epsg:4326'


# In[12]:


osm_motorway['u-v'] = osm_motorway['u'].astype(str) + '-' + osm_motorway['v'].astype(str)


# In[13]:


from shapely.ops import linemerge
from shapely.geometry import LineString
from shapely.geometry import Point, LineString
from shapely.ops import nearest_points


new_items = []
closest_line_list = []
    
for _, item in tqdm.tqdm(sensor_gdf.iterrows(), total=len(sensor_gdf)): 
#     print(item)
    closest_edge = osm_edges[(osm_edges['u'] == item.u) & (osm_edges['v'] == item.v)].iloc[0]
    closest_line = closest_edge.geometry
    closest_line_list.append(closest_line)
    #closest_point_on_line, closest_point_on_point = nearest_points(closest_line, item.geometry)
    closest_point_on_line = item.geometry
    nitem = dict(item)
    nitem['geometry'] = closest_point_on_line
    nitem['u'] = str(closest_edge['u'])
    nitem['v'] = str(closest_edge['v'])
    nitem['uv'] = str(closest_edge['u']) + '-' + str(closest_edge['v'])
    new_items.append(nitem)
new_sensor_gdf = gpd.GeoDataFrame(new_items)
new_sensor_gdf.crs='epsg:4326'


# In[14]:


new_sensor_gdf_3310 = new_sensor_gdf.to_crs('epsg:3310')


# In[43]:


osm_motorway[['u', 'v', 'u-v', 'geometry']].to_file('osm-pems-bay-motorway.geojson', driver='GeoJSON')


# In[15]:


Map([
#     Layer(osm_nodes,  basic_style(color='black'),
#          popup_click=[popup_element('osmidstr')]),
    Layer(osm_others,  basic_style(color='#bbbbbb'), encode_data=False),
    Layer(osm_secondary,  basic_style(color='#777777'), encode_data=False),
    Layer(osm_primary,  basic_style(color='black')),
    Layer(osm_motorway,  basic_style(color='blue')),
    Layer(gpd.GeoDataFrame(geometry=closest_line_list)),
    Layer(sensor_gdf, basic_style(color='pink'),
          popup_click=[popup_element('sid')], popup_hover=[popup_element('sid')]),
    Layer(new_sensor_gdf, basic_style(color='red'),
          popup_click=[popup_element('sid')], popup_hover=[popup_element('sid')]),
])


# In[16]:


discover_path_list = []
with open(f'{DATASET_NAME}/generated_paths_{EXECUTION_DATE}.txt') as fp:
    for line in fp:
        discover_path_list.append(line.strip().split())


# In[17]:


print('len(discover_path_list)', len(discover_path_list))


# In[18]:


osm_nodes.crs = 'epsg:4326'
osm_nodes_3310 = osm_nodes.to_crs('epsg:3310')
osm_edges.crs = 'epsg:4326'
osm_edges_3310 = osm_edges.to_crs('epsg:3310')


# In[19]:


fosm_edges_3310 = osm_edges_3310
fosm_edges_3310 = fosm_edges_3310[fosm_edges_3310['highway'] != 'residential'].copy()


# In[20]:


fmotorway = osm_edges_3310[osm_edges_3310['highway'] == 'motorway'].copy()
fmotorway['u'] = fmotorway['u'].astype(str)
fmotorway['v'] = fmotorway['v'].astype(str)
fmotorway['u-v'] = fmotorway['u'].astype(str) + '-' + fmotorway['v'].astype(str)


motorway_list = {u + '-' + v:-1 for u, v in zip(fmotorway['u'].astype(str).tolist(), fmotorway['v'].astype(str).tolist())}

highway_path_list = []
for path in tqdm.tqdm(discover_path_list):
    used_highway = False
    for i, node_u in enumerate(path[:-1]):
        node_v = path[i+1]
        if node_u + '-' + node_v in motorway_list:
            used_highway = True
            break
    if used_highway:
        highway_path_list.append(path)


# In[21]:


myedges = fosm_edges_3310.copy()
myedges['u'] = myedges['u'].astype(str)
myedges['v'] = myedges['v'].astype(str)
myedges['u-v'] = myedges['u'] + '-' + myedges['v']
uv2edgeid = {(str(u), str(v)):eid for u, v, eid in zip(myedges['u'], myedges['v'], myedges['u-v'])}


# In[22]:


mitem_list = dict()
for _, item in new_sensor_gdf_3310.iterrows():
    mitem = dict(myedges[(myedges['u'] == item['u']) & (myedges['v'] == item['v'])].iloc[0])
    
    found_edge = mitem['u-v']
    mitem_list.setdefault(found_edge, mitem)
    mitem_list[found_edge].setdefault('sid2dist', dict())
    
    node_u = osm_nodes_3310[osm_nodes_3310['osmidstr'] == str(mitem['u'])].iloc[0]
    sid = item['sid']
    dist = node_u.geometry.distance(item.geometry)
    mitem_list[found_edge]['sid2dist'][sid] = dist
    
    
synch_gdf = gpd.GeoDataFrame([mitem_list[k] for k in mitem_list])

sensor_words = []
total = 0
for _, item in synch_gdf.iterrows():
    
    w_list = []
    d = item['sid2dist']
    for w in sorted(d, key=d.get, reverse=False):
        w_list.append('S' + str(w))
        total += 1
    #print(w_list)
    sensor_words.append(' '.join(w_list))
synch_gdf['sensors'] = sensor_words

eid2sw  = {eid:sw for eid, sw in zip(synch_gdf['u-v'], synch_gdf['sensors'])}


# In[ ]:





# In[23]:


all_path_sentences = []
for path in tqdm.tqdm(highway_path_list):
    path_sentence = str(path[0])
    for i in range(len(path)-1):
        u, v = path[i], path[i+1]
        eid = uv2edgeid[u, v]
        #path_count.setdefault(eid, 0)
        #path_count[eid] += 1
        if eid in eid2sw:
            path_sentence += ' ' + eid2sw[eid]
        path_sentence += ' ' + str(v)
    all_path_sentences.append(path_sentence)


# # N2V similiarity

# In[24]:


from gensim.models import Word2Vec

vector_size = 64
# Define your list of sentences
sentences = [sent.split() for sent in all_path_sentences]

# Generate the Word2Vec model
model = Word2Vec(sentences, window=7, min_count=1, workers=4, vector_size=vector_size)

# Print the vector for the word 'sentence'
# print(model['sentence'])

import numpy as np
wv_array = []
for sid in data_df.columns:
    q = f'S{sid}'
    if q in model.wv:
        wv_array.append(model.wv[q])
    else:
        wv_array.append(np.zeros(vector_size))
        
wv_array = np.array(wv_array)
def cosine_similarity(vector1, vector2):
    dot_product = np.dot(vector1, vector2)
    if dot_product == 0:
        return -1
    magnitude1 = np.linalg.norm(vector1)
    magnitude2 = np.linalg.norm(vector2)
    cosine_similarity = dot_product / (magnitude1 * magnitude2)
    return cosine_similarity

sim_array = np.eye(NUM_SENSORS)
for i in range(wv_array.shape[0]):
    for j in range(i+1, wv_array.shape[0]):
        sim_array[j, i] = sim_array[i, j] = cosine_similarity(wv_array[i], wv_array[j])

with open(f'{DATASET_NAME}/n2v_sim_{EXECUTION_DATE}.pkl', 'wb') as f:
    pickle.dump([sensor_ids, sensor_id_to_ind, sim_array], f, protocol=2)


# In[25]:


with open(f'{DATASET_NAME}/n2v_SE_{EXECUTION_DATE}.txt', 'w') as fp:
    fp.write(f'{data_df.shape[1]} {vector_size}\n')
    for sid in data_df.columns.astype(str):
        q = f'S{sid}'
        if q in model.wv:
            q_w2v = model.wv[q]
        else:
            q_w2v = np.zeros(vector_size)

#         print(sensor_id_to_ind[sid], model.wv[q])
        fp.write(str(sensor_id_to_ind[sid]) + ' ' + ' '.join([str(v) for v in model.wv[q]]) + '\n')


# In[26]:


plt.matshow(sim_array)


# In[27]:


print('node2vec distance_graph_loaded', np.count_nonzero(sim_array))


# # Co-occurence Matrix

# In[28]:


sentences = all_path_sentences #tmp_path_sentences

co_occurrence_vectors = pd.DataFrame(
    np.zeros([len(sensor_ids), len(sensor_ids)]),
    index = ['S'+s for s in sensor_ids],
    columns = ['S'+s for s in sensor_ids]
)

word_count = dict()
word_co_occur = dict()
for sent in tqdm.tqdm(sentences):
    ext_sent = [w for w in sent.split() if w[0] == 'S']
    for i, w in enumerate(ext_sent):
        word_count.setdefault(w, 0)
        co_occurrence_vectors.loc[w, w] +=1
        
        for w2 in ext_sent[i+1:]:
            if w != w2:
                co_occurrence_vectors.loc[w, w2] += 1
                co_occurrence_vectors.loc[w2, w] += 1


# In[29]:


sim_array2 = np.eye(NUM_SENSORS)

for i in range(NUM_SENSORS):
    for j in range(i, NUM_SENSORS):
        w = 'S'+ind_to_sensor_id[i]
        w2 = 'S'+ind_to_sensor_id[j]
        wc1 = co_occurrence_vectors.loc[w, w]
        wc2 = co_occurrence_vectors.loc[w2, w2]
        sim_array2[j, i] = sim_array2[i, j] = co_occurrence_vectors.loc[w2, w]/((wc1*wc2)**.5+1)


# In[30]:


plt.matshow(sim_array2)


# In[31]:


with open(f'{DATASET_NAME}/cooccur_sim_{EXECUTION_DATE}.pkl', 'wb') as f:
    pickle.dump([sensor_ids, sensor_id_to_ind, sim_array2], f, protocol=2)


# In[32]:


print('cooccur distance_graph_loaded', np.count_nonzero(sim_array2))


# # Reachable Distance Matrix

# In[33]:


sid2eid = dict()
for eid, sw in eid2sw.items():
    sids = sw.split()
    for sid in sids:
        sid2eid[sid] = eid


# In[34]:


def track_path(path_sentence, sid1, sid2):
    track_switch = False
    track_paths = []
    for node in path_sentence.split():

        if not track_switch and node == sid1:
            track_switch = True
            track_paths.append(sid1)

        if track_switch and node[0] != 'S':
            track_paths.append(node)

        if node == sid2:
            track_paths.append(sid2)
            break
    return track_paths


# In[35]:


def sid_dist(sid1, sid2, track_paths):
    between_sid_dist = 0
    sgeo1 = new_sensor_gdf_3310[new_sensor_gdf_3310['sid'] == int(sid1[1:])].iloc[0].geometry
    sgeo2 = new_sensor_gdf_3310[new_sensor_gdf_3310['sid'] == int(sid2[1:])].iloc[0].geometry
    if len(track_paths) == 2:
        between_sid_dist = sgeo1.distance(sgeo2)
    else:
        edge1 = myedges[myedges['u-v'] == sid2eid[sid1]].iloc[0]
        edge2 = myedges[myedges['u-v'] == sid2eid[sid2]].iloc[0]

        edge1_v = osm_nodes_3310[osm_nodes_3310['osmidstr'] == str(edge1.v)].iloc[0].geometry
        edge2_u = osm_nodes_3310[osm_nodes_3310['osmidstr'] == str(edge2.u)].iloc[0].geometry

        between_sid_dist += sgeo1.distance(edge1_v)
        between_sid_dist += edge2_u.distance(sgeo2)

        rest_paths = track_paths[1:-1]
        for i, node_u in enumerate(rest_paths[:-1]):
            node_v = rest_paths[i+1]
            between_sid_dist += myedges[(myedges['u'] == node_u) & (myedges['v'] == node_v)].iloc[0].geometry.length
            
    return between_sid_dist


# In[36]:


sid_dist_dict = dict()

for path in tqdm.tqdm(discover_path_list):
    path_sentence = str(path[0])
    for i in range(len(path)-1):
        u, v = path[i], path[i+1]
        eid = uv2edgeid[u, v]
        if eid in eid2sw:
            path_sentence += ' ' + eid2sw[eid]
        path_sentence += ' ' + str(v)
    
    co_sensors = [node for node in path_sentence.split() if node[0] == 'S']
    for i, sid1 in enumerate(co_sensors[:-1]):
        sid2 = co_sensors[i+1]
        
        if sid1 in sid_dist_dict and sid2 in sid_dist_dict[sid1]:
            continue
        
        track_paths = track_path(path_sentence, sid1, sid2)
        between_sid_dist = sid_dist(sid1, sid2, track_paths)
        sid_dist_dict.setdefault(sid1, dict())
        sid_dist_dict[sid1][sid2] = between_sid_dist
        
    
    for i, sid in enumerate(co_sensors[:-1]):
        cum_dist = 0
        psid = sid
        for qsid in co_sensors[i+1:]:
            cum_dist += sid_dist_dict[psid][qsid]
            psid = qsid
            
            if sid in sid_dist_dict and qsid in sid_dist_dict[sid]:
                continue
            else:
                sid_dist_dict[sid][qsid] = cum_dist
            


# In[37]:


dist_mat = np.zeros((len(sensor_ids), len(sensor_ids)))
dist_mat.fill(np.inf)
np.fill_diagonal(dist_mat, 0)
for k1 in sid_dist_dict:
    for k2 in sid_dist_dict[k1]:
#         print(k1, k2, sid_dist_dict[k1][k2])
        
        ii = sensor_id_to_ind[k1[1:]]
        jj = sensor_id_to_ind[k2[1:]]
        val = sid_dist_dict[k1][k2]
        
        dist_mat[jj, ii] = val


# In[38]:


plt.matshow(dist_mat)


# In[39]:


with open(f'{DATASET_NAME}/dist_meters_{EXECUTION_DATE}.pkl', 'wb') as f:
    pickle.dump([sensor_ids, sensor_id_to_ind, dist_mat], f, protocol=2)

dist_mat[dist_mat > MILE_TO_METER*80] = np.inf
dist_vals_meters = dist_mat[~np.isinf(dist_mat)].flatten()
print(dist_vals_meters.std())
dist_sigma = 5*MILE_TO_METER
# dist_std = distances.std()
# print(dist_std)
new_adj_mx = np.exp(-np.square(dist_mat / dist_sigma))
# new_adj_mx[new_adj_mx < .1] = 0
print('distance_graph_loaded', np.count_nonzero(new_adj_mx))

with open(f'{DATASET_NAME}/new_dist_sim_{EXECUTION_DATE}.pkl', 'wb') as f:
    pickle.dump([sensor_ids, sensor_id_to_ind, new_adj_mx], f, protocol=2)
    
final_adj_mx = new_adj_mx*sim_array2
with open(f'{DATASET_NAME}/urban_activity_sim_{EXECUTION_DATE}.pkl', 'wb') as f:
    pickle.dump([sensor_ids, sensor_id_to_ind, final_adj_mx], f, protocol=2)


# In[40]:


print('Original\t distance_graph_loaded', np.count_nonzero(adj_mx))
print('Node2vec\t distance_graph_loaded', np.count_nonzero(sim_array))
print('Cooccur \t distance_graph_loaded', np.count_nonzero(sim_array2))
print('DistMat \t distance_graph_loaded', np.count_nonzero(dist_mat))
print('New Dist\t distance_graph_loaded', np.count_nonzero(new_adj_mx))
print('Final   \t distance_graph_loaded', np.count_nonzero(final_adj_mx))


# In[41]:


plt.matshow(adj_mx)
plt.matshow(sim_array2)
plt.matshow(new_adj_mx)
plt.matshow(final_adj_mx)


# In[ ]:




