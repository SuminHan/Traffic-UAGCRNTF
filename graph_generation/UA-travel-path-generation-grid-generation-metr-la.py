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

DATASET_NAME = 'metr-la'
# DATASET_NAME = 'pems-bay'
# DATASET_NAME = 'pemsd7'

if not os.path.isdir(DATASET_NAME):
    os.mkdir(DATASET_NAME)


# In[3]:


if DATASET_NAME == 'metr-la':
    FILE_PATH = '../dataset/metr-la/'
    FILE_SENSOR_IDS = pjoin(FILE_PATH, 'graph_sensor_ids.txt')
    FILE_SENSOR_LOC = pjoin(FILE_PATH, 'graph_sensor_locations_corrected.csv')
    FILE_ADJ_MX = pjoin(FILE_PATH, 'adj_mx.pkl')
    FILE_DATA = pjoin(FILE_PATH, 'metr-la.h5')
    
    sensor_df = pd.read_csv(FILE_SENSOR_LOC, index_col=0)
    sensor_df.columns = ['sid', 'lat', 'lng']
    _sensor_ids, _sensor_id_to_ind, adj_mx = load_graph_data(FILE_ADJ_MX)
    
    
elif DATASET_NAME == 'pems-bay':    
    FILE_PATH = '../dataset/pems-bay/'
    FILE_SENSOR_IDS = pjoin(FILE_PATH, 'graph_sensor_ids_bay.txt')
    FILE_SENSOR_LOC = pjoin(FILE_PATH, 'graph_sensor_locations_bay.csv')
    FILE_ADJ_MX = pjoin(FILE_PATH, 'adj_mx_bay.pkl')
    FILE_DATA = pjoin(FILE_PATH, 'pems-bay.h5')

    sensor_df = pd.read_csv(FILE_SENSOR_LOC, names=['sid', 'lat', 'lng'])
    _sensor_ids, _sensor_id_to_ind, adj_mx = load_graph_data(FILE_ADJ_MX)
    

elif DATASET_NAME == 'pemsd7':
    FILE_PATH = '../dataset/pemsd7/'
    FILE_SENSOR_LOC = pjoin(FILE_PATH, 'PeMSD7_M_Station_Info.csv')
    sensor_df = pd.read_csv(FILE_SENSOR_LOC, index_col=0)
    sensor_df.columns = ['sid', 'fwy', 'dir', 'district', 'lat', 'lng']
    
    if True or (not os.path.isfile(pjoin(FILE_PATH, 'adj_mx.pkl')) or not os.path.isfile(pjoin(FILE_PATH, 'pemsd7.h5'))):
        FILE_DIST_CSV = pjoin(FILE_PATH, 'PeMSD7_W_228.csv')
        FILE_DATA_CSV = pjoin(FILE_PATH, 'PeMSD7_V_228.csv')

        _sensor_ids = sensor_df['sid'].astype(str).tolist()
        _sensor_id_to_ind = {k:i for i, k in enumerate(_sensor_ids)}
        _dist_df = pd.read_csv(FILE_DIST_CSV, header=None)
        _dist_mx = _dist_df.values /1609.34
        _sigma = 10**.5
        _adj_mx =  np.exp(- (_dist_mx / _sigma)**2)
        _adj_mx[_adj_mx < .1] = 0

        adj_mx = _adj_mx
        with open(pjoin(FILE_PATH, 'adj_mx.pkl'), 'wb') as f:
            pickle.dump([_sensor_ids, _sensor_id_to_ind, adj_mx], f, protocol=2)
        
        _data_df = pd.read_csv(FILE_DATA_CSV, header=None)
        _data_df.columns = _sensor_ids
        start_time = datetime.datetime(2012, 5, 1, 0, 0, 0)
        end_time = datetime.datetime(2012, 7, 1, 0, 0, 0)

        five_mins = datetime.timedelta(minutes=5)
        timeslot = []
        curr_time = start_time
        while curr_time < end_time:
            if curr_time.weekday() < 5:
                timeslot.append(curr_time)
            curr_time = curr_time + five_mins

        _data_df.index = timeslot
        _data_df.to_hdf('pemsd7/pemsd7.h5', key='df')
    
    FILE_ADJ_MX = pjoin(FILE_PATH, 'adj_mx.pkl')
    FILE_DATA = pjoin(FILE_PATH, 'pemsd7.h5')
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


# In[54]:


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


from shapely.ops import linemerge
from shapely.geometry import LineString
from shapely.geometry import Point, LineString
from shapely.ops import nearest_points


new_items = []
closest_line_list = []
    
for _, item in tqdm.tqdm(sensor_gdf.iterrows(), total=len(sensor_gdf)): 
    closest_edge = osm_motorway.iloc[osm_motorway.distance(item.geometry).argmin()]
    closest_line = closest_edge.geometry
    closest_line_list.append(closest_line)
    closest_point_on_line, closest_point_on_point = nearest_points(closest_line, item.geometry)
    nitem = dict(item)
    nitem['geometry'] = closest_point_on_line
    nitem['u'] = str(closest_edge['u'])
    nitem['v'] = str(closest_edge['v'])
    nitem['uv'] = str(closest_edge['u']) + '-' + str(closest_edge['v'])
    new_items.append(nitem)
new_sensor_gdf = gpd.GeoDataFrame(new_items)
new_sensor_gdf.crs='epsg:4326'


# In[13]:


new_sensor_gdf_3310 = new_sensor_gdf.to_crs('epsg:3310')


# In[59]:


fosm_nodes_3310


# In[60]:


Map([
    Layer(fosm_nodes_3310,  basic_style(color='black'),
         popup_click=[popup_element('osmidstr')]),
    Layer(osm_others,  basic_style(color='#bbbbbb')),
    Layer(osm_secondary,  basic_style(color='#777777')),
    Layer(osm_primary,  basic_style(color='black')),
    Layer(osm_motorway,  basic_style(color='blue')),
    Layer(gpd.GeoDataFrame(geometry=closest_line_list)),
    Layer(sensor_gdf, basic_style(color='pink'),
          popup_click=[popup_element('sid')], popup_hover=[popup_element('sid')]),
    Layer(new_sensor_gdf, basic_style(color='red'),
          popup_click=[popup_element('sid')], popup_hover=[popup_element('sid')]),
])


# In[15]:


osm_nodes.crs = 'epsg:4326'
osm_nodes_3310 = osm_nodes.to_crs('epsg:3310')
osm_edges.crs = 'epsg:4326'
osm_edges_3310 = osm_edges.to_crs('epsg:3310')


# In[16]:


from shapely.geometry import MultiPoint

# fosm_edges_3310 = osm_edges_3310[osm_edges_3310.distance(sensor_hull_3310) < 1000]
fosm_edges_3310 = osm_edges_3310
fosm_edges_3310 = fosm_edges_3310[fosm_edges_3310['highway'] != 'residential'].copy()
# fosm_edges_3310['osmidstr'] = fosm_edges_3310['osmid'].astype(str)


osmidpos = {osmidstr: (x, y) for osmidstr, y, x in 
                zip(osm_nodes_3310['osmidstr'], osm_nodes_3310.geometry.y, osm_nodes_3310.geometry.x)}
# mffgdf_edges = mosm_edges[mosm_edges['u'].astype(str).isin(osmidpos) & mosm_edges['v'].astype(str).isin(osmidpos)]

coefficients = [1, 0.9, 0.8]
amygraph_list = dict()
for coef in coefficients:
    tamygraph = dict()
    for _, item in fosm_edges_3310.iterrows():
        us = str(item['u'])
        vs = str(item['v'])
        
        if us not in osmidpos or vs not in osmidpos:
            continue
        dist = item['length']

        if item['highway'] == 'motorway':
            dist *= coef

        tamygraph.setdefault(us, {'pos': osmidpos[us]})
        tamygraph.setdefault(vs, {'pos': osmidpos[vs]})
        tamygraph[us][vs] = dist
    amygraph_list[coef] = tamygraph
    


# In[17]:


appeared_nodes = []
for tamygraph in amygraph_list.values():
    appeared_nodes.extend(list(tamygraph.keys()))
appeared_nodes = list(set(appeared_nodes))

fosm_nodes_3310 = osm_nodes_3310[osm_nodes_3310['osmidstr'].isin(appeared_nodes)]


# In[18]:


import random

# osmid2geo = {osmid:geo for osmid, geo in zip(mmfgdf_nodes['osmidstr'], mmfgdf_nodes['geometry'])}
osmid2geo_3310 = {osmid:geo for osmid, geo in zip(osm_nodes_3310['osmidstr'], osm_nodes_3310['geometry'])}

elem_list = fosm_nodes_3310['osmidstr'].tolist()


# In[19]:


import heapq
from typing import Dict, List
            
distances = {}
def a_star(graph: Dict[str, Dict[str, float]], 
           start: str, end: str) -> List[str]:
    # Heuristic function for estimating the distance between two nodes
    def h(node):
        if (node, end) not in distances:
            distances[(node, end)] = distances[(end, node)] = osmid2geo_3310[node].distance(osmid2geo_3310[end])
        return distances[(node, end)]
    
    # Initialize distance and previous node dictionaries
    g_score = {node: float('inf') for node in graph}
    g_score[start] = 0
    f_score = {node: float('inf') for node in graph}
    f_score[start] = h(start)
    prev = {node: None for node in graph}
    
    # Initialize heap with start node and its f score
    heap = [(f_score[start], start)]
    
    while heap:
        # Pop the node with the smallest f score from the heap
        (f, curr_node) = heapq.heappop(heap)
        
        # If we have reached the end node, return the shortest path
        if curr_node == end:
            path = []
            while curr_node is not None:
                path.append(curr_node)
                curr_node = prev[curr_node]
                
            return path[::-1]
        
        # Otherwise, update the f and g scores of all adjacent nodes
        for neighbor, weight in graph[curr_node].items():
            # Check if there is an edge between the current node and the neighbor
            if neighbor not in g_score:
                continue
                
            new_g_score = g_score[curr_node] + weight
            if new_g_score < g_score[neighbor]:
                g_score[neighbor] = new_g_score
                f_score[neighbor] = new_g_score + h(neighbor)
                prev[neighbor] = curr_node
                heapq.heappush(heap, (f_score[neighbor], neighbor))
    
    # If we get here, there is no path from start to end
    return None


# In[20]:


myedges = fosm_edges_3310.copy()
myedges['edgeid'] = range(len(myedges))


# In[21]:


print('Used for navigation nodes and edges:', len(elem_list), len(myedges))


# In[22]:


path_dict = dict()
for _, item in fosm_edges_3310.iterrows():
    path_dict[(str(item['u']), str(item['v']))] = item.geometry


# In[ ]:





# In[67]:


help(basic_style)


# In[73]:


tmp_gdf.geometry.translate(1, 0)


# In[85]:


sample_O = '123031567' #elem_list[-100]
sample_D = '14956249' # elem_list[45]
tmp_gdf_list = []
mcolor = ['#5555FF', '#FF55FF', '#55ee55']
for ik, k in enumerate(amygraph_list):
    path = a_star(amygraph_list[k], sample_O, sample_D)
    path_list = []
    for i in range(len(path)-1):
        u, v = path[i], path[i+1]
        path_list.append(path_dict[(u, v)])

    tmp_gdf = gpd.GeoDataFrame(geometry=path_list, crs='epsg:3310')
    if ik < 2:
        tmp_gdf.geometry = tmp_gdf.geometry.translate(ik*100, ik*100)
#     else: tmp_gdf.geometry = tmp_gdf.geometry.translate(ik*100, ik*100)
    tmp_gdf_list.append(tmp_gdf)

Map(
    [
        Layer(osm_motorway,  basic_style(color='black'))
    ] 
    + [
        Layer(tmp_gdf, basic_style(color=mcolor[i], size = 5)) for i, tmp_gdf in enumerate(tmp_gdf_list)
    ]
    + [
        Layer(new_sensor_gdf, basic_style(color='gray'),
          popup_click=[popup_element('sid')], popup_hover=[popup_element('sid')]),
    ]
    + [
        Layer(fosm_nodes_3310[fosm_nodes_3310['osmidstr'].isin([sample_O])]
             , basic_style(color='red', size=20))
    ] 
    + [
        Layer(fosm_nodes_3310[fosm_nodes_3310['osmidstr'].isin([sample_D])]
             , basic_style(color='orange', size=20))
    ] 
)
    


# In[ ]:





# In[23]:


uv2edgeid = {(str(u), str(v)):eid for u, v, eid in zip(myedges['u'], myedges['v'], myedges['edgeid'])}


# In[24]:


mitem_list = dict()
for _, item in new_sensor_gdf_3310.iterrows():
    tmyedges = myedges[myedges['highway'] == 'motorway']
    found_edge = tmyedges.distance(item.geometry).argmin()
    
    mitem = dict(tmyedges.iloc[found_edge])
    
    mitem_list.setdefault(found_edge, mitem)
    mitem_list[found_edge].setdefault('sid2dist', dict())
    
    node_u = fosm_nodes_3310[fosm_nodes_3310['osmidstr'] == str(mitem['u'])].iloc[0]
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

eid2sw  = {eid:sw for eid, sw in zip(synch_gdf['edgeid'], synch_gdf['sensors'])}


# In[25]:


MAX_SEARCH_TRIAL = 3

x1, y1, x2, y2 = sensor_gdf_3310.total_bounds
GRID_W = int((x2 - x1) / GRID_UNIT_METER) + 1
GRID_H = int((y2 - y1) / GRID_UNIT_METER) + 1

print('GRID_H+2, GRID_W+2', GRID_H+2, GRID_W+2)

grid_width = (x2 - x1) / GRID_W
grid_height = (y2 - y1) / GRID_H

x1 = x1 - grid_width
y1 = y1 - grid_height
x2 = x2 + grid_width
y2 = y2 + grid_height

from shapely.geometry import Polygon

grid_items = []
for j in range(GRID_H+2):
    for i in range(GRID_W+2):
        by, bx = y1 + j*grid_height, x1 + i*grid_width
        
        # Define the vertices of the square
        vertices = [(bx, by), (bx, by + grid_height), 
                    (bx + grid_width, by + grid_height), (bx + grid_width, by)]
        
        # Create a polygon object from the vertices
        square = Polygon(vertices)
        grid_items.append(square)

grid_gdf = gpd.GeoDataFrame(geometry=grid_items, crs='epsg:3310')
grid_gdf['idx'] = range(len(grid_gdf))


# In[26]:


grid_idx_elems = dict()
for _, item in tqdm.tqdm(grid_gdf.iterrows(), total=len(grid_gdf)):
    idx = item['idx']
    grid_idx_elems[idx] = fosm_nodes_3310[fosm_nodes_3310.intersects(item.geometry)]['osmidstr'].tolist()
    
grid_gdf['num_nodes'] = [len(grid_idx_elems[idx]) for idx in grid_gdf['idx'].tolist()]

# Rough path existence check

grid_connectivity = np.eye(len(grid_gdf))
for _, item in tqdm.tqdm(myedges.iterrows(), total=len(myedges)):
    connected_grids = np.arange(len(grid_gdf))[grid_gdf.intersects(item.geometry)]
    for k, idx in enumerate(connected_grids):
        for jdx in connected_grids[k+1:]:
            grid_connectivity[idx, jdx] = grid_connectivity[jdx, idx] = 1


# In[27]:


new_connectivity = grid_connectivity.copy()
prev_connectivity = np.zeros_like(new_connectivity)
trial = 0
while np.sum(new_connectivity != prev_connectivity) != 0:
    trial += 1
    print(trial)
    prev_connectivity = new_connectivity > 0
    new_connectivity = (new_connectivity + new_connectivity @ new_connectivity) > 0
grid_connectivity = new_connectivity


# In[28]:


Map([
    Layer(grid_gdf[grid_gdf['num_nodes'] > 0], color_continuous_style('num_nodes', opacity=.5), popup_click=popup_element('idx')),
    Layer(new_sensor_gdf)
    
])


# In[29]:


import time

# Start measuring the execution time
start_time = time.time()

# Code block to measure execution time
# Place your code here that you want to measure

discover_path_list = []
with open(f'{DATASET_NAME}/generated_paths_{EXECUTION_DATE}.txt', 'w') as fp:
    for i in tqdm.tqdm(range(len(grid_gdf))):
        for j in range(len(grid_gdf)):
            if i == j or grid_connectivity[i, j] == 0:
                continue
            if len(grid_idx_elems[i]) < 5 or len(grid_idx_elems[j]) < 5:
                continue

            for tamygraph in amygraph_list.values():
                path = None
                for _ in range(MAX_SEARCH_TRIAL): #max trial
                    rn1 = random.choice(grid_idx_elems[i])
                    rn2 = random.choice(grid_idx_elems[j])
                    while rn1 == rn2:
                        rn2 = random.choice(grid_idx_elems[j])

                    path = a_star(tamygraph, rn1, rn2)
                    if path:
                        discover_path_list.append(path)
                        fp.write(' '.join(path) + '\n')

                    

# End measuring the execution time
end_time = time.time()

# Calculate the elapsed time
elapsed_time = end_time - start_time

# Print the elapsed time
print(f"Elapsed time: {elapsed_time} seconds")


# In[30]:


print('len(discover_path_list)', len(discover_path_list))


# In[31]:


fmotorway = fosm_edges_3310[fosm_edges_3310['highway'] == 'motorway']
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


# In[32]:


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

# In[33]:


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


# In[34]:


plt.matshow(sim_array)


# In[35]:


print('node2vec distance_graph_loaded', np.count_nonzero(sim_array))


# # Co-occurence Matrix

# In[36]:


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


# In[37]:


sim_array2 = np.eye(NUM_SENSORS)

for i in range(NUM_SENSORS):
    for j in range(i, NUM_SENSORS):
        w = 'S'+ind_to_sensor_id[i]
        w2 = 'S'+ind_to_sensor_id[j]
        wc1 = co_occurrence_vectors.loc[w, w]
        wc2 = co_occurrence_vectors.loc[w2, w2]
        sim_array2[j, i] = sim_array2[i, j] = co_occurrence_vectors.loc[w2, w]/((wc1*wc2)**.5+1)


# In[38]:


plt.matshow(sim_array2)


# In[39]:


with open(f'{DATASET_NAME}/cooccur_sim_{EXECUTION_DATE}.pkl', 'wb') as f:
    pickle.dump([sensor_ids, sensor_id_to_ind, sim_array2], f, protocol=2)


# In[40]:


print('cooccur distance_graph_loaded', np.count_nonzero(sim_array2))


# # Reachable Distance Matrix

# In[41]:


sid2eid = dict()
for eid, sw in eid2sw.items():
    sids = sw.split()
    for sid in sids:
        sid2eid[sid] = eid


# In[42]:


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


# In[43]:


def sid_dist(sid1, sid2, track_paths):
    between_sid_dist = 0
    sgeo1 = new_sensor_gdf_3310[new_sensor_gdf_3310['sid'] == int(sid1[1:])].iloc[0].geometry
    sgeo2 = new_sensor_gdf_3310[new_sensor_gdf_3310['sid'] == int(sid2[1:])].iloc[0].geometry
    if len(track_paths) == 2:
        between_sid_dist = sgeo1.distance(sgeo2)
    else:
        edge1 = myedges[myedges['edgeid'] == sid2eid[sid1]].iloc[0]
        edge2 = myedges[myedges['edgeid'] == sid2eid[sid2]].iloc[0]

        edge1_v = osm_nodes_3310[osm_nodes_3310['osmidstr'] == str(edge1.v)].iloc[0].geometry
        edge2_u = osm_nodes_3310[osm_nodes_3310['osmidstr'] == str(edge2.u)].iloc[0].geometry

        between_sid_dist += sgeo1.distance(edge1_v)
        between_sid_dist += edge2_u.distance(sgeo2)

        rest_paths = track_paths[1:-1]
        for i, node_u in enumerate(rest_paths[:-1]):
            node_v = rest_paths[i+1]
            node_u, node_v = int(node_u), int(node_v)
            between_sid_dist += myedges[(myedges['u'] == node_u) & (myedges['v'] == node_v)].iloc[0].geometry.length
            
    return between_sid_dist


# In[44]:


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
            


# In[45]:


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


# In[46]:


plt.matshow(dist_mat)


# In[47]:


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


# In[48]:


print('Original\t distance_graph_loaded', np.count_nonzero(adj_mx))
print('Node2vec\t distance_graph_loaded', np.count_nonzero(sim_array))
print('Cooccur \t distance_graph_loaded', np.count_nonzero(sim_array2))
print('DistMat \t distance_graph_loaded', np.count_nonzero(dist_mat))
print('New Dist\t distance_graph_loaded', np.count_nonzero(new_adj_mx))
print('Final   \t distance_graph_loaded', np.count_nonzero(final_adj_mx))


# In[53]:


plt.matshow(sim_array2)
plt.matshow(new_adj_mx)
plt.matshow(final_adj_mx)


# In[ ]:




