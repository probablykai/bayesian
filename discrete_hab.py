# discrete bayesian net for hab

## imports 
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
import time
import networkx as nx
import matplotlib.pyplot as plt
from pgmpy.inference import VariableElimination


# -----------------GOAL---------------- #
# find most likely anomaly given Prob(anomaly|low ppO2)
# two evidences requires nosiy_MAX

## functions
# create cpd
def gen_cpd(symptom,evidence,data):
    
    #check inputs
    # print('symptom', symptom)
    # print('evidence', evidence)
    # print('data', data)
    
    prob = {'low' : [],
            'nom' : [],
            'high': []}
    
    # loop through each bit to assemble CPD one column at a time
    for i in range(2**len(evidence)):
        low_sum = 0       # create containers
        nom_sum = 0 
        high_sum = 0 
        for j in range(len(evidence)):
            # Use bitwise operation to determine if the bit is set( 0 or 1)
            bit = (i >> j) & 1
            if bit == 1:
                low_sum += data['low'][j]
                nom_sum += data['nom'][j]
                high_sum += data['high'][j]
                
            low_avg = low_sum/len(evidence)  # avg from sum of true evidences, divided by number of evidences
            nom_avg = nom_sum/len(evidence)
            high_avg = high_sum/len(evidence)

        prob['low'].append(low_avg)
        prob['nom'].append(nom_avg)
        prob['high'].append(high_avg)

    prob['low'].remove(0)               # insert 0 column and replace with guess distribution
    prob['nom'].remove(0)
    prob['high'].remove(0)
    prob['low'].insert(0, .1)
    prob['nom'].insert(0, .8)
    prob['high'].insert(0, .1)

    prob = normalize(prob)
       


    card_list = [2 for a in range(len(all_anomalies[symptoms[node]]))] # define cardinality for correct number of evidences
    return TabularCPD(symptom,3,[prob['low'],prob['nom'],prob['high']], evidence=evidence, evidence_card=card_list)
                        # evidence=evidence) # declare CPD variables as inputed evidence

def normalize(data : dict):

    normed_data = {'low':[], 'nom':[], 'high':[] }

    low_val0 = data['low'].pop(0)       # remove 0th index that was already guessed [.1, .8, .1]
    nom_val0 = data['nom'].pop(0) 
    high_val0 = data['high'].pop(0) 
    low_val = data['low']   
    nom_val = data['nom']
    high_val = data['high']

    for col in range(len(low_val)): # low, nom, and high have same columns

        alpha = 1 / (low_val[col] + nom_val[col] + high_val[col])                       # normalization factor
     
        normed_data['low'].append(low_val[col] * alpha) 
        normed_data['nom'].append(nom_val[col] * alpha)
        normed_data['high'].append(high_val[col] * alpha)

    normed_data['low'].insert(0, low_val0)         # add back popped vals
    normed_data['nom'].insert(0, nom_val0)
    normed_data['high'].insert(0, high_val0)

    return normed_data

def gen_marginal(anomaly, data):
    return TabularCPD(anomaly,2,[[1-data],[data]])

def visualize_dag(model):
    # Create a directed graph from the model
    G = nx.DiGraph()
    G.add_edges_from(model.edges())

    # Set up the plot
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G)

    # Draw the nodes
    nx.draw_networkx_nodes(G, pos, node_size=1000, node_color='darkred')
    nx.draw_networkx_edges(G, pos, edge_color='pink', arrows=True, arrowsize=20)
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')

    # # Add CPTs as text next to nodes
    # for node in G.nodes():
    #     cpd = hab.get_cpds(node)
    #     if cpd:
    #         cpt_text = str(cpd).split('\n')
    #         # cpt_text = cpt_text[:5] + ['...'] if len(cpt_text) > 5 else cpt_text  # Truncate if too long
    #         cpt_string = '\n'.join(cpt_text)
    #         x, y = pos[node]
    #         plt.text(x + 0.1, y + 0.1, cpt_string, fontsize=4, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))

    # Remove axis
    plt.axis('off')

    # Show the plot
    plt.title("HAB--DAG", fontsize=16)
    plt.tight_layout()
    plt.show()


## code
t1 = time.perf_counter(), time.process_time()   # (real, cpu)

#symptoms
symptoms = ['ppo2', 'ppco2', 'n2', 'humidity', 'pdu1_current', 'pdu2_current', 'cabinfan1_rpm',
            'cabinfan2_rpm', 'temp', 'butanone', 'fuel_cell', 'pressure', 'h2o']

# anomalies
all_anomalies = {   # ppo2_anomalies = all_anomalies['ppo2'] to unpack
'ppo2': ['CDRA', 'LIOH', 'FUEL', 'LOP', 'RWGSR'],
'ppco2': ['CDRA', 'LIOH', 'RWGSR'],
'n2': ['LOP', 'TANK'],
'humidity': ['CDRA', 'PDU1', 'PDU2', 'FAN1', 'FAN2'],
'pdu1_current': ['PDU1'],
'pdu2_current': ['PDU2'],
'cabinfan1_rpm': ['PDU1', 'FAN1'],
'cabinfan2_rpm': ['PDU2', 'FAN2'],
'temp': ['PDU1', 'PDU2', 'FAN1', 'FAN2'],
'butanone': ['PDU1', 'PDU2', 'FAN1', 'FAN2', 'TCCS'],
'fuel_cell': ['FUEL'],
'pressure': ['LOP', 'TANK'],
'h2o': ['RWGSR']
                    }
    
# probabilities
all_probs = {
'ppo2': {
    'low': [0.8, 0.7, 0.6, 0.8, 0.8],
    'nom': [0.2, 0.3, 0.4, 0.2, 0.2],
    'high': [0.0, 0.0, 0.0, 0.0, 0.0]},

'ppco2': {
    'low': [0.0, 0.0, 0.0],
    'nom': [0.2, 0.1, 0.3],
    'high': [0.8, 0.9, 0.7]},

'n2': {
    'low': [0.8, 0.0],
    'nom': [0.2, 0.4],
    'high': [0.0, 0.6]},

'humidity': {
    'low': [0.0, 0.0, 0.0, 0.0, 0.0],
    'nom': [0.4, 0.3, 0.3, 0.2, 0.2],
    'high': [0.6, 0.7, 0.7, 0.8, 0.8]},

'pdu1_current': {
    'low': [0.9],
    'nom': [0.1],
    'high': [0.0]},

'pdu2_current': {
    'low': [0.9],
    'nom': [0.1],
    'high': [0.0]},

'cabinfan1_rpm': {
    'low': [0.9, 0.0],
    'nom': [0.1, 0.1],
    'high': [0.0, 0.9]},

'cabinfan2_rpm': {
    'low': [0.9, 0.0],
    'nom': [0.1, 0.1],
    'high': [0.0, 0.9]},

'temp': {
    'low': [0.0, 0.0, 0.0, 0.0],
    'nom': [0.4, 0.4, 0.2, 0.2],
    'high': [0.6, 0.6, 0.8, 0.8]},

'butanone': {
    'low': [0.0, 0.0, 0.0, 0.0, 0.0],
    'nom': [0.4, 0.4, 0.3, 0.3, 0.1],
    'high': [0.6, 0.6, 0.7, 0.7, 0.9]},

'fuel_cell': {
    'low': [0.8],
    'nom': [0.2],
    'high': [0.0]},

'pressure': {
    'low': [0.8, 0.0],
    'nom': [0.2, 0.4],
    'high': [0.0, 0.6]},

'h2o': {
    'low': [0.9],
    'nom': [0.1],
    'high': [0.0]}
            }

# marginal probabilities
all_marginal = {
    "CDRA": 1e-4,
    "PDU1": 1e-4,
    "PDU2": 1e-4,
    "FAN1": 1e-2,
    "FAN2": 1e-2,
    "TCCS": 1e-3,
    "LIOH": 1e-3,
    "FUEL": 1e-4,
    "LOP": 1e-5,
    "TANK": 1e-5,
    "RWGSR": 1e-4
                }    
marginal_keys = list(all_marginal.keys()) # create accessible library for keys

# create the directed acyclic
hab = BayesianNetwork()
DAG = []
for key in all_anomalies:
    for anom in all_anomalies[key]:
        edge = (anom, key) # anomalies are the parents of the symptoms (dependent of the given marginal distributions)
        DAG.append(edge)
hab.add_edges_from(DAG) 

# define loop
total_nodes = len(symptoms)
node = 0

while node < total_nodes: # create CPD's for each symptom node until all created
    
    # add symptom CPDs
    current_symptom = symptoms[node]
    new_cpd = gen_cpd(current_symptom, all_anomalies[current_symptom] , all_probs[current_symptom])
    hab.add_cpds(new_cpd)
    print(new_cpd)

    node += 1 # go to the next iteration

# define parent loop
total_par = len(all_marginal)
par_node = 0

while par_node < total_par:
    # add anomaly marginals
    current_anomaly = marginal_keys[par_node]
    new_marg = gen_marginal(current_anomaly, all_marginal[current_anomaly])
    hab.add_cpds(new_marg)

    par_node += 1 # go to the next iteration


# visualize_dag(hab)

inference = VariableElimination(hab)

prob_query = inference.query(['RWGSR'],{'h2o':0})

# print out probability distribution
print(prob_query)

t2 = time.perf_counter(), time.process_time()


print(f" Real time:\t{t2[0] - t1[0]:.2f} [s]") 
print(f" CPU time:\t{t2[1] - t1[1]:.2f} [s]") 