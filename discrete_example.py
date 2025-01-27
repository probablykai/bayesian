from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete.CPD import TabularCPD
from pgmpy.inference import VariableElimination

graph = [('TANK', 'ppo2'), ('TANK', 'ppco2')]
hab = BayesianNetwork(graph)

# given two parameters, what is the probability of a tank burst

# Symptom CPD
symptoms = ['ppo2','ppco2']
for i in symptoms:
    sym_cpd = TabularCPD(i,3,[[.2,.6],
                              [.6,.2],
                              [.2,.2]],
                              evidence=['TANK'],
                              evidence_card=[2])
    
    hab.add_cpds(sym_cpd)

# TANK marginal CPD
TANK_marg = TabularCPD('TANK', 2, [[.2],[.8]])
hab.add_cpds(TANK_marg)

inference = VariableElimination(hab)

# probability P(TANK|ppo2=low,ppco2=nom)   --->  returns TANK=true  ?
# 0=low, 1=nom, 2=high
query = inference.map_query(['TANK'],{'ppo2':0, 'ppco2':1})
query1 = inference.query(['TANK'],{'ppo2':2, 'ppco2':1})

print(query1)