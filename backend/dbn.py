from pgmpy.models import DynamicBayesianNetwork as DBN
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import DBNInference
from pgmpy.inference import VariableElimination
from pgmpy.inference import ApproxInference
import numpy as np
import pandas as pd
class Student_DBN:
    def __init__(self, root: str, nodes_names: list[str], problems_db):
        self.nodes_names = []
        self.dbn = DBN()
        self.nodes_names = nodes_names
        self.edges, self.nodes = self.set_nodes(root, nodes_names)
        self.dbn.add_edges_from(self.edges)
        self.problems_id = {item: [] for item in nodes_names if item != root}
        print(self.problems_id)
        self.set_problems_db(problems_db)
        self.set_cpds()
        self.dbn.initialize_initial_state()
        self.infer = ApproxInference(self.dbn)
        self.evidence_history = {}
        self.belief_history = {node: [] for node in self.nodes_names}
        self.posterior_belief = {node: [] for node in self.nodes_names}
    def set_nodes(self, root: str, nodes: list[str]) -> list[tuple[str, int]]: # Corrected type hint here
        edges_list = []
        nodes_list = [(f"UP {root}", 0), (f"UP {root}", 1)]
        edges_list.append(((f"UP {root}", 0), (f"UP {root}", 1)))
        for node in nodes:
            if node == root:
                continue
            nodes_list.append((f"UP {node}", 0))
            nodes_list.append((f"UP {node}", 1))
            nodes_list.append((f"CP {node}", 0))
            edges_list.append(((f"UP {root}", 0), (f"UP {node}", 0)))
            edges_list.append(((f"UP {node}", 0), (f"UP {node}", 1)))
            edges_list.append(((f"UP {node}", 0),(f"CP {node}", 0)))
        return edges_list, nodes_list
    def set_cpds(self):
        transition_matrix = np.array([
     [0.85, 0.75, 0.60, 0.40],
     [0.15, 0.25, 0.40, 0.60]
])
        up_node_matrix = [
             [0.8, 0.1],
             [0.2, 0.9]
                          ]
        cp_node_matrix = [[0.75, 0.1],
                          [0.25, 0.9]]
        p_node_matrix = [[0.8, 0.2],
                         [0.2, 0.8]]

        cpd_root_0 = TabularCPD(variable=self.nodes[0], variable_card=2, values=[[0.3], [0.7]])

        cpd_root_1 = TabularCPD(
            variable=self.nodes[1],
            variable_card=2,
            values=[[0.8, 0.25],
                    [0.2, 0.75]],
            evidence=[self.nodes[0]],
            evidence_card=[2]
        )
        self.dbn.add_cpds(cpd_root_0, cpd_root_1)

        for node in self.nodes:
            if node[0] == self.nodes[0][0]:
                continue
            if node[1] == 1 and node[0].startswith('UP'):
                parents = [self.nodes[1], (node[0], 0)]
                cpd = TabularCPD(
                    variable=node,
                    variable_card=2,
                    values=transition_matrix,
                    evidence=parents,
                    evidence_card=[2, 2]
                )
                self.dbn.add_cpds(cpd)
            elif node[1] == 0 and node[0].startswith('UP'):
                cpd = TabularCPD(
                    variable=node,
                    variable_card=2,
                    values=up_node_matrix,
                    evidence=[self.nodes[0]],
                    evidence_card=[2]
                )
                self.dbn.add_cpds(cpd)
            elif node[1] == 0 and node[0].startswith('CP'):
                parent = tuple(self.dbn.get_parents(node)[0])
                cpd = TabularCPD(
                    variable=node,
                    variable_card=2,
                    values=cp_node_matrix,
                    evidence=[parent],
                    evidence_card=[2]
                )
                self.dbn.add_cpds(cpd)
            elif node[1] == 0 and node[0].startswith('P'):
                parent = tuple(self.dbn.get_parents(node)[0])
                cpd = TabularCPD(
                    variable=node,
                    variable_card=2,
                    values=p_node_matrix,
                    evidence=[parent],
                    evidence_card=[2]
                )
                self.dbn.add_cpds(cpd)

        return
    def set_problems_db(self, problems):
        i = 0
        for item in problems.keys():
          for j in range(len(problems[item])):
            print((f'P {i}', 0), i)
            self.dbn.add_edges_from([((f'CP {item}', 0), (f'P {i}', 0))])
            self.nodes.append((f'P {i}', 0))
            self.edges.append(((f'CP {item}', 0), (f'P {i}', 0)))
            self.problems_id[item].append(i)
            i += 1
    def do_inference(self, node: str, evidence: list[int]):

        virtual_evidences = []
        for t, e in enumerate(evidence):
            actual_t_nodes = self.get_nodes_t_slice(t)
            posterior_t_nodes = self.get_nodes_t_slice(t+1)
            for i in self.problems_id[node]:
              var_name = (f'P {i}', t)
              if isinstance(e, (list, np.ndarray)):
                    soft_cpd = TabularCPD(
                        variable=var_name,
                        variable_card=2,
                        values=[[prob] for prob in e],

                    )
                    virtual_evidences.append(soft_cpd)


              else:
                    self.evidence_history[var_name] = e
            actual_belief_joint = self.infer.query(
                variables=actual_t_nodes,
                evidence=self.evidence_history,
                virtual_evidence=virtual_evidences
            )

            posterior_belief_joint = self.infer.query(
                variables=posterior_t_nodes,
                evidence=self.evidence_history,
                virtual_evidence=virtual_evidences
            )
            for i in range(len(self.nodes_names)):
                target_node_actual = actual_t_nodes[i]
                target_node_posterior = posterior_t_nodes[i]

                nodes_to_marginalize_actual = [n for n in actual_t_nodes if n != target_node_actual]
                actual_belief_node = actual_belief_joint.marginalize(nodes_to_marginalize_actual, inplace=False)

                nodes_to_marginalize_posterior = [n for n in posterior_t_nodes if n != target_node_posterior]
                posterior_belief_node = posterior_belief_joint.marginalize(nodes_to_marginalize_posterior, inplace=False)

                self.belief_history[self.nodes_names[i]].append(actual_belief_node.values[1])
                self.posterior_belief[self.nodes_names[i]].append(posterior_belief_node.values[1])
            return
    def get_nodes_t_slice(self, t):
        nodes_t_0 = [node for node in self.nodes if node[1] == 0 and not (node[0][0] == 'P')]
        return [(node, t) for node, i in nodes_t_0]
    def get_nodes_percentage(self):
        nodes_percentage = {}
        for node in self.belief_history.keys():
            nodes_percentage[node] = self.belief_history[node][-1]
        return nodes_percentage
# db = SessionLocal()
# insert_student(db, 'Vitor Gabriel', 0)
# insert_bayesian_network(db, 0, 'tutor_model', 'T', 0)
# insert_bayesian_network(db, 1, 'student_model', 'S', 0)


# for i, node in enumerate(model.belief_history):
#     insert_node(db, i, node, 'n', 0)
#     insert_node(db, len(model.belief_history)+i, node, 'n', 1)
#     insert_node_probability(db, model.belief_history[node][-1], 'domina', 'domina', i, 0)
#     insert_node_probability(db, student_model.belief_history[node][-1], 'domina', 'domina', len(model.belief_history)+i, 0)
        