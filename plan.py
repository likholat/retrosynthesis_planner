import json
import molvs
import random
import policies
from tqdm import tqdm
from mcts import Node, mcts
import tensorflow as tf
from rdkit import Chem
from rdkit.Chem import AllChem

import numpy as np
import argparse
from collections import namedtuple

import time

start = time.time()
# Load base compounds
starting_mols = set()
with open('data/emolecules.smi', 'r') as f:
    for line in tqdm(f, desc='Loading base compounds'):
        try:
            smi = line.strip()
            smi = molvs.standardize_smiles(smi)
            starting_mols.add(smi)

            if len(starting_mols) == 10:
                break
        except Exception as e:
            # pass
            print('WARNING', e)
end = time.time()
print('TIME:')
print(end - start)
# exit()
print('Base compounds:', len(starting_mols))

# Load policy networks
with open('model/rules.json', 'r') as f:
    rules = json.load(f)
    rollout_rules = rules['rollout']
    expansion_rules = rules['expansion']
    expansion_rules = {idx: rule for rule, idx in expansion_rules.items()}

rollout_net = policies.RolloutPolicyNet(n_rules=len(rollout_rules), is_training=False)
expansion_net = policies.ExpansionPolicyNet(n_rules=len(expansion_rules), is_training=False)
filter_net = policies.InScopeFilterNet()

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
saver = tf.train.Saver()
saver.restore(sess, 'model/model.ckpt')

# Results for tests
exp_res = [[],[]]
rollout_res = []

# Get input nodes shapes
exp_net_x_shape = int(expansion_net.X.shape[1])
roll_net_x_shape = int(rollout_net.X.shape[1])

# Openvino networks
exp_net, exp_exec_net = None, None
roll_exec_net = None

# Input nodes names
exp_inp_node_x_name, exp_inp_node_k_name = None, None
roll_inp_node_name = None


def transform(mol, rule):
    """Apply transformation rule to a molecule to get reactants"""
    rxn = AllChem.ReactionFromSmarts(rule)
    results = rxn.RunReactants([mol])

    if not results:
        return []

    # Only look at first set of results (TODO any reason not to?)
    results = results[0]
    reactants = [Chem.MolToSmiles(smi) for smi in results]
    return reactants


def expansion(node, use_openvino=False):
    """Try expanding each molecule in the current state
    to possible reactants"""

    # Assume each mol is a SMILES string
    mols = node.state

    # Convert mols to format for prediction
    # If the mol is in the starting set, ignore
    mols = [mol for mol in mols if mol not in starting_mols]
    fprs = policies.fingerprint_mols(mols, exp_net_x_shape)

    if not use_openvino:
        # Predict applicable rules
        preds = sess.run(expansion_net.pred, feed_dict={
            expansion_net.X: fprs,
            expansion_net.k: 5
        })

        exp_res[0].append(preds.indices[0])
        exp_res[1].append(preds.values[0])

    else:
        # Get output nodes names
        values_blob = list(iter(exp_net.outputs))[0]
        indices_blob = list(iter(exp_net.outputs))[1]

        # Predict applicable rules
        preds = exp_exec_net.infer(inputs={exp_inp_node_x_name: fprs})
        
        indices = preds[indices_blob][0]
        values = preds[values_blob][0]

        exp_res[0].append(indices)
        exp_res[1].append(values)

        top_k = namedtuple('TopKV2', 'values indices')
        preds = top_k([values], [indices])

    # Generate children for reactants
    children = []
    for mol, rule_idxs in zip(mols, preds.indices):
        # State for children will
        # not include this mol

        smol = set(mol)
        new_state = [x for x in mols if x in smol]

        mol = Chem.MolFromSmiles(mol)
        for idx in rule_idxs:
            # Extract actual rule
            rule = expansion_rules[idx]

            # TODO filter_net should check if the reaction will work?
            # should do as a batch

            # Apply rule
            reactants = transform(mol, rule)

            if not reactants: continue

            state = new_state | set(reactants)
            terminal = all(mol in starting_mols for mol in state)
            child = Node(state=state, is_terminal=terminal, parent=node, action=rule)
            children.append(child)
    return children


def rollout(node, max_depth=200, use_openvino=False):
    cur = node
    for _ in range(max_depth):
        if cur.is_terminal:
            break

        # Select a random mol (that's not a starting mol)
        mols = [mol for mol in cur.state if mol not in starting_mols]
        mol = random.choice(mols)
        fprs = policies.fingerprint_mols([mol], roll_net_x_shape)

        if not use_openvino:
            # Predict applicable rules
            preds = sess.run(rollout_net.pred_op, feed_dict={
                rollout_net.X: fprs,
            })

            rollout_res.append(preds[0])

        else:
            # Predict applicable rules
            preds = roll_exec_net.infer(inputs={roll_inp_node_name: fprs})
            preds = next(iter(preds.values()))

            rollout_res.append(preds[0])

        rule = expansion_rules[preds[0]]
        reactants = transform(Chem.MolFromSmiles(mol), rule)
        state = cur.state | set(reactants)

        # State for children will
        # not include this mol
        state = state - {mol}

        terminal = all(mol in starting_mols for mol in state)
        cur = Node(state=state, is_terminal=terminal, parent=cur, action=rule)

    # Max depth exceeded
    else:
        print('Rollout reached max depth')

        # Partial reward if some starting molecules are found
        reward = sum(1 for mol in cur.state if mol in starting_mols)/len(cur.state)

        # Reward of -1 if no starting molecules are found
        if reward == 0:
            return -1.

        return reward

    # Reward of 1 if solution is found
    return 1.


def convert_to_ir(sess, input_shape, output_node_names, input_node_names=None, placeholder_value=None):
    import subprocess
    import sys
    import mo_tf
    import os

    from openvino.inference_engine import IECore
    ie = IECore()

    pb_model_path = 'model.pb'
    model_xml = 'model.xml'
    model_bin = 'model.bin'

    # Save frozen graph
    graph_def = tf.graph_util.convert_variables_to_constants(
            sess,
            sess.graph_def,
            output_node_names)

    with tf.io.gfile.GFile(pb_model_path, 'wb') as f:
        f.write(graph_def.SerializeToString())

    inp_shape = [] 

    for shape in input_shape:
        inp_shape.append([1])
        if shape > 0:
            inp_shape[-1].append(shape)

    inp_shape = str(inp_shape)[1:-1]
    convert_script = [sys.executable, mo_tf.__file__, 
                    '--input_model', pb_model_path, '--input_shape', inp_shape]

    if input_node_names:
        convert_script.append('--input')
        convert_script.append(','.join(input_node_names))

    if placeholder_value:
        convert_script.append('--freeze_placeholder_with_value'),
        convert_script.append(placeholder_value)

    # Convert to Openvino IR
    subprocess.run(convert_script, check=True)

    net = ie.read_network(model=model_xml, weights=model_bin)
    exec_net = ie.load_network(network=net, device_name='CPU')

    os.remove(pb_model_path)
    os.remove(model_xml)
    os.remove(model_bin)

    return net, exec_net


def plan(target_mol, use_openvino=False):
    """Generate a synthesis plan for a target molecule (in SMILES form).
    If a path is found, returns a list of (action, state) tuples.
    If a path is not found, returns None."""
    root = Node(state={target_mol})

    if use_openvino:
        global exp_net, exp_exec_net, roll_exec_net
        global exp_inp_node_x_name, exp_inp_node_k_name, roll_inp_node_name

        # Get input nodes names
        exp_inp_node_x_name = expansion_net.X.name.split(':')[0]
        exp_inp_node_k_name = expansion_net.k.name.split(':')[0]
        roll_inp_node_name = rollout_net.X.name.split(':')[0]

        exp_input_nodes = [exp_inp_node_x_name, exp_inp_node_k_name]
        exp_output_nodes = ['TopKV2']
        roll_output_nodes = ['ArgMax']

        # Get Openvino networks
        exp_net, exp_exec_net = convert_to_ir(sess, [exp_net_x_shape, 0], exp_output_nodes, exp_input_nodes, "Placeholder_2->5")
        _, roll_exec_net = convert_to_ir(sess, [roll_net_x_shape], roll_output_nodes)

    path = mcts(root, expansion, rollout, iterations=2000, max_depth=200, use_openvino=use_openvino)
    if path is None:
        print('No synthesis path found. Try increasing `iterations` or `max_depth`.')
    else:
        print('Path found:')
        path = [(n.action, n.state) for n in path[1:]]
    return path


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Select model representation")
    parser.add_argument("--use_openvino", help="Enable OpenVINO optimization", action='store_true')
    args = parser.parse_args()

    # target_mol = '[H][C@@]12OC3=C(O)C=CC4=C3[C@@]11CCN(C)[C@]([H])(C4)[C@]1([H])C=C[C@@H]2O'
    target_mol = 'CC(=O)NC1=CC=C(O)C=C1'
    root = Node(state={target_mol})
    path = plan(target_mol, use_openvino=args.use_openvino)
    
    if not args.use_openvino:
        np.save('expansion_preds_tf', exp_res)
        np.save('rollout_preds_tf', rollout_res)
    else:
        np.save('expansion_preds_ov', exp_res)
        np.save('rollout_preds_ov', rollout_res)

    # import ipdb; ipdb.set_trace()
