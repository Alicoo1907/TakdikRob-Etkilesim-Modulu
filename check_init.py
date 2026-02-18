import torch
import sys
import os
import yaml
from collections import OrderedDict

# Add paths
sys.path.append(os.path.join(os.getcwd(), 'FC-GCN', 'FC-GCN'))
from fcsa_gcn import Model

CONFIG_PATH = 'FC-GCN/FC-GCN/config/ntu60_xview.yaml'
WEIGHTS_PATH = 'FC-GCN/FC-GCN/work_dir/ntu60/xview/joint/best_model.pt'

with open(CONFIG_PATH, 'r') as f:
    args = yaml.load(f, Loader=yaml.SafeLoader)

m_args = args['model_args']
print(f"Config num_person: {m_args['num_person']}")
print(f"Config input_channels: {m_args['input_channels']}")

model = Model(
    num_classes=m_args['num_classes'],
    residual=m_args['residual'],
    dropout=m_args['dropout'],
    num_person=m_args['num_person'],
    graph=m_args['graph'],
    num_nodes=m_args['num_nodes'],
    input_channels=m_args['input_channels']
)

print(f"Initial BN size: {model.gfe_one.bn.running_mean.shape}")

weights = torch.load(WEIGHTS_PATH, map_location='cpu')
new_weights = OrderedDict()
for k, v in weights.items():
    name = k.replace("module.", "")
    new_weights[name] = v

try:
    model.load_state_dict(new_weights)
    print("Loaded weights successfully.")
except Exception as e:
    print(f"Load failed: {e}")

print(f"BN size after load: {model.gfe_one.bn.running_mean.shape}")
