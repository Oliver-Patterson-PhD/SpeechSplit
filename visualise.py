import os
import argparse
import torchview
import solver
import data_loader
import hparams


def str2bool(v):
    return v.lower() in ("true")


# fmt: off
parser = argparse.ArgumentParser()
parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for Adam optimizer')
parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam optimizer')
parser.add_argument('--device_id', type=int, default=0)
parser.add_argument('--g_lr', type=float, default=0.0001, help='learning rate for G')
parser.add_argument('--log_dir', type=str, default='run/logs')
parser.add_argument('--log_step', type=int, default=10)
parser.add_argument('--model_save_dir', type=str, default='run/models')
parser.add_argument('--model_save_step', type=int, default=1000)
parser.add_argument('--num_iters', type=int, default=1000000, help='number of total iterations')
parser.add_argument('--resume_iters', type=int, default=None, help='resume training from this step')
parser.add_argument('--sample_dir', type=str, default='run/samples')
parser.add_argument('--sample_step', type=int, default=1000)
parser.add_argument('--use_tensorboard', type=str2bool, default=False)
# fmt: on

config = parser.parse_args()
print(config)
print(hparams.hparams_debug_string())

if not os.path.exists(config.log_dir):
    os.makedirs(config.log_dir)
if not os.path.exists(config.model_save_dir):
    os.makedirs(config.model_save_dir)
if not os.path.exists(config.sample_dir):
    os.makedirs(config.sample_dir)

# Data loader.
vcc_loader = data_loader.get_loader()

slvr = solver.Solver(None, config, hparams)

first_data = next(iter(vcc_loader))

model_graph = torchview.draw_graph(
    slvr.G,
    input_data=first_data,
    graph_name="SpeechSplit",
    expand_nested=True,
    hide_inner_tensors=False,
    hide_module_functions=False,
)

model_graph.visual_graph
