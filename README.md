# <img src="https://cage-policy.github.io/static/images/cage.png" width=26>CAGE: Causal Attention Enables Data-Efficient Generalizable Robotic Manipulation

[[Paper]](https://arxiv.org/pdf/2410.14974) [[Project Page]](https://cage-policy.github.io/) [[Sample Data]](https://drive.google.com/drive/folders/1BHY-hiYlHjHPNlvUMePhZ3alQ7dMYRtd?usp=sharing)

<p align="center">
  <img src="https://cage-policy.github.io/static/images/teaser.png" width="1000">
</p>
CAGE is a data-efficient generalizable robotic manipulation policy. Extensive eperiments demonstrate that CAGE can effectively complete the task in test environments with different levels of distribution shifts.

## 🛫 Getting Started

### 💻 Installation

To set up the conda environment for CAGE, we provide a minimal package requirement list for training & inference (See [CAGE.yaml](https://github.com/cage-policy/CAGE/blob/master/CAGE.yaml)), as well as a full environment export ([CAGE_full.yaml](https://github.com/cage-policy/CAGE/blob/master/CAGE_full.yaml)) for reproducibility reference.

```bash
conda env create -f CAGE.yaml
```

### 🛢️ Data Collection

We apply the data collection process in the <a href="https://rh20t.github.io/">RH20T</a> paper. We provide the sample data for each tasks on [Google Drive](https://drive.google.com/drive/folders/1BHY-hiYlHjHPNlvUMePhZ3alQ7dMYRtd?usp=sharing) and [Baidu Netdisk](https://pan.baidu.com/s/1l-P9fAqlrDHS1LvT5gNnYA) (code: vwxq). For detailed descriptions, please refer to [RISE data collection](https://github.com/rise-policy/rise?tab=readme-ov-file#%EF%B8%8F-data-collection).

### 🧑🏻‍💻 Training

We adopt `accelerate` and `deepspeed` for configurable single-GPU / multi-GPU training. The main training script is [train_cage.py](https://github.com/cage-policy/CAGE/blob/master/train_cage.py). Its config file is located in [configs](https://github.com/cage-policy/CAGE/tree/master/configs) with `cage` prefix. (e.g. `cage-rh20t.yaml` is used for training CAGE on the [RH20T](https://rh20t.github.io) dataset)

An example of the training command is as follows:

```
accelerate launch \
  --num_machines=1 \
  --num_processes=4 \
  --gpu_ids=0,1,2,3 \
  -m train_cage \
  --tag [real|rh20t] \
  [--test] \
  [--resume PATH] \
  [other parameters defined in the config file (e.g. batch_size=16 dataset.obs_horizon=2)]
```

To train on single GPU, you can simply set `--num_processes=1` and `--gpu_ids=0`.

With a batch size of 16, CAGE takes around 60 GB of VRAM on each GPU. To reduce the memory usage, you can use a smaller batch size combined with gradient accumulation (*e.g.*, `batch_size=8` with `gradient_accumulation_steps=2`) to fit the model on your GPUs.

### 📁 Checkpoint
Checkpoints are automatically saved each `checkpoint_steps` and after each epoch. Once the training is done, a `final_ckpt` will be saved for evaluation. 

If you want to evaluate the policy from an intermediate checkpoint, you should first extract `model.bin` from DeepSpeed states using the provided script under each checkpoint directory:

```bash
cd [path_to_checkpoint] && python zero_to_fp32.py . model.bin
```

Since `model.bin` is directly exported from the training states, frozen parameters of DINOv2 is also embedded. To further reduce the file size, you can use the following command to store only trainable parameters:

```bash
python merge_weight.py --unmerge --ckpt [path_to_model.bin] --config [path_to_config.yaml]
```

### 🤖 Inference 

Running inference on CAGE is as simple as:

```python
import os
from agent import CAGEAgent
from omegaconf import OmegaConf

conf = OmegaConf.load(os.path.join('configs', 'eval-cage.yaml'))
# create the inference agent from config file
agent = CAGEAgent(conf)

# each list is a sequence of observations in order of time
#   (the last element is the most recent one)
# 'proprio' is a list of end-effector pose of the robot
# If the model is trained without proprioception, 
#   the current pose of the robot is still required
#   for relative action calculation. (length should be 1)
obs_dict = {
  'global_cam': [...],  # [PIL.Image] * obs_horizon
  'wrist_cam': [...],   # [PIL.Image] * obs_horizon
  'proprio': [...],     # [np.ndarray] * obs_horizon or 1
}

# get next 8 actions based on current observations
xyz, rot, w = agent(obs_dict, act_horizon=8)
```
For advanced usage, please refer to [agent/cage.py](https://github.com/cage-policy/CAGE/tree/master/agent/cage.py), [agent/generic.py](https://github.com/cage-policy/CAGE/tree/master/agent/generic.py) and [eval.py](https://github.com/cage-policy/CAGE/tree/master/eval.py) for more details.

### 🤖 Evaluation

We provide a sample script [eval.py](https://github.com/cage-policy/CAGE/tree/master/eval.py) for evaluation on our platform (Flexiv Rizon 4 robotic arm + Dahuan AG-95 gripper) with the following command:

```
python eval.py \
  --config [path_to_config_file] \
  --ckpt [path_to_model_ckpt] \
  --ctrl_freq 10 \
  --pred_interval 4 \
  --t_ensemble
```

To evaluate on the platform with the same setup as ours, extra python libraries are required:

```bash
pip install pyserial modbus_tk pyrealsense2
```

And you also need to install the [Flexiv RDK](https://rdk.flexiv.com/manual/getting_started.html) for robot control. Specifically, download [FlexivRDK v0.9](https://github.com/flexivrobotics/flexiv_rdk/releases/tag/v0.9) and copy `lib_py/flexivrdk.cpython-310-[arch].so` to `hardware/robot` directory.

For other platforms, you should modify the codes in `hardware/` and the evaluation script to adapt to your own configuration.

## 📈 Results

As the level of distribution shift increases, the performance of selected 2D/3D baselines drops significantly, while **CAGE maintains a stable performance**, even when evaluating in a completely new environment. 

In *similar environments*, CAGE offers an average of **42% increase in task completion rate**. While all baselines fail to execute the task in *unseen environments*, CAGE manages to obtain **a 43% completion rate and a 51% success rate** in average.

**L0 Evaluation Results.**
<p align="center">
  <img src="https://cage-policy.github.io/static/images/base-results.png" width="1000">
</p>

**L1 Evaluation Results.**
<p align="center">
  <img src="https://cage-policy.github.io/static/images/l1-results.png" width="1000">
</p>

**L2 Evaluation Results.**
<p align="center">
  <img src="https://cage-policy.github.io/static/images/l2-results.png" width="1000">
</p>

## 🙏 Acknowledgement

- Our diffusion module is adapted from [Diffusion Policy](https://github.com/real-stanford/diffusion_policy). This part is under MIT License.
- Our real-world evaluation code is adapted from [RISE](https://github.com/rise-policy/rise). This part is under CC-BY-NC-SA 4.0 License.

## ✍️ Citation

```bibtex
@article{
  xia2024cage,
  title   = {CAGE: Causal Attention Enables Data-Efficient Generalizable Robotic Manipulation},
  author  = {Xia, Shangning and Fang, Hongjie and Fang, Hao-Shu and Lu, Cewu},
  journal = {arXiv preprint arXiv:2410.14974},
  year    = {2024}
}
```

## 📃 License

<a href="https://cage-policy.github.io/">CAGE</a> (including data and codebase) by <a href="https://github.com/Xiashangning">Shangning Xia</a>, <a href="https://tonyfang.net/">Hongjie Fang</a>, <a href="https://fang-haoshu.github.io/">Hao-Shu Fang</a>, <a href="https://www.mvig.org/">Cewu Lu</a> is licensed under <a href="https://creativecommons.org/licenses/by-nc-sa/4.0/?ref=chooser-v1" target="_blank" rel="license noopener noreferrer" style="display:inline-block;">CC BY-NC-SA 4.0<img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/cc.svg?ref=chooser-v1" alt=""><img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/by.svg?ref=chooser-v1" alt=""><img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/nc.svg?ref=chooser-v1" alt=""><img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/sa.svg?ref=chooser-v1" alt=""></a></p>
