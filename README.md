# [ArXiv 2026] Text-Guided 6D Object Pose Rearrangement via Closed-Loop VLM Agents

## [Project Page](https://tlb-miss.github.io/vlmpose/) &nbsp;|&nbsp; [Paper](https://arxiv.org/pdf/2604.09781) 

## Installation

### System Requirements

This code has been tested in the following settings, but is expected to work in other systems. 
- Ubuntu 20.04
- CUDA 12.1
- NVIDIA RTX 5880 Ada

### Conda Environment
``` bash
conda create -n vlmpose python=3.10
conda activate vlmpose
pip install openai==2.17.0 pyrender==0.1.45 trimesh[easy] open3d==0.19.0
pip uninstall numpy
pip install -U "numpy>=1.23.5,<1.25"
pip install -e .
```

## Inference

### Mesh Setting
``` bash
export OPENAI_API_KEY=your_openai_key
# Example 1
python3 mesh_setting.py  --target_scene_dir demo_data/table_teacup_teapot --text_prompt 'Pour the tea into the teacup using the teapot. The teapot is held in a natural pouring pose, with the handle slightly raised and the spout tilted downward toward the teacup.'
# Example 2
python3 mesh_setting.py  --target_scene_dir demo_data/chess --text_prompt 'Move the black knight from g8 to f6 using standard chessboard coordinates. Place the knight centered entirely within square f6, without crossing into adjacent squares.'
```

### Tip

As in the example, the more detailed you write the ``--text_prompt``, the more likely it is to work as intended.

## TODO

  - [ ] Code for RGB-D setting
  - [ ] Code for open-source models (e.g., Qwen)

## Citation
```bibtex
@article{vlmpose,
      title={Text-Guided 6D Object Pose Rearrangement via Closed-Loop VLM Agents}, 
      author={Baik, Sangwon and Kim, Gunhee and Choi, Mingi and Joo, Hanbyul},
      journal={arXiv:2604.09781},
      year={2026}
}
```