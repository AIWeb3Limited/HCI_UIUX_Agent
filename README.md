# HCI_UIUX_Agent
## Introduction
### AIGC Task
This project focuses on delivering advanced image editing functionalities within the AIGC domain, specifically designed for tasks requiring precision and user-friendly interaction. A key feature of this system is its ability to **dynamically generate intuitive and user-friendly interfaces** tailored to each editing task. This system supports **localized recoloring** along with adjustments to **brightness**, **contrast**, **saturation**, and other essential image properties. The following video demonstrates the system's capabilities:

<div style="display: flex; justify-content: center; align-items: center; width: 100%; height: vh;">
  <video style="width: 90%; height: auto;" controls>
    <source src="demo_aigc.mp4" type="video/mp4">
    Your browser does not support the video tag.
  </video>
</div>

## Setup
1. git clone repo.
```
git clone https://github.com/AIWeb3Limited/HCI_UIUX_Agent.git
```
2. create environment.
```
conda env create -f environment.yaml
```
3. download checkpoints. 
```
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
```

## 
