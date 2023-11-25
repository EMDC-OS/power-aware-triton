# Know Your Enemy To Save Cloud Energy: Energy-Performance Characterization of Machine Learning Serving

This is the artifact repository for the HPCA'23 paper "Know Your Enemy To Save Cloud Energy: Energy-Performance Characterization of Machine Learning Serving". The prototype and the simulation code are available.

## Contents
![](./note/Contents.png)
### Prototype Server
The prototype inference server is based on the Triton Inference Server ([TIS](https://github.com/triton-inference-server/server)). The source code is available [here](https://github.com/EMDC-OS/power-aware-triton/tree/main/src).

### Cloud-scale Simulation 
The simulation was performed based on the values obtained from the prototype to observe the effectiveness of the proposed schemes in the cloud-scale environment.
The source code is available [here](https://github.com/EMDC-OS/power-aware-triton/tree/main/simulation).

## Citation
```
@INPROCEEDINGS{yu2023know,
  author={Yu, Junyeol and Kim, Jongseok and Seo, Euiseong},
  title={Know Your Enemy To Save Cloud Energy: Energy-Performance Characterization of Machine Learning Serving},
  booktitle={2023 IEEE International Symposium on High-Performance Computer Architecture (HPCA)},
  year={2023},
  pages={842-854},
  doi={10.1109/HPCA56546.2023.10070943}}
```
## Contact
If you have any questions, please contact junyeol.yu@skku.edu