# FlashBias for Neural Solver

## The code of this part can be found [here](https://github.com/thuml/Neural-Solver-Library/blob/main/models/Transformer_Spatial_Bias.py)

FlashBias can be used to accelerate the computation of the [Neural PDE Solver](https://github.com/thuml/Neural-Solver-Library), which encodes relative position information for better geometry learning.

<u>**Implementation**:</u> Relative position can be exactly decomposed as two basis functions:

<p align="center">
<img src="..\pic\exact_decompose_pde.png" height = "90" alt="" align=center />
<br><br>
<b>Figure 1.</b> Exact decomposition of spatial distance bias.
</p>

## Get Started

Our experiments are based on the car design task from [Neural-Solver-Library](https://github.com/thuml/Neural-Solver-Library).

<p align="center">
<img src="..\pic\pde_task.png" height = "200" alt="" align=center />
<br><br>
<b>Figure 2.</b> Car design task that requires the model to estimate the surrounding wind speed and surface pressure for a driving car.
</p>

1. Check the `Transformer_Spatial_Bias.py` ([here](https://github.com/thuml/Neural-Solver-Library/blob/main/models/Transformer_Spatial_Bias.py)) for the code.
2. Execute the following scripts

```bash
bash scripts/DesignBench/car/Transformer_Spatial_Bias.sh
```

## Efficiency Comparison

FlashBias is the only approach that enables training on the car design task (with 32,186 mesh points per sample).

| 32,186 mesh points       | Training GPU Usage | Training Running Time | Infer GPU Usage | Infer Running Time |
| ------------------------ | ------------------ | --------------------- | --------------- | ------------------ |
| FlashAttention with bias | OOM                | OOM                   | OOM             | OOM                |
| FlashBias                | 2.97 GB            | 51.1 s/iter           | 1.13 GB         | 12.7 s/iter        |

## Citation

If you find this repo useful, please cite our paper. 

```
@inproceedings{wu2025flashbias,
  title={FlashBias: Fast Computation of Attention with Bias},
  author={Haixu Wu and Minghao Guo and Yuezhou Ma and Yuanxu Sun and Jianmin Wang and Wojciech Matusik and Mingsheng Long},
  booktitle={Advances in Neural Information Processing Systems},
  year={2025}
}
```

## Contact

If you have any questions or want to use the code, please contact wuhaixu98@gmail.com
