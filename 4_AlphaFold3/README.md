# FlashBias for AlphaFold 3

FlashBias can be used to accelerate Pairformer in [AlphaFold 3 (Nature)](https://www.nature.com/articles/s41586-024-07487-w), which involves the triangular attention mechanism with pair representation bias.

<u>**Implementation**:</u> Since the bias term of AlphaFold 3 is learned from pair representations, we propose a *neural decomposition version*, which employs lightweight neural networks to approximate the decomposed factor functions.

<p align="center">
<img src="..\pic\neural_decomp.png" height = "40" alt="" align=center />
<br><br>
<b>Figure 1.</b> Neural decomposition uses lightweight neural networks to approximate the bias term.
</p>
## Get Started

This part is modified from [Protenix](https://github.com/bytedance/Protenix).

1. Prepare the environment.

```bash
pip install -r requirements.txt
```

2. [Optional] Finetune the original AlphaFold 3 to get the approximated factor neural network.

```bash
wget -P /data/af3-dev/release_model/ https://af3-dev.tos-cn-beijing.volces.com/release_model/model_v0.2.0.pt # Get original AlphaFold 3 model
bash finetune_demo_flashbias.sh # Finetune Flashbias factor functions.
```

Check ./protenix/openfold_local/model/triangular_attention.py for our modifications.

Alternatively, you can skip this step and use our fine-tuned model (coming soon).

3. Inference with FlashBias

You can find several examples in this fold ./examples

Please check ./configs/configs_inference.py for inference configuration and modify `code_directory` to your finetuned flashbias model path.

```bash
bash inference_demo_flashbias.sh # Inference with ID: 7wux
```

4. Evaluate FlashBias on a dataset.

Download test data with the following command

```bash
bash down_data.sh
```

Execute the following command for a comprehensive evaluation

```bash
bash eval_demo_flashbias.sh # evaluation on recentPDB_1536_sample384_0925
```

## Efficiency Comparison

<p align="center">
<img src="..\pic\alphafold_case.png" height = "150" alt="" align=center />
<br><br>
<b>Figure 2.</b> Visualization and efficiency comparison in protein-folding.
</p>
## Project Guidance

Since [Protenix](https://github.com/bytedance/Protenix) is a large project, we provide some guidance here for a quick start.

To ensure the fine-tuning and inference with FlashBias, we have modified the following files in sequence:

- ./runner/train_flashbias.py (Line 423): Catch and optimize the approximation loss. 
- ./protenix/model/protenix.py (Line 250): Convey initial feature s_inputs to Pairformer to estimate the decomposed factor tensors.
- ./protenix/model/modules/pairformer.py (Line 138, Line 151, Line 258): Convey s_input and the id of layers into the TriangleAttention.
- ./protenix/openfold_local/model/triangular_attention.py (Lines 152-195): Adopt neural networks to generate decomposed factor tensors.
- ./protenix/openfold_local/model/primitives.py (Lines 605-643): Adopt flash_bias_triton.py for speed up.

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


## Acknowledgements

This part is modified from [Protenix](https://github.com/bytedance/Protenix), a PyTorch implementation of AlphaFold 3 released by Bytedance.
