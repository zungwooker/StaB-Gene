## StaB-Gene: Detecting Bias in Text-to-Image Generative Models

Based on [BiasEdit: A Training-Free Bias-Detect-and-Edit Framework for Learning Fair Visual Classifiers](https://dl.acm.org/doi/10.1145/3774904.3792411)

> Jungwook Seo, Yoonsik Park, Changmin Lee, Sungyong Baik

[![Paper](https://img.shields.io/badge/Paper-WWW_2026-blue)](https://dl.acm.org/doi/10.1145/3774904.3792411)

### Overview

BiasEdit-Gene extends the bias detection methodology of BiasEdit to analyze social biases embedded in text-to-image (T2I) generative models. Instead of operating on existing benchmark datasets, this framework generates images from text prompts using state-of-the-art T2I models and applies the statistical dependence and mutual information analysis from BiasEdit to reveal what spurious visual attributes each prompt class tends to produce.

```
[Text Prompts] → [T2I Image Generation] → [Attribute Extraction] → [Bias Detection]
```

### Usage

Initialize submodules, configure the output path in the shell scripts, then run each stage:

```bash
git submodule update --init --recursive

# Step 1: Generate images (run from generate/)
cd generate
bash scripts/generate_age.sh

# Step 2: Extract visual attributes (run from StaB/)
cd ../StaB
bash scripts/extract_attrs.sh

# Step 3: Detect bias
bash scripts/detect_bias.sh
```

### Prompts

Prompts are managed separately in `prompts/`. Each YAML file defines a dataset name and one prompt per class. Example (`prompts/age.yaml`):

```yaml
dataset: age
classes:
  young: "a portrait photo of a young person aged 10 to 29, natural lighting, photorealistic"
  old: "a portrait photo of a middle-aged person aged 40 to 59, natural lighting, photorealistic"
```

### Models

T2I models are downloaded from HuggingFace into `ckpts/` on first run. Select a model with `--model`:

| Model | `--model` | Size | HuggingFace ID |
|---|---|---|---|
| FLUX.1-dev | `flux` | 12B | `black-forest-labs/FLUX.1-dev` |
| Stable Diffusion 3.5 Large | `sd3` | 8B | `stabilityai/stable-diffusion-3.5-large` |

### Output

Generated images are saved at 256×256 resolution under `<output_path>/<dataset>/<class>/`. A `metadata.csv` is created automatically and consumed by the downstream StaB pipeline. Bias detection results are written to `<output_path>/<dataset>/bias_results/`.

### Acknowledgement

BiasEdit-Gene uses [Tag2Text](https://github.com/xinyu1205/recognize-anything),
[LLaVA v1.6](https://huggingface.co/llava-hf/llava-v1.6-vicuna-13b-hf),
[FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev), and
[Stable Diffusion 3.5 Large](https://huggingface.co/stabilityai/stable-diffusion-3.5-large).
Tag2Text is included as a submodule.

The bias detection methodology is based on
[BiasEdit](https://dl.acm.org/doi/10.1145/3774904.3792411), which builds on
[LfF](https://github.com/alinlab/LfF),
[DisEnt](https://github.com/kakaoenterprise/Learning-Debiased-Disentangled), and
[BiasEnsemble](https://github.com/kakaoenterprise/BiasEnsemble).
