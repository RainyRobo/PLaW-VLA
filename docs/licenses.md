# Code, model, and data terms

## Repository code

PLaW-VLA's own code is under [Apache-2.0](../LICENSE). [NOTICE](../NOTICE) records upstream attributions and modified code. openpi, Big Vision, Transformers, DINO/DINOv2-derived positional interpolation, LeRobot-derived writer code, and LingBot-VLA-derived client portions retain their Apache-2.0 notices. Meta V-JEPA 2 and RoboTwin-derived portions retain MIT notices in [LICENSES](../LICENSES). The client serializer contains BSD-3-Clause msgpack-numpy code with its full copyright notice and license.

The `plaw-vla` and `openpi-client` distributions include their applicable licenses and notices. External submodule source, simulation assets, datasets, and downloaded checkpoints are excluded from wheels and source distributions. Installed third-party packages remain under their own licenses.

## Base models and checkpoints

The code license does not license downloaded model weights. π₀.₅ resources are provided by [Physical Intelligence](https://github.com/Physical-Intelligence/openpi), V-JEPA 2 by [Meta](https://huggingface.co/facebook/vjepa2-vitl-fpc64-256), and Gemma/PaliGemma components by Google under the [Gemma Terms of Use](https://ai.google.dev/gemma/terms) and associated use restrictions.

[LICENSE_GEMMA.txt](../LICENSE_GEMMA.txt) preserves the February 21, 2024 terms supplied with the upstream code. Obtain and follow the terms applicable to the particular model version you download; that retained text does not replace a provider's model-specific terms. PLaW-VLA checkpoints are pending release, and their terms will be stated with the release.

## Source datasets

| Source | Terms and provider |
| --- | --- |
| InternData-A1 | CC BY-NC-SA 4.0 and provider access terms on [its dataset page](https://huggingface.co/datasets/InternRobotics/InternData-A1) |
| AgiBotWorld | CC BY-NC-SA 4.0 and provider access terms for [Alpha](https://huggingface.co/datasets/agibot-world/AgiBotWorld-Alpha) / [Beta](https://huggingface.co/datasets/agibot-world/AgiBotWorld-Beta) |
| EgoDex | CC BY-NC-ND data terms described by [Apple](https://github.com/apple-aiml-research/ml-egodex); its upstream code has separate Apple terms |
| LIBERO | MIT notices in [the upstream repository](https://github.com/Lifelong-Robot-Learning/LIBERO); retain notices accompanying downloaded data |
| RoboTwin | MIT code terms in [the upstream repository](https://github.com/RoboTwin-Platform/RoboTwin); simulation assets and collected datasets retain their accompanying terms |

These restrictions remain relevant to converted data and any intended redistribution or downstream use. The converters operate on local files and default to no upload. Converting a dataset does not grant additional rights to its content or to a model trained from it.

## Optional benchmark dependencies

LIBERO and RoboTwin retain their upstream MIT code licenses. LIBERO-Plus is an optional external submodule whose GitHub repository has not declared a code license ([upstream clarification request](https://github.com/sylvestf/LIBERO-plus/issues/68)). The MIT label on its asset dataset does not establish a license for its code. PLaW-VLA's own adapter is covered by this repository's code license; upstream LIBERO-Plus source and resources are not included in Python distributions. Obtain any needed upstream permission before redistribution.
