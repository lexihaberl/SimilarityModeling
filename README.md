
# SimilarityModeling

## Repo structure

- /data/ - evaluation csv dataframes, frames used for the the examples, ground truth.

- /src/ - code.

- /src/utils/ - custom functions for data preparation, feature engineering, training etc.

- /src/SIM_intermediate_hand_in_clean.ipynb - notebook meant to be looked for the intermediate hand-in, short summary of ideas with examples and first results.

- /src/SIM_in_process.ipynb - messy "in progress results" - messy notebook, showing how results showed in intermediate hand in notebook are obtained. Not cleaned up!

## Execution after installation

```bash
conda activate SimilarityModeling
jupyter lab
```

## Adding data

Put the episodes from 'The Muppet Show' into the ```data/videos``` folder.
The ground truth annotations are already part of the repository.
They are found in the ```data/gt_annotations``` folder.

## Creating Conda environment

```bash
conda env create -f conda_env.yml
```
