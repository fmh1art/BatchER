# BatchER

## Quick Start

### Step 1: Environment Set up

Before install packages, use conda to create an enveriment `batcher` with `python==3.9.13`:

```bash
conda create -name batcher python==3.9.13
```

Then, install necessary packages with `requirements.txt`

```bash
pip install -r requirements.txt
```

### Step 2: Run

Our code is designed for one-click execution. But before execution, you should: 

1. go to the project path: `cd ./batcher/`
2. fill the `keys.txt` with OPENAI API keys. To facilitate user testing, we provide three keys for free, seen in `keys.txt`.

**(1) Run the code to compare Batch Prompting and Standard Prompting**

```bash
python bp_vs_sp.py
```

**(2) Run BatchER**

You can run BatchER in two modes:

- `run_all`: reproduce the main results in our paper

```bash
python batcher.py --run_all
```

- run specific variants of BatchER, for example, if you want to run `diverse_batch + covering_based` on dataset `WA`:

```bash
python batcher.py --dataset_name "em-wa" --pkg_type "diverse" --demo_selection "covering"
```

### Results

After that, all results will be saved in `/output/`. A complete structure of results:

```sh
|-- output
    |-- feature_extractor # main results
        |-- StructureAware-ratio # structure aware feature extractor based on Levenshtein Ratio
            |-- em-wa # datasets
            ..
        |-- StructureAware-jaro_winkler # structure aware feature extractor based on jw similarity
            |-- em-wa
        |-- SemanticAware-SBERT # semantic aware feature extractor based on SBERT embedding
            |-- em-wa
    |-- BPvsSP # compare batchprompting and standard prompting
        |-- em-wa # datasets
            |-- trial1 # one trial
                |-- BatchPrompting # results of BatchPrompting
                |-- StandardPrompting # results of StandardPrompting
            |-- trial2
            ...
        ...
```

We also provide the experimental log of BatchER, seen in dir `output/__batcher_results_in_paper`
