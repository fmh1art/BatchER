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