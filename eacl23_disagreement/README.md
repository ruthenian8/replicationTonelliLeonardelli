# Reproducing EACL'23 "Why Don’t You Do It Right?" experiments

## Requirements
- Python 3.8+
- [MaChAmp](https://github.com/machamp-nlp/machamp): clone the repo and run `pip install -r machamp/requirements.txt`
- `scikit-learn` for scoring (`pip install scikit-learn`)

## Data
Request/download MD-Agreement and MD-Agreement-v2 (taxonomy) from the authors’ repository (see their README and request form). The EACL paper points to the same link.

## Steps
1. **Normalize data**
   ```bash
   python3 prepare_data.py \
     --train_csv /path/to/MD-Agreement/train.csv \
     --dev_csv   /path/to/MD-Agreement/dev.csv \
     --test_csv  /path/to/MD-Agreement/test.csv \
     --taxonomy_path /path/to/MD-Agreement-v2/Category_dataset.tsv \
     --prefer_taxonomy_agr \
     --outdir data/md_agreement
   ```

   The defaults expect the original MD-Agreement columns (`ID`, `Text`,
   `Agreement_level`, `Offensive_binary_label`, and `Individual_Annotations`).
   Pass `--id_col`, `--text_col`, `--gold_col`, `--agr_col`, `--ann_cols`, or
   `--ann_field` to adapt the script to alternative schemas.

2. **Make training variants (Sec. 5.2)**

   ```bash
   python3 make_splits.py --base_dir data/md_agreement --out_root data/splits
   ```

3. **Run everything (20 restarts each)**

   ```bash
   bash run_experiments.sh
   ```

4. **Outputs**
   - Predictions: `results/*.pred.txt`
   - Table 2 JSONs: `results/table2_by_subtype.json`, `results/table2_by_category.json`
   - Table 3 JSONs: `results/table3_App.json`, `results/table3_App_A0all.json`, `results/table3_App_A0_SUBJ.json`, `results/table3_App_A0_MISS.json`, `results/table3_App_A0_AMB.json`
   - Table 4 JSONs: `results/table4_MTL3.json`, `results/table4_MTL6.json`

The JSONs include mean and standard deviation across 20 runs for `ALL`, `OFF`, and `NOT`. They should align with the reported micro-F1 and deviations in Tables 2–4.
