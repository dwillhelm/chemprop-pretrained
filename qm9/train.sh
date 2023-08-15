chemprop_train \
--data_path ../datasets/processed/qm9-chemprop.csv \
--dataset_type regression \
--save_dir ./logs \
--smiles_column "smiles_relaxed" \
--target_columns "gap" \
# --split_type cv-no-test \
--no_cache_mol \
--epochs 1 