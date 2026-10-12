"""End-to-end pipeline, equivalent to running notebooks/mammals_full_pipeline.ipynb top to bottom.

    python train.py --work-dir work

Steps: download -> clean -> split -> tf.data -> model -> custom training loop ->
training curves -> logit adjustment + temperature scaling -> test evaluation ->
UMAP + prototype images -> reliability diagram / ECE -> Keras + TFLite export.
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")  # save figures instead of showing them

from src.config import Config, set_seed  # noqa: E402


def parse_args() -> argparse.Namespace:
    cfg = Config()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--work-dir", default=cfg.work_dir, help="where data, figures and exports are written")
    p.add_argument("--epochs", type=int, default=cfg.epochs)
    p.add_argument("--batch-size", type=int, default=cfg.batch_size)
    p.add_argument("--seed", type=int, default=cfg.seed)
    p.add_argument("--skip-download", action="store_true",
                   help="reuse <work-dir>/dataset_raw instead of downloading from Kaggle")
    p.add_argument("--skip-prepare", action="store_true",
                   help="reuse an existing <work-dir>/subset_clean split (skips download + cleaning + split)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = Config(work_dir=args.work_dir, epochs=args.epochs, batch_size=args.batch_size, seed=args.seed)
    os.makedirs(cfg.figure_dir, exist_ok=True)
    set_seed(cfg.seed)

    from src import calibration, data, embedding, evaluation, export, model as model_lib, training

    # 1-3. data preparation
    if not args.skip_prepare:
        if not args.skip_download:
            data.download_dataset(cfg.raw_dir, cfg.classes)
        data.clean_dataset(cfg.raw_dir, cfg.clean_dir, cfg.classes,
                           cfg.min_side, cfg.blur_thr, cfg.phash_dup_thr)
        data.stratified_split(cfg.clean_dir, cfg.split_dir, cfg.classes, cfg.seed,
                              cfg.test_size, cfg.val_size)

    # 4. tf.data pipeline
    train_ds, val_ds, test_ds, class_names = data.make_datasets(
        cfg.split_dir, cfg.img_size, cfg.batch_size, cfg.seed)
    k = len(class_names)
    print("Classes:", class_names)

    # 5. model + prototypes
    model = model_lib.build_model(k, cfg.img_size, cfg.embed_dim)
    prototypes = model_lib.make_prototypes(k, cfg.embed_dim)

    # 6. custom training loop
    history = training.train(model, prototypes, train_ds, val_ds, epochs=cfg.epochs,
                             lambda_center=cfg.lambda_center, proto_momentum=cfg.proto_momentum,
                             learning_rate=cfg.learning_rate)
    training.plot_training_curves(history, os.path.join(cfg.figure_dir, "training_curves.png"))

    # 7. logit adjustment + temperature scaling (fitted on validation)
    train_dir = os.path.join(cfg.split_dir, "train")
    adj = calibration.logit_adjustment_from_dir(train_dir, class_names)
    val_logits, val_labels = calibration.collect_logits(model, val_ds, adj)
    best_t = calibration.fit_temperature(val_logits, val_labels)
    print("Best T =", best_t)

    # 8. test evaluation (raw logits)
    res = evaluation.evaluate_test(model, test_ds, class_names)
    evaluation.plot_confusion_matrix(res["confusion_matrix"], class_names,
                                     os.path.join(cfg.figure_dir, "confusion_matrix.png"))

    # 9. embedding analysis
    embs, lbls, paths, _ = embedding.extract_embeddings(model, train_dir, cfg.img_size)
    embedding.plot_umap(embs, lbls, class_names, os.path.join(cfg.figure_dir, "umap_train_embeddings.png"))
    proto_imgs = embedding.nearest_to_prototype(embs, lbls, paths, prototypes, class_names)
    embedding.plot_prototype_images(proto_imgs, os.path.join(cfg.figure_dir, "prototype_images.png"))

    # 10. calibration quality on the test split
    rel = evaluation.reliability(model, test_ds, adj, best_t)
    evaluation.plot_reliability_diagram(rel["bin_acc"], rel["bin_avg"],
                                        os.path.join(cfg.figure_dir, "reliability_diagram.png"))

    # 11. export
    export.export_artifacts(model, class_names, cfg.export_dir, adj, best_t)

    summary = {"history": history, "best_temperature": best_t,
               "test_accuracy": float(res["accuracy"]), "ece": rel["ece"]}
    with open(os.path.join(cfg.work_dir, "metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("Saved", os.path.join(cfg.work_dir, "metrics.json"))


if __name__ == "__main__":
    main()
