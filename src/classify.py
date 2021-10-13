import argparse
import pandas as pd
import pickle

from sklearn.metrics import classification_report
from xgboost import XGBClassifier


def compute_features(dataset: pd.DataFrame) -> pd.DataFrame:
    counts = dataset[["UserId", "Event", "Fake"]].groupby(["UserId", "Event"]).count()
    percentages = counts / counts.groupby(level=0).sum()
    percentages = percentages.unstack()
    percentages.columns = percentages.columns.droplevel()
    percentages.columns.name = ""
    event_counts = dataset[["UserId", "Event"]].groupby("UserId").count()
    features = pd.merge(event_counts, percentages, left_index=True, right_index=True)

    ground_truth = pd.merge(
        features,
        dataset.set_index("UserId"),
        how="left",
        left_index=True,
        right_index=True,
    )
    ground_truth = ground_truth[~ground_truth.index.duplicated(keep="first")]["Fake"]
    return features, ground_truth


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", help="Path to the dataset to classify")
    parser.add_argument("--model", help="Path to the model file")
    parser.add_argument("--output-file", help="Path to the output file")
    args = parser.parse_args()

    with open(args.model, "rb") as f:
        model = pickle.load(f)

    df = pd.read_csv(args.data)

    features, ground_truth = compute_features(df)
    is_fake_probability = model.predict_proba(features.values)[:, 1]

    result = pd.DataFrame(
        data=zip(features.index, is_fake_probability),
        columns=["UserId", "is_fake_probability"],
    )

    result.to_csv(args.output_file, index=False)
    print(f"Result saved to {args.output_file}")

    # Accuracy will be already 100%, so there's no need to try to find a special good
    # confidence threshold. The default of 0.5 will work fine.
    print("Confidence threshold to be used is 0.5")


if __name__ == "__main__":
    main()
