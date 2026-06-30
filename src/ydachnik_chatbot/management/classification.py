import argparse
import random
import unicodedata
from typing import Any

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


def _remove_surrogates(value: str) -> str:
    return "".join(ch for ch in value if not 0xD800 <= ord(ch) <= 0xDFFF)


def _normalize_text(value: str) -> str:
    return _remove_surrogates(unicodedata.normalize("NFC", value)).strip()


def _normalize_embedding(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm


def _build_category_stats(
    df: pd.DataFrame,
    embedding_model: SentenceTransformer,
) -> dict[str, dict[str, Any]]:
    category_stats: dict[str, dict[str, Any]] = {}

    for category in sorted(df["category"].unique()):
        if not category:
            continue

        titles = df.loc[df["category"] == category, "title"].tolist()
        if not titles:
            continue

        embeddings = np.asarray(
            embedding_model.encode(
                titles,
                normalize_embeddings=True,
            )
        )

        centroid = _normalize_embedding(embeddings.mean(axis=0))
        category_stats[category] = {
            "embeddings": embeddings,
            "titles": np.asarray(titles, dtype=object),
            "centroid": centroid,
        }

    return category_stats


def _add_category_example(
    category_stats: dict[str, dict[str, Any]],
    category: str,
    embedding: np.ndarray,
    title: str,
) -> None:
    if category not in category_stats:
        category_stats[category] = {
            "embeddings": np.asarray([embedding]),
            "titles": np.asarray([title], dtype=object),
            "centroid": embedding,
        }
        return

    embeddings = category_stats[category]["embeddings"]
    titles = category_stats[category]["titles"]
    if embeddings.ndim == 1:
        embeddings = embeddings.reshape(1, -1)

    category_stats[category]["embeddings"] = np.vstack([embeddings, embedding])
    category_stats[category]["titles"] = np.append(titles, title)
    category_stats[category]["centroid"] = _normalize_embedding(
        category_stats[category]["embeddings"].mean(axis=0)
    )


def _score_category(
    title_embedding: np.ndarray,
    category_stats: dict[str, Any],
) -> tuple[float, float, float, list[tuple[str, float]]]:
    centroid_score = float(np.dot(title_embedding, category_stats["centroid"]))
    similarities = category_stats["embeddings"] @ title_embedding
    nearest_score = float(np.max(similarities))
    combined_score = 0.4 * centroid_score + 0.6 * nearest_score
    top_indices = np.argsort(similarities)[::-1][:3]
    nearest_neighbors = [
        (str(category_stats["titles"][idx]), float(similarities[idx])) for idx in top_indices
    ]
    return combined_score, centroid_score, nearest_score, nearest_neighbors


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("csv")
    parser.add_argument(
        "--reset-categories",
        action="store_true",
        help="Clear all existing category values before classification.",
    )

    args = parser.parse_args()

    print("Loading CSV...")
    df = pd.read_csv(args.csv, sep=";")
    if args.reset_categories and "category" in df.columns:
        df["category"] = ""

    df["category"] = df["category"].fillna("").astype(str)

    # Count unclassified
    unclassified_mask = df["category"].str.strip() == ""
    print(f"Unclassified items: {unclassified_mask.sum()}/{len(df)}")

    # Build category embeddings
    print("Loading embedding model...")
    embedding_model = SentenceTransformer(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    category_stats = _build_category_stats(df, embedding_model)

    # Predict unclassified products
    unclassified_indices = df.index[unclassified_mask].tolist()
    random.shuffle(unclassified_indices)

    for idx in unclassified_indices:
        row = df.loc[idx]
        title_embedding = embedding_model.encode(
            row["title"],
            normalize_embeddings=True,
        )

        best_category = None
        best_score = -1
        best_centroid_score = -1
        best_nearest_score = -1
        best_nearest_neighbors: list[tuple[str, float]] = []

        for category, stats in category_stats.items():
            score, centroid_score, nearest_score, nearest_neighbors = _score_category(
                title_embedding,
                stats,
            )

            if score > best_score:
                best_score = score
                best_category = category
                best_centroid_score = centroid_score
                best_nearest_score = nearest_score
                best_nearest_neighbors = nearest_neighbors

        if best_score < 0.9:
            continue

        print(f"Title: {row['title']}")
        print(
            f"Suggested: {best_category} ({best_score:.3f}, "
            f"centroid={best_centroid_score:.3f}, nearest={best_nearest_score:.3f})"
        )
        print(
            "\nNearest neighbors: "
            + ", ".join(f"{title} ({score:.3f})" for title, score in best_nearest_neighbors)
        )

        answer = input("\nAccept? [Y/n/q/custom]: ").strip()

        if answer.lower() in ("", "y", "yes"):
            df.loc[idx, "category"] = best_category
            _add_category_example(
                category_stats,
                best_category,
                title_embedding,
                row["title"],
            )
        elif answer.lower() == "n":
            continue
        elif answer.lower() == "q":
            break
        else:
            custom_category = _normalize_text(answer)
            df.loc[idx, "category"] = custom_category
            _add_category_example(
                category_stats,
                custom_category,
                title_embedding,
                row["title"],
            )

    # Final stats
    classified = (df["category"].str.strip() != "").sum()

    print()
    print(f"Classified: {classified}/{len(df)}")
    print(f"Unclassified: {len(df) - classified}/{len(df)}")

    df.to_csv(args.csv, sep=";", index=False, encoding="utf-8")

    print("Done")


if __name__ == "__main__":
    main()
