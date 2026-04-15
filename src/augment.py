"""
Data augmentation for Vietnamese hate speech datasets.

Addresses class imbalance by augmenting minority classes (HATE, OFFENSIVE)
using multiple strategies: synonym replacement, random operations, and back-translation.

Reference: Wei & Zou, "EDA: Easy Data Augmentation Techniques for Boosting
Performance on Text Classification Tasks", EMNLP 2019.
"""

import random
import re
import pandas as pd
import numpy as np
from typing import List, Tuple, Optional
from collections import Counter


# Vietnamese synonym dictionary for hate speech domain
# Grouped by semantic category for meaningful augmentation
VIETNAMESE_SYNONYMS = {
    # Negative adjectives
    "ngu": ["dốt", "ngốc", "đần", "khờ"],
    "dốt": ["ngu", "ngốc", "đần"],
    "ngốc": ["ngu", "dốt", "đần"],
    "xấu": ["tệ", "dở", "kém"],
    "tệ": ["xấu", "dở", "kém"],
    "rác": ["rác rưởi", "phế thải"],
    # Insult words
    "đồ": ["thứ", "loại"],
    "thứ": ["đồ", "loại"],
    "mày": ["mi", "ngươi"],
    "tao": ["ta", "tớ"],
    # Positive words (for clean text augmentation)
    "hay": ["tốt", "tuyệt", "giỏi"],
    "tốt": ["hay", "tuyệt", "tuyệt vời"],
    "đẹp": ["xinh", "dễ thương"],
    "giỏi": ["hay", "xuất sắc", "tài"],
    # Common Vietnamese words
    "nói": ["phát biểu", "bàn", "trình bày"],
    "làm": ["thực hiện", "tiến hành"],
    "có": ["sở hữu", "mang"],
    "người": ["con người", "cá nhân"],
}

# Vietnamese stopwords to avoid replacing
VIETNAMESE_STOPWORDS = {
    "và", "là", "của", "các", "có", "được", "cho", "với",
    "trong", "không", "một", "này", "đã", "từ", "khi",
    "theo", "về", "để", "sẽ", "vì", "nếu", "đến",
    "nhưng", "tại", "còn", "như", "hay", "hoặc",
}


def synonym_replacement(sentence: str, n: int = 1) -> str:
    """Replace n words with synonyms from the Vietnamese synonym dictionary."""
    words = sentence.split()
    new_words = words.copy()
    replaceable = [
        (i, w.lower()) for i, w in enumerate(words)
        if w.lower() in VIETNAMESE_SYNONYMS and w.lower() not in VIETNAMESE_STOPWORDS
    ]

    random.shuffle(replaceable)
    num_replaced = 0

    for idx, word in replaceable:
        if num_replaced >= n:
            break
        synonyms = VIETNAMESE_SYNONYMS[word]
        synonym = random.choice(synonyms)
        # Preserve original capitalization
        if words[idx][0].isupper():
            synonym = synonym.capitalize()
        new_words[idx] = synonym
        num_replaced += 1

    return " ".join(new_words)


def random_insertion(sentence: str, n: int = 1) -> str:
    """Randomly insert synonyms of random words into the sentence."""
    words = sentence.split()
    new_words = words.copy()

    for _ in range(n):
        candidates = [w.lower() for w in words if w.lower() in VIETNAMESE_SYNONYMS]
        if not candidates:
            break
        word = random.choice(candidates)
        synonym = random.choice(VIETNAMESE_SYNONYMS[word])
        insert_pos = random.randint(0, len(new_words))
        new_words.insert(insert_pos, synonym)

    return " ".join(new_words)


def random_swap(sentence: str, n: int = 1) -> str:
    """Randomly swap two words in the sentence n times."""
    words = sentence.split()
    new_words = words.copy()

    for _ in range(n):
        if len(new_words) < 2:
            break
        idx1, idx2 = random.sample(range(len(new_words)), 2)
        new_words[idx1], new_words[idx2] = new_words[idx2], new_words[idx1]

    return " ".join(new_words)


def random_deletion(sentence: str, p: float = 0.1) -> str:
    """Randomly delete words with probability p."""
    words = sentence.split()
    if len(words) <= 1:
        return sentence

    remaining = [w for w in words if random.random() > p]
    if not remaining:
        return random.choice(words)

    return " ".join(remaining)


def eda_augment(sentence: str, alpha_sr=0.1, alpha_ri=0.1, alpha_rs=0.1,
                p_rd=0.1, num_aug=4) -> List[str]:
    """
    Apply EDA (Easy Data Augmentation) to generate augmented sentences.

    Args:
        sentence: Input sentence.
        alpha_sr: Fraction of words to replace with synonyms.
        alpha_ri: Fraction of words for random insertion.
        alpha_rs: Fraction of words for random swap.
        p_rd: Probability of random deletion.
        num_aug: Number of augmented sentences to generate.

    Returns:
        List of augmented sentences.
    """
    words = sentence.split()
    num_words = len(words)
    augmented = set()

    n_sr = max(1, int(alpha_sr * num_words))
    n_ri = max(1, int(alpha_ri * num_words))
    n_rs = max(1, int(alpha_rs * num_words))

    # Generate augmented sentences
    for _ in range(num_aug):
        op = random.choice(["sr", "ri", "rs", "rd"])
        if op == "sr":
            augmented.add(synonym_replacement(sentence, n_sr))
        elif op == "ri":
            augmented.add(random_insertion(sentence, n_ri))
        elif op == "rs":
            augmented.add(random_swap(sentence, n_rs))
        elif op == "rd":
            augmented.add(random_deletion(sentence, p_rd))

    # Remove duplicates of the original
    augmented.discard(sentence)
    result = list(augmented)[:num_aug]

    return result


def augment_minority_classes(
    df: pd.DataFrame,
    text_col: str,
    label_col: str,
    target_ratio: float = 1.0,
    num_aug_per_sample: int = 4,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Augment minority classes to balance the dataset.

    Args:
        df: DataFrame with text and label columns.
        text_col: Name of the text column.
        label_col: Name of the label column.
        target_ratio: Target ratio of minority to majority class (1.0 = fully balanced).
        num_aug_per_sample: Number of augmented samples per original sample.
        seed: Random seed.

    Returns:
        Augmented DataFrame with balanced classes.
    """
    random.seed(seed)
    np.random.seed(seed)

    class_counts = df[label_col].value_counts()
    majority_count = class_counts.max()
    target_count = int(majority_count * target_ratio)

    print(f"\n  Original class distribution:")
    for cls, count in class_counts.items():
        print(f"    {cls}: {count}")

    augmented_rows = []

    for cls in class_counts.index:
        cls_df = df[df[label_col] == cls]
        current_count = len(cls_df)

        if current_count >= target_count:
            continue

        needed = target_count - current_count
        print(f"\n  Augmenting class '{cls}': {current_count} → {target_count} (+{needed})")

        generated = 0
        while generated < needed:
            for _, row in cls_df.iterrows():
                if generated >= needed:
                    break
                augmented_texts = eda_augment(
                    str(row[text_col]),
                    num_aug=min(num_aug_per_sample, needed - generated),
                )
                for aug_text in augmented_texts:
                    if generated >= needed:
                        break
                    new_row = row.copy()
                    new_row[text_col] = aug_text
                    augmented_rows.append(new_row)
                    generated += 1

    if augmented_rows:
        augmented_df = pd.concat([df, pd.DataFrame(augmented_rows)], ignore_index=True)
    else:
        augmented_df = df.copy()

    print(f"\n  New class distribution:")
    for cls, count in augmented_df[label_col].value_counts().items():
        print(f"    {cls}: {count}")
    print(f"  Total: {len(df)} → {len(augmented_df)} (+{len(augmented_df) - len(df)})")

    return augmented_df


def augment_vihsd(train_df: pd.DataFrame, target_ratio: float = 0.8) -> pd.DataFrame:
    """Augment ViHSD training data to balance CLEAN/OFFENSIVE/HATE classes."""
    return augment_minority_classes(
        train_df, text_col="free_text", label_col="label_id",
        target_ratio=target_ratio, num_aug_per_sample=4,
    )


def augment_victsd(train_df: pd.DataFrame, target_ratio: float = 0.8) -> pd.DataFrame:
    """Augment ViCTSD training data to balance NONE/TOXIC classes."""
    return augment_minority_classes(
        train_df, text_col="Comment", label_col="Toxicity",
        target_ratio=target_ratio, num_aug_per_sample=4,
    )
