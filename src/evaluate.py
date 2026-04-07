"""
Evaluation script for trained T5 models.
Follows author's evaluation approach for multi-task T5 models.
"""

import argparse
from pathlib import Path
import os

import numpy as np
import pandas as pd
import torch
import unicodedata
import ast
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from data_loader import load_dataset_by_name


def map_data_vihsd(set_df):
    map_labels = {
        0: "CLEAN",
        1: "OFFENSIVE",
        2: "HATE",
    }
    set_df["source"] = set_df["free_text"].apply(lambda x: "hate-speech-detection: " + str(x))
    set_df["target"] = set_df["label_id"].apply(lambda x: map_labels[int(x)])
    set_df = set_df[["source", "target"]]
    return set_df


def map_data_victsd(set_df):
    map_labels = {
        0: "NONE",
        1: "TOXIC",
    }
    set_df["source"] = set_df["Comment"].apply(lambda x: "toxic-speech-detection: " + str(x))
    set_df["target"] = set_df["Toxicity"].apply(lambda x: map_labels[int(x)])
    set_df = set_df[["source", "target"]]
    return set_df


def process_spans(lst):
    if lst == '[]' or pd.isna(lst) or lst == '':
        return []
    lst = [int(x) for x in str(lst).strip("[]'").split(', ') if x.strip()]
    if not lst:
        return []
    result = []
    temp = [lst[0]]
    for i in range(1, len(lst)):
        if lst[i] == lst[i-1] + 1:
            temp.append(lst[i])
        else:
            result.append(temp)
            temp = [lst[i]]
    result.append(temp)
    return result


def add_tags(text, indices):
    if indices == '[]' or pd.isna(indices) or indices == '':
        return text
    indices = process_spans(indices)
    if not indices:
        return text
    for i in range(len(indices)):
        text = text[:indices[i][0]] + "[HATE]" + text[indices[i][0]:]
        text = text[:indices[i][-1]+7] + "[HATE]" + text[indices[i][-1]+7:]
        for j in range(i + 1, len(indices)):
            indices[j] = [x + 12 for x in indices[j]]
    return text


def map_data_vihos(set_df):
    set_df["source"] = set_df["content"].apply(lambda x: "hate-spans-detection: " + str(x))
    target = []
    for i in range(len(set_df['content'])):
        target.append(add_tags(str(set_df['content'].iloc[i]), set_df['index_spans'].iloc[i]))
    set_df["target"] = target
    set_df = set_df[["content", "source", "target", "index_spans"]]
    return set_df


def generate_output_batch(test_df, model, tokenizer, batch_size=32):
    """Generate outputs for test dataframe (optimized with batch processing)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    output_texts = []
    
    with torch.no_grad():
        for i in tqdm(range(0, len(test_df), batch_size), desc="Generating outputs"):
            batch_sources = test_df['source'].iloc[i:i+batch_size].tolist()
            
            # Tokenize batch
            enc = tokenizer(
                batch_sources,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=256
            ).to(device)
            
            # Generate outputs for batch (following author's approach: max_length=256)
            output_ids = model.generate(**enc, max_length=256)
            
            # Decode batch
            decoded = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            output_texts.extend(decoded)

    # Add the generated outputs to the DataFrame
    test_df['output'] = output_texts

    return test_df


def find_and_extract_substrings(original_str, input_str):
    """Extract character indices from generated text with [HATE] ... [HATE] tags (following author's approach)."""
    start_tag = '[hate]'
    end_tag = '[hate]'

    input_str = unicodedata.normalize('NFC', input_str.lower())
    original_str = unicodedata.normalize('NFC', original_str.lower())

    # Extract substrings
    substrings = []
    start_index = input_str.find(start_tag)
    while start_index != -1:
        end_index = input_str.find(end_tag, start_index + len(start_tag))
        if end_index != -1:
            substrings.append(input_str[start_index + len(start_tag):end_index])
            start_index = input_str.find(start_tag, end_index + len(end_tag))
        else:
            break

    if not substrings:
        return '[]'

    # Find indices in the original string and merge into one list
    indices_list = []
    for substring in substrings:
        start_index = original_str.find(substring)
        while start_index != -1:
            indices_list.extend(list(range(start_index, start_index + len(substring))))
            start_index = original_str.find(substring, start_index + 1)

    deduplicated_sorted_indices_list = sorted(set(indices_list))

    return str(deduplicated_sorted_indices_list)


def process_output_spans(lst_output, text_input):
    """Process output spans for ViHOS (following author's approach)."""
    processed_lst = []
    for i in range(len(lst_output)):
        processed_lst.append(find_and_extract_substrings(text_input[i], lst_output[i]))
    return processed_lst


def digitize_spans(vihos_test_df, vihos_results):
    """Convert predicted span strings to binary vectors (following author's approach)."""
    # Convert string representations of lists to actual lists using ast.literal_eval
    vihos_preds = [ast.literal_eval(x) for x in vihos_results['output_spans']]
    vihos_labels = [ast.literal_eval(x) for x in vihos_test_df['index_spans']]

    # Calculate the lengths of the content in vihos_test_df
    vihos_lengths = [len(x) for x in vihos_test_df['content']]

    preds = []
    labels = []

    for i in range(len(vihos_lengths)):
        pred = []
        label = []
        for idx in range(vihos_lengths[i]):
            if idx in vihos_preds[i]:
                pred.append(1)
            else:
                pred.append(0)

            if idx in vihos_labels[i]:
                label.append(1)
            else:
                label.append(0)

        preds.append(pred)
        labels.append(label)

    return labels, preds


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate fine-tuned T5 model')
    parser.add_argument('--model_id', type=str, required=True, help='Repo_ID of fine-tuned model on HuggingFace or local path')
    parser.add_argument('--result_filepath', type=str, default='results/evaluation_results.csv', help='Path for saving the results')
    parser.add_argument('--batch_size', type=int, default=1000, help='Batch size for generation (default: 32)')
    
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 80)
    print("Evaluation Configuration:")
    print("=" * 80)
    print(f"  Model ID: {args.model_id}")
    print(f"  Result file: {args.result_filepath}")
    print("=" * 80)

    # Load fine-tuned models from HuggingFace or local path
    print(f'\nLoading {args.model_id}...')
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    
    # Try loading PyTorch weights first; fall back to Flax if that fails.
    try:
        print("  Attempting to load PyTorch weights...")
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model_id)
        print("  Loaded PyTorch weights successfully.")
    except Exception as e:
        print(f"  PyTorch load failed: {e}")
        print("  Attempting to load from Flax weights (fallback)...")
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model_id, from_flax=True)
    
    model.eval()
    print("  Model loaded successfully!")

    # Load datasets using data_loader
    print('\nPROGRESS|Loading datasets...')
    vihsd_train_df, vihsd_val_df, vihsd_test_df, _ = load_dataset_by_name('ViHSD')
    victsd_train_df, victsd_val_df, victsd_test_df, _ = load_dataset_by_name('ViCTSD')
    vihos_train_df, vihos_val_df, vihos_test_df, _ = load_dataset_by_name('ViHOS')

    # Map datasets to source-target format (following author's approach)
    vihsd_test_df = map_data_vihsd(vihsd_test_df.copy())
    victsd_test_df = map_data_victsd(victsd_test_df.copy())
    vihos_test_df = map_data_vihos(vihos_test_df.copy())

    # Keep original vihos_test_df for span extraction
    vihos_test_df_orig = vihos_test_df.copy()

    # Inferencing (following author's approach)
    print('\nPROGRESS|Generating predictions...')
    vihsd_results = generate_output_batch(vihsd_test_df.copy(), model, tokenizer, batch_size=args.batch_size)
    victsd_results = generate_output_batch(victsd_test_df.copy(), model, tokenizer, batch_size=args.batch_size)
    vihos_results = generate_output_batch(vihos_test_df.copy(), model, tokenizer, batch_size=args.batch_size)
    
    # Process ViHOS spans
    vihos_results['output_spans'] = process_output_spans(vihos_results['output'], vihos_test_df_orig['content'])

    # ViHSD: Hate Speech Detection
    # Turn outputs into labels, 'CLEAN', 'OFFENSIVE', 'HATE' to 0, 1, 2
    vihsd_results['output'] = vihsd_results['output'].apply(
        lambda x: 0 if x == 'CLEAN' or x == 'clean' else (1 if x == 'OFFENSIVE' or x == 'offensive' else 2)
    )
    vihsd_test_df['target'] = vihsd_test_df['target'].apply(
        lambda x: 0 if x == 'CLEAN' or x == 'clean' else (1 if x == 'OFFENSIVE' or x == 'offensive' else 2)
    )

    # Compute accuracy, weighted f1 score, and macro f1 score
    vihsd_accuracy = accuracy_score(vihsd_results['output'], vihsd_test_df['target'])
    vihsd_f1_weighted = f1_score(vihsd_results['output'], vihsd_test_df['target'], average='weighted')
    vihsd_f1_macro = f1_score(vihsd_results['output'], vihsd_test_df['target'], average='macro')

    # ViCTSD: Toxic Speech Detection
    # Turn outputs into labels, 'NONE', 'TOXIC' to 0, 1
    victsd_results['output'] = victsd_results['output'].apply(lambda x: 0 if x == 'NONE' or x == 'none' else 1)
    victsd_test_df['target'] = victsd_test_df['target'].apply(lambda x: 0 if x == 'NONE' or x == 'none' else 1)

    # Compute accuracy, weighted f1 score, and macro f1 score
    victsd_accuracy = accuracy_score(victsd_results['output'], victsd_test_df['target'])
    victsd_f1_weighted = f1_score(victsd_results['output'], victsd_test_df['target'], average='weighted')
    victsd_f1_macro = f1_score(victsd_results['output'], victsd_test_df['target'], average='macro')

    # ViHOS: Hate Spans Detection
    vihos_labels_digits, vihos_preds_digits = digitize_spans(vihos_test_df_orig, vihos_results)

    # Check for length mismatches
    mismatches = []
    for i in range(len(vihos_labels_digits)):
        if len(vihos_labels_digits[i]) != len(vihos_preds_digits[i]):
            mismatches.append(i)
    
    if mismatches:
        print(f"  ⚠️  Found {len(mismatches)} length mismatches in ViHOS predictions")

    # Initialize lists for storing accuracy, weighted f1 score, and macro f1 score
    vihos_accuracy = []
    vihos_f1_weighted = []
    vihos_f1_macro = []

    # Compute accuracy, weighted f1 score, and macro f1 score for vihos_labels_digits and vihos_preds_digits
    for i in range(len(vihos_labels_digits)):
        preds_temp = list(vihos_preds_digits[i])
        labels_temp = list(vihos_labels_digits[i])
        
        # Skip if lengths don't match
        if len(preds_temp) != len(labels_temp):
            continue

        vihos_accuracy.append(accuracy_score(labels_temp, preds_temp))
        vihos_f1_weighted.append(f1_score(labels_temp, preds_temp, average='weighted', zero_division=0))
        vihos_f1_macro.append(f1_score(labels_temp, preds_temp, average='macro', zero_division=0))

    # Compute the average accuracy, weighted f1 score, and macro f1 score
    vihos_accuracy = np.mean(vihos_accuracy) if vihos_accuracy else 0.0
    vihos_f1_weighted = np.mean(vihos_f1_weighted) if vihos_f1_weighted else 0.0
    vihos_f1_macro = np.mean(vihos_f1_macro) if vihos_f1_macro else 0.0

    # Print the results in tabular format
    print('\n' + "=" * 80)
    print('Results:')
    print("=" * 80)
    print('ViHSD:')
    print(f'  Accuracy: {vihsd_accuracy:.4f}')
    print(f'  Weighted F1 Score: {vihsd_f1_weighted:.4f}')
    print(f'  Macro F1 Score: {vihsd_f1_macro:.4f}')
    print('ViCTSD:')
    print(f'  Accuracy: {victsd_accuracy:.4f}')
    print(f'  Weighted F1 Score: {victsd_f1_weighted:.4f}')
    print(f'  Macro F1 Score: {victsd_f1_macro:.4f}')
    print('ViHOS:')
    print(f'  Accuracy: {vihos_accuracy:.4f}')
    print(f'  Weighted F1 Score: {vihos_f1_weighted:.4f}')
    print(f'  Macro F1 Score: {vihos_f1_macro:.4f}')
    print("=" * 80)

    # Save the results to a CSV file with 4 decimal places
    results = {
        'Model': args.model_id,
        'Task': ['ViHSD', 'ViCTSD', 'ViHOS'],
        'Accuracy': [vihsd_accuracy, victsd_accuracy, vihos_accuracy],
        'Weighted F1 Score': [vihsd_f1_weighted, victsd_f1_weighted, vihos_f1_weighted],
        'Macro F1 Score': [vihsd_f1_macro, victsd_f1_macro, vihos_f1_macro]
    }

    # Round values to 4 decimal places
    for key in results:
        if key != 'Model' and key != 'Task':
            results[key] = [round(x, 4) for x in results[key]]

    results_df = pd.DataFrame(results)

    # Create output directory if needed
    output_file = Path(args.result_filepath)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Append the results to the existing results.csv file (following author's approach)
    results_df.to_csv(output_file, mode='a', header=not output_file.exists(), index=False)
    print(f'\n💾 Saved results to {output_file}')

    print('\nDone!')


if __name__ == "__main__":
    main()
