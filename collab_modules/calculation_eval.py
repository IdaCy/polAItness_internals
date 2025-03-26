# File: collab_modules/calculation_evaluator.py
import os
import re
import torch
import logging
from glob import glob

def parse_prediction_text(text, allowed_operations):
    """
    Extracts from a prediction string:
      - The first two numbers found.
      - The first occurrence of an allowed operation.
      - The last number in the text (predicted answer).
      - Computes the correct result and determines if the predicted answer is correct.
    Returns a dictionary with the extracted pieces.
    """
    numbers_found = re.findall(r"\d+", text)
    ops_found = re.findall(r"[+\-*/÷]", text)

    result_info = {
        "first_num": None,
        "operation": None,
        "second_num": None,
        "predicted_answer": None,
        "computed_answer": None,
        "correct": False,
    }

    if len(numbers_found) < 2:
        return result_info  # Not enough numbers to perform a calculation

    # First two numbers
    first_num_str = numbers_found[0]
    second_num_str = numbers_found[1]

    # Find the first allowed operation symbol
    operation = None
    for op_candidate in ops_found:
        if op_candidate in allowed_operations:
            operation = op_candidate
            break

    predicted_answer_str = numbers_found[-1]

    # Store initial extracted info
    result_info["first_num"] = first_num_str
    result_info["second_num"] = second_num_str
    result_info["operation"] = operation
    result_info["predicted_answer"] = predicted_answer_str

    # Compute the result if an operation is valid
    if operation is not None:
        try:
            num1 = float(first_num_str)
            num2 = float(second_num_str)

            if operation == "+":
                computed_val = num1 + num2
            elif operation == "-":
                computed_val = num1 - num2
            elif operation in ["*", "×"]:
                computed_val = num1 * num2
            elif operation in ["÷", "/"]:
                if abs(num2) < 1e-12:  # Avoid division by zero
                    return result_info
                computed_val = num1 / num2
            else:
                return result_info

            # Convert computed result to string (using no decimals if integer)
            if computed_val.is_integer():
                computed_str = str(int(computed_val))
            else:
                computed_str = f"{computed_val:.4f}"  # keeping four decimals

            result_info["computed_answer"] = computed_str

            # Compare predicted answer to computed value (with a tolerance for floats)
            if predicted_answer_str == computed_str:
                result_info["correct"] = True
            else:
                try:
                    pred_val = float(predicted_answer_str)
                    if abs(pred_val - computed_val) < 1e-4:
                        result_info["correct"] = True
                except ValueError:
                    pass

        except Exception:
            pass

    return result_info

def evaluate_predictions(pt_file="output/extractions/gemma2bit/normal",
                         log_file="logs/extraction_log.txt",
                         allowed_operations="+-*÷",
                         results_file="logs/calculation_results.txt"):
    """
    Evaluates the predictions stored in .pt files (or a single .pt file) by checking if the
    predicted answer (last number in the text) correctly corresponds to the calculation
    from the first two numbers and the allowed operation symbol.
    
    All logging is directed to the log file and a results file is also written.
    The function returns the overall percentage of correct predictions (as a float).

    Parameters:
      - pt_file: A directory containing .pt files or a single .pt file path.
      - log_file: Path to the log file.
      - allowed_operations: A string of allowed operation symbols.
      - results_file: Path to the results summary file.

    Returns:
      - overall_accuracy (float): Percentage of correct calculations.
    """
    # Ensure directories for log and results files exist
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    os.makedirs(os.path.dirname(results_file), exist_ok=True)

    # Set up logging (only to file, so cell output remains clean)
    logger = logging.getLogger("PTExtractionLogger")
    logger.setLevel(logging.DEBUG)
    # Clear any previous handlers
    if logger.hasHandlers():
        logger.handlers.clear()
    file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    logger.info("=== Starting extraction evaluation ===")
    logger.info(f"PT_FILE (dir or file) = {pt_file}")
    logger.info(f"LOG_FILE = {log_file}")
    logger.info(f"ALLOWED_OPERATIONS = {allowed_operations}")
    logger.info(f"RESULTS_FILE = {results_file}")

    # Prepare results file for a summary
    with open(results_file, "w", encoding="utf-8") as summary_f:
        # Determine if pt_file is a directory or a single file
        if os.path.isdir(pt_file):
            pt_paths = sorted(glob(os.path.join(pt_file, "*.pt")))
        else:
            pt_paths = [pt_file]

        logger.info(f"Found {len(pt_paths)} .pt file(s) to process.")

        overall_total = 0
        overall_correct = 0

        for pt_path in pt_paths:
            logger.info("--------------------------------------------------")
            logger.info(f"Processing file: {pt_path}")
            logger.info("------")

            try:
                data = torch.load(pt_path)
            except Exception as e:
                logger.error(f"Failed to load {pt_path}: {str(e)}")
                continue

            predictions = data.get("final_predictions", [])
            logger.info(f"Number of predictions in this file: {len(predictions)}")

            file_correct = 0
            file_total = 0

            for idx, pred_text in enumerate(predictions):
                parsed = parse_prediction_text(pred_text, allowed_operations)
                logger.debug(
                    f"[File={os.path.basename(pt_path)} | Idx={idx}] "
                    f"Extracted -> first_num={parsed['first_num']}, "
                    f"operation={parsed['operation']}, "
                    f"second_num={parsed['second_num']}, "
                    f"predicted_answer={parsed['predicted_answer']}, "
                    f"computed_answer={parsed['computed_answer']}, "
                    f"correct={parsed['correct']}"
                )

                # Build and log detailed calculation info
                first = parsed['first_num']
                second = parsed['second_num']
                op = parsed['operation']
                pred_ans = parsed['predicted_answer']
                comp_ans = parsed['computed_answer']
                is_correct = parsed['correct']

                if op is not None and first is not None and second is not None:
                    calc_str = f"Calculation: {first} {op} {second} = {pred_ans} -> "
                    if is_correct:
                        calc_str += "TRUE"
                    else:
                        calc_str += f"FALSE, {first} {op} {second} = {comp_ans}"
                    logger.debug(calc_str)
                else:
                    logger.debug("Calculation: Could not parse properly.")

                logger.debug(f"Prediction output:\n{pred_text}\n")

                file_total += 1
                if is_correct:
                    file_correct += 1

            # Summarize per file
            accuracy = (file_correct / file_total * 100) if file_total > 0 else 0
            logger.info(
                f"File summary: correct={file_correct}, total={file_total}, accuracy={accuracy:.2f}%"
            )
            summary_f.write(
                f"File: {pt_path}, correct={file_correct}, total={file_total}, accuracy={accuracy:.2f}%\n"
            )

            overall_total += file_total
            overall_correct += file_correct

        # Final overall summary
        overall_acc = (overall_correct / overall_total * 100) if overall_total > 0 else 0
        logger.info("==================================================")
        logger.info(
            f"Overall correctness across all files: {overall_correct} / {overall_total} ({overall_acc:.2f}%)"
        )
        summary_f.write(
            "==================================================\n"
            f"Overall correctness: {overall_correct}/{overall_total} ({overall_acc:.2f}%)\n"
        )

    # Return only the overall accuracy percentage (so that your notebook cell prints just this)
    return overall_acc
