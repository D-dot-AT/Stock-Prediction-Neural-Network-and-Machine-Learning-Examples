from scipy.stats import fisher_exact

# Level to binarize our output.
PREDICTION_THRESHOLD = 0.5

def print_statistics(tp: int, fp: int, fn: int, tn: int):
    # Step 5: Statistical Analysis
    # Using the counts obtained from Step 4, perform Fisher's exact test to determine the p-value.
    overall_positive_rate = float(fn + tp) / (tp + fp + tn + fn)
    precision = float(tp) / (tp + fp)
    accuracy = float(tp + tn) / (tp + fp + tn + fn)
    p_value = calculate_precision_p_value(tp=tp, fp=fp, fn=fn, tn=tn)

    # Step 6: Output
    # Print the following information:
    print(f'TN: {tn}')
    print(f'TP: {tp}')
    print(f'FN: {fn}')
    print(f'FP: {fp}')
    print(f'Overall positive rate: {overall_positive_rate}')
    print(f'Precision: {precision}')
    print(f'Accuracy: {accuracy}')
    print(f'P-value of precision: {p_value}')


def calculate_precision_p_value(tp: int, fp: int, fn: int, tn: int):
    # if the prediction precision is lower than the overall rate, return 1.0
    if (tp / (tp + fp)) < ((fn + tp) / (tp + fp + tn + fn)):
        return 1.0
    contingency_table = [[tp, tp + fp], [fn + tp, tp + fp + tn + fn]]
    _, p_value = fisher_exact(contingency_table)
    return p_value