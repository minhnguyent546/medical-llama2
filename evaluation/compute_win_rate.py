import argparse
import json
import random

from eval_with_gpt4 import Verdict


def main(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    model_first_results = []
    ref_first_results = []
    with open(args.model_first_result_file, 'r') as f:
        model_first_results = json.load(f)
    with open(args.ref_first_result_file, 'r') as f:
        ref_first_results = json.load(f)

    assert len(model_first_results) == len(ref_first_results), (
        'Expected number of records are the same, '
        f'found {len(model_first_results)} and {len(ref_first_results)}'
    )
    for idx in range(len(model_first_results)):
        assert model_first_results[idx]['model_output_id'] == ref_first_results[idx]['model_output_id'], (
            f'Expected `model_output_id` are the same, '
            f'found {model_first_results[idx]["model_output_id"]} and {ref_first_results[idx]["model_output_id"]} '
            f'at index {idx}'
        )
    win_cnt, lose_cnt, tie_cnt, err_cnt, consistency_cnt = 0, 0, 0, 0, 0
    for model_first_result, ref_first_result in zip(model_first_results, ref_first_results):
        if (
            model_first_result['final_verdict'] == Verdict.UNDETERMINED.value or
            ref_first_result['final_verdict'] == Verdict.UNDETERMINED.value
        ):
            # there is an error in one of the run
            err_cnt += 1
            continue

        if model_first_result['final_verdict'] == ref_first_result['final_verdict']:
            consistency_cnt += 1

        if (
            model_first_result['final_verdict'] == Verdict.TIE.value and
            ref_first_result['final_verdict'] == Verdict.TIE.value
        ):
            # both are ties
            tie_cnt += 1
            continue

        if (
            model_first_result['final_verdict'] == Verdict.TIE.value or
            ref_first_result['final_verdict'] == Verdict.TIE.value
        ):
            # one of them is a tie (we might want to consider this as a non-tie...)
            tie_cnt += 1
            continue

        if model_first_result['final_verdict'] != ref_first_result['final_verdict']:
            # if the results are inconsistent between runs, we consider it as a tie
            tie_cnt += 1
        else:
            if model_first_result['final_verdict'] == Verdict.WIN.value:
                win_cnt += 1
            else:
                lose_cnt += 1

    total = len(model_first_results)
    win_rate = win_cnt / (win_cnt + lose_cnt + tie_cnt)
    win_rate_tie_as_half = (win_cnt + 0.5 * tie_cnt) / (win_cnt + lose_cnt + tie_cnt)
    win_rate_non_tied = win_cnt / (win_cnt + lose_cnt)
    consistency = consistency_cnt / (total - err_cnt)

    print('***** Win rate results *****')
    print(f'  Total: {total}')
    print(f'  Win count: {win_cnt}')
    print(f'  Lose count: {lose_cnt}')
    print(f'  Tie count: {tie_cnt}')
    print(f'  Error count: {err_cnt}')
    print(f'  Win rate: {win_rate}')
    print(f'  Win rate (non-tied): {win_rate_non_tied}')
    print(f'  Win rate (tie as half score): {win_rate_tie_as_half}')
    print(f'  Consistency: {consistency}')
    print(f'  Error: {err_cnt / total}')

def set_seed(seed: int) -> None:
    random.seed(seed)

def add_opts(parser) -> None:
    parser.add_argument(
        '--seed',
        type=int,
        help='Seed value for random number generators',
        default=998244353,
    )
    parser.add_argument(
        '--model_first_result_file',
        type=str,
        help='Model first evaluation result file',
        required=True,
    )
    parser.add_argument(
        '--ref_first_result_file',
        type=str,
        help='Reference first evaluation result file',
        required=True,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Computing win rate',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    add_opts(parser)
    args = parser.parse_args()
    main(args)
