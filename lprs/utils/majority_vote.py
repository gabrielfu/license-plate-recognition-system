from typing import List, Tuple, Optional


def majority_vote(ocr_results: List[Tuple[str, float]]) -> Tuple[str, Optional[float]]:
    """
    Majority vote on a list of OCR results and output the majority result.

    Args:
        ocr_results (list of tuples): e.g. [('PV1954',0.99),('PV1954',0.97),('PV1934',0.91),...]

    Returns:
        tuple(num, conf) e.g. ('PV1954', 0.99)
    """
    if not ocr_results:  # Empty
        return 'Recognition fail', None

    counter = {}
    license_num_prob = {}
    for license_num, min_conf in ocr_results:
        # Count number of votes
        counter[license_num] = counter.get(license_num, 0) + 1

#             if license_num not in license_num_max_prob:
#                 license_num_max_prob[license_num] = avg_conf
#             elif avg_conf > license_num_max_prob[license_num]:
#                 license_num_max_prob[license_num] = avg_conf
        if license_num not in license_num_prob:
            license_num_prob[license_num] = [min_conf]
        else:
            license_num_prob[license_num].append(min_conf)

    license_num_prob = {num:(sum(scores)/len(scores)) for num, scores in license_num_prob.items()}
    # Unique majority --> output major result
    # Multi/No majority --> output highest avg_conf result
    major_candidates = [lic for lic, count in counter.items() if count == max(counter.values())]
    major_candidates_conf = {lic:license_num_prob[lic] for lic in major_candidates}
    lic_num, conf = max(major_candidates_conf.items(), key=lambda x: x[1])
    return lic_num, conf