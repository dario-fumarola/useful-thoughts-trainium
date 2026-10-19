"""Prospective paired endpoints; training and inference are separate decisions."""
import math
from weekend_thinking.stats import paired_cluster_bootstrap

REPEATS,SEED=20000,2026101915


def contrast(reference,treatment,*,token_nll=False):
    """Treatment minus reference: summed LL,or ordinary mean token NLL."""
    def project(rows):
        if len(rows)!=256 or len({r['item_id'] for r in rows})!=256 or len({r['document_id'] for r in rows})!=256:
            raise ValueError('Exactly256distinct paired documents required;never discard failures')
        result=[]
        for row in rows:
            values=row['logps16']
            if len(values)!=16 or any(not math.isfinite(v) for v in values):
                raise ValueError('Every actual native target token must have a finite score')
            total=math.fsum(values)
            result.append(dict(item_id=row['item_id'],document_id=row['document_id'],
                score=-total/16 if token_nll else total))
        return result
    return paired_cluster_bootstrap(project(reference),project(treatment),repeats=REPEATS,seed=SEED)


def decisions(training_gain,ordinary_regression,inference_gain,own_minus_raw,own_minus_empty,
              *,inference_execution_complete):
    """Implement the already declared independent gates for one fixed seed.

    Ordinary regression here is the post arm's raw-path mean token NLL minus
    raw control. Prior-arm ordinary quality must also be reported as a cost;
    it is not silently added to the predeclared inference gate.
    """
    training=(training_gain['difference_b_minus_a']>=.25 and training_gain['ci95'][0]>0
        and ordinary_regression['difference_b_minus_a']<=.02 and ordinary_regression['ci95'][1]<=.05)
    inference=(inference_execution_complete and inference_gain['difference_b_minus_a']>=.25
        and inference_gain['ci95'][0]>0 and own_minus_raw['ci95'][0]>0 and own_minus_empty['ci95'][0]>0)
    return dict(training_benefit_pass=bool(training),inference_benefit_pass=bool(inference),
        limits='Conditionalonthisseed/checkpoint;reportbothseedsseparately,andB0absolutepaths,costsandallregressions.')
