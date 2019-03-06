import rouge  # pip install py-rouge
from collections import defaultdict


def compute_score(hypotheses, references):
    ''' Compute the average F1 rouge scores taking into account all elements in dataset.

    Args:
        - hypotheses : list of list of tokens for each hypotheses
        - references : list of list of tokens for each reference
    Return:
        - dict with average score for rouge-1, rouge-2, rouge-l, scaled between 0 and 100

    '''

    evaluator = rouge.Rouge(metrics=['rouge-n', 'rouge-l'],
                            max_n=2,
                            limit_length=True,
                            length_limit=200,
                            length_limit_type='words',
                            apply_avg=False,
                            apply_best=True,
                            alpha=0.5,  # Default F1_score
                            weight_factor=1.2,
                            stemming=True)

    avg_f_score = defaultdict(float)

    for hypothesis, reference in zip(hypotheses, references):
        scores = evaluator.get_scores(hypothesis, reference)

        # print(scores)
        for key in scores:
            avg_f_score[key] += scores[key]['f']

    for key, v in avg_f_score.items():
        avg_f_score[key] = v/len(hypotheses)*100

    return avg_f_score


def update(avg_rouge_score, number_elts_avg, avg_rouge_score_batch, nb_elts_batch):
    
    for key in avg_rouge_score_batch:
        avg_rouge_score[key] = (avg_rouge_score[key]*number_elts_avg
                             + avg_rouge_score_batch[key]*nb_elts_batch) \
                                 /(number_elts_avg+nb_elts_batch)


if __name__ == "__main__":

    summary_0_hypothesis = 'In a 2008 Faculty Newsletter article , “ Change in Education : The cost of sacrificing fundamentals , ” MIT Professor Emeritus Ernst G. Frankel expresses his concerns regarding the current state of American engineering education .   He notes that the number of students focusing on traditional areas of engineering has decreased while the number interested in the high-technology end of the field has increased . Frankel points out that other industrial nations produce far more traditionally trained engineers than we do , and believes we have fallen seriously behind . '
    summary_0_reference = 'MIT Professor Emeritus Ernst G. Frankel (2008) has called for a return to a course of study that emphasizes the traditional skills of engineering , noting that the number of American engineering graduates with these skills has fallen sharply when compared to the number coming from other countries . '

    summary_1_hypothesis = 'This book looks at the thin line between right and wrong . '
    summary_1_reference = 'This book explores the meaning of truth, and asks if it really has a sense . '

    hypotheses = [summary_0_hypothesis, summary_1_hypothesis]
    references = [summary_0_reference, summary_1_reference]

    print(compute_score(hypotheses, references))
