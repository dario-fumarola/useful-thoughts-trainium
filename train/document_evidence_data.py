"""Author-distributed MuSiQue TRAIN: evidence task input/label separation.

Only the question and visible paragraphs enter the model. Answers, gold
decompositions, supporting flags and retrieval scores are offline annotations.
This prepares a candidate diagnostic, not a model or evaluation release.
"""


def question_projection(row):
    """Read only the four allowed textual input fields; labels may be poisoned."""
    contexts = [{'title': c['wikipedia_title'], 'text': c['paragraph_text']}
                for c in row['contexts']]
    # Authors' raw question_text is a composition program with [[CQS]] and
    # [[RQS]] placeholders. The authored natural-language question is here.
    result = dict(item_id=row['id'], question=row['composed_question_text'], contexts=contexts)
    if (not isinstance(result['item_id'], str) or not result['item_id']
            or not isinstance(result['question'], str) or not result['question'].strip()
            or any(marker in result['question'] for marker in ('[[CQS]]','[[RQS]]','[[CQE]]'))
            or not contexts or any(not isinstance(v,str) or not v.strip()
                                   for c in contexts for v in c.values())):
        raise ValueError('An actual question and complete visible title/text paragraphs are required')
    return result


def render(question):
    if set(question) != {'item_id','question','contexts'}:
        raise ValueError('Only the explicit question/context projection may be rendered')
    paragraphs = []
    for index, context in enumerate(question['contexts'],1):
        if set(context) != {'title','text'}:
            raise ValueError('Do not expose source support or retrieval annotations')
        paragraphs.append(f'[{index}] {context["title"]}\n{context["text"]}')
    return 'Documents:\n' + '\n\n'.join(paragraphs) + '\n\nQuestion: ' + question['question'] + '\n'


def supplied_support(row, question):
    """An OFFLINE oracle consisting solely of paragraphs visible to the model.

    It can contain the final answer because the answer is already in the input
    documents. Gold evidence selection is privileged supervision, not a deployed
    note and not proof that the prefix-only model can select the same evidence.
    """
    if question_projection(row) != question:
        raise ValueError('Evidence labels must refer to the exact visible input')
    support = [i for i,c in enumerate(row['contexts']) if c['is_supporting']]
    if not support:
        raise ValueError('Supporting paragraphs are required for this answerable-task candidate')
    return '\n\n'.join(f'[{i+1}] {question["contexts"][i]["title"]}\n'
                       + question['contexts'][i]['text'] for i in support)
