"""Question-only QA prompts and explicit offline evidence interventions."""
import copy
import re

from train.document_evidence_data import render
from train.document_qa_packing import prompt, FINAL_OPEN


# Invented demonstration facts, fixed before model outputs. They are not
# selected from held questions or pre-existing model responses.
REFERENCE = '''Use the documents to answer the question. Show the relevant steps.
Finish with a separate line beginning "Final answer:" and the short answer.

Documents:
[1] Oriole project
The Oriole project is managed by Nera.
[2] Nera
Nera lives in Northport.
[3] Another manager
Kavo manages a different project and lives in Riverton.

Question: Where does the manager of the Oriole project live?
Reasoning:
The Oriole project is managed by Nera. Nera lives in Northport.
Final answer: Northport

Documents:
[1] Copper Prize
The Copper Prize was awarded to Seli.
[2] Seli
Seli was born in Dawnfield.
[3] Dawnfield
Dawnfield belongs to the Cedar region.
[4] Other place
Eveningford belongs to the Ash region.

Question: In which region was the Copper Prize winner born?
Reasoning:
The winner was Seli. Seli was born in Dawnfield, which belongs to Cedar.
Final answer: Cedar

'''


def context(question, mode, trace=None):
    """No answers, aliases, supporting flags or decomposition are accepted."""
    if mode=='direct':
        return prompt(question,'direct')
    if mode=='prior':
        return prompt(question,'prior')
    if mode=='reference_cot':
        return REFERENCE+prompt(question,'prior')
    if mode=='empty':
        return prompt(question,'prior')+FINAL_OPEN
    if mode=='replay':
        if not isinstance(trace,str):raise ValueError('Explicit generated or oracle trace required')
        return prompt(question,'prior')+trace+FINAL_OPEN
    raise ValueError('Declared QA context mode required')


def question_only(question):
    # Validate the original allowed projection before constructing the ablation.
    render(question)
    return dict(item_id=question['item_id'],question=question['question'],
                contexts=[dict(title='No documents supplied',text='The source passages are omitted for this ablation.')])


def omit_evidence(question, index):
    """OFFLINE intervention index is prepared from support metadata elsewhere.

    Keep every title and paragraph index stable; replace one paragraph's text.
    This intervention is diagnostic, not a deployed selection policy.
    """
    render(question)
    if type(index) is not int or not 0<=index<len(question['contexts']):
        raise ValueError('Existing paragraph index required')
    result=copy.deepcopy(question)
    result['contexts'][index]['text']='[Paragraph omitted for evidence ablation.]'
    return result


def trace_from_output(text):
    fields=list(re.finditer(r'(?im)^[ \t]*Final answer:',text))
    return (text[:fields[0].start()].rstrip('\r\n'),True) if fields else (text.rstrip('\r\n'),False)
