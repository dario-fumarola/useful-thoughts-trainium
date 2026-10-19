"""Full-support continuation likelihood for explicit native causal contexts.

Generation constructs contexts from prefixes alone. Only this scoring API
receives the true next16 tokens. No sample or token is discarded or truncated.
"""
import torch


@torch.no_grad()
def span_logprobs(backend, contexts, targets, *, width=672, pad_id=151643):
    """Return FP32 [documents,16] native next-token log probabilities.

    Position len(context)-1 predicts target[0]. Original target tokens are
    teacher-forced for the subsequent fifteen predictions, never before them.
    Padding lies after the entire real context and continuation.
    """
    vocab = backend.model.config.vocab_size
    if (not contexts or len(contexts)!=len(targets) or type(width) is not int
            or type(pad_id) is not int or not 0<=pad_id<vocab
            or any(not c or len(y)!=16 or len(c)+16>width
                   or any(type(t) is not int or not 0<=t<vocab for t in list(c)+list(y))
                   for c,y in zip(contexts,targets))):
        raise ValueError('Complete bounded native contexts and16actual targets required;never truncate')
    x = torch.full((len(contexts),width),pad_id,dtype=torch.long)
    positions = []
    for i,(context,target) in enumerate(zip(contexts,targets)):
        ids = list(context)+list(target)
        x[i,:len(ids)] = torch.tensor(ids,dtype=torch.long)
        positions.append([len(context)-1+j for j in range(16)])
    positions = torch.tensor(positions,dtype=torch.long).to(backend.device)
    labels = torch.tensor(targets,dtype=torch.long).to(backend.device)
    hidden = backend.hidden(x.to(backend.device))
    selected = hidden.gather(1,positions[:,:,None].expand(-1,-1,hidden.shape[-1]))
    with backend.head_amp():
        logits = backend.model.lm_head(selected).float()
    logps = logits.log_softmax(-1).gather(-1,labels[:,:,None])[:,:,0]
    backend.sync()
    result = logps.cpu()
    if result.shape!=(len(contexts),16) or not torch.isfinite(result).all():
        raise RuntimeError('Full native sixteen-token likelihood is incomplete or nonfinite')
    return result
