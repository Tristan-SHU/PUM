import copy
import numpy as np
import random
import torch
from torch import nn
import torch.nn.functional as F



def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_kl_divergence(model, target_model, inputs):
    with torch.no_grad():
        ref_outputs = target_model(**inputs)

    ref_probs = F.log_softmax(ref_outputs.logits, dim=-1)
    ref_probs = F.log_softmax(ref_outputs.logits, dim=-1)
    ref_probs = ref_probs.view(-1, ref_outputs.logits.shape[-1])

    outputs = model(**inputs)
    current_probs = F.log_softmax(outputs.logits, dim=-1)
    current_probs = current_probs.view(-1, outputs.logits.shape[-1])

    # minimum KL divergence
    return nn.functional.kl_div(
        current_probs, ref_probs, reduction="batchmean", log_target=True
    ), outputs


def compute_batch_nll(model, inputs):
    # get the sum loss for each sequence in a batch
    # NOTE: not same as model(**inputs).loss but has sum loss for each seq in a batch
    outputs = model(**inputs)
    logits = outputs.logits
    labels = inputs["labels"]
    shifted_labels = labels[..., 1:].contiguous()
    logits = logits[..., :-1, :].contiguous()
    loss_function = nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
    loss = loss_function(logits.transpose(-1, -2), shifted_labels).sum(dim=-1)
    return loss, outputs


def prepare_ref_model(self, model):
    ref_model = copy.deepcopy(model).to(self.accelerator.device)
    ref_model.eval()
    if self.is_deepspeed_enabled:
        ref_model = self._prepare_deepspeed(ref_model)
    else:
        ref_model = self.accelerator.prepare_model(ref_model, evaluation_mode=True)
    return ref_model

def compute_retain_loss(model, retain_inputs, retain_loss_type="NLL"):
    ref_model = None
    if retain_loss_type == "KL":
        ref_model = prepare_ref_model(model)
    retain_outputs = model(**retain_inputs)
    retain_loss = 0.0
    if retain_loss_type == "NLL":
        retain_loss += retain_outputs.loss
    elif retain_loss_type == "KL":
        kl_loss, retain_outputs = compute_kl_divergence(
            model, ref_model, retain_inputs
        )
        retain_loss += kl_loss
    else:
        raise NotImplementedError(
            f"{retain_loss_type} not implemented for retain set"
        )
    return retain_loss


# Loss Functions for Forgetting Methods

def compute_dpo_loss(model, ref_model, win_inputs=None, lose_inputs=None, beta=0.1):
    if win_inputs is None and lose_inputs is None:
        raise ValueError("Both win_inputs and lose_inputs can't be None")

    win_log_ratio, lose_log_ratio = 0.0, 0.0
    win_outputs, lose_outputs = None, None

    if win_inputs is not None:
        win_loss, win_outputs = compute_batch_nll(model, win_inputs)
        with torch.no_grad():
            win_ref_loss, _ = compute_batch_nll(ref_model, win_inputs)
        win_log_ratio = -(win_loss - win_ref_loss)

    if lose_inputs is not None:
        lose_loss, lose_outputs = compute_batch_nll(model, lose_inputs)
        with torch.no_grad():
            lose_ref_loss, _ = compute_batch_nll(ref_model, lose_inputs)
        lose_log_ratio = -(lose_loss - lose_ref_loss)

    loss = -2 / beta * F.logsigmoid(beta * (win_log_ratio - lose_log_ratio)).mean()
    return loss, (win_outputs, lose_outputs)


def compute_undial_loss(model, ref_model, inputs, beta):
    # Forward pass on the student (trainable) model
    outputs = model(**inputs)
    logits = outputs.logits
    labels = inputs["labels"]

    shift_labels = labels[..., 1:].contiguous()
    shift_logits = logits[..., :-1, :].contiguous()

    # Forward pass on the teacher model (no grad)
    with torch.no_grad():
        teacher_logits = ref_model(**inputs).logits
    shift_teacher_logits = teacher_logits[..., :-1, :].contiguous()

    # Build the mask that identifies the tokens need to be unlearned
    mask = torch.zeros_like(shift_teacher_logits)
    batch_idx = torch.arange(mask.shape[0]).view(-1, 1, 1)
    seq_idx = torch.arange(mask.shape[1]).view(1, -1, 1)
    mask[batch_idx, seq_idx, shift_labels.unsqueeze(-1)] = 1.0

    # Adjust teacher logits: subtract di_strength on the correct token
    pre_softmax = shift_teacher_logits - mask * beta
    soft_label = F.softmax(pre_softmax, dim=-1)

    loss_fct = nn.CrossEntropyLoss(reduction="none")
    loss = loss_fct(
        shift_logits.view(-1, shift_logits.size(-1)),
        soft_label.view(-1, soft_label.size(-1)),
    )
    return loss.mean(), outputs


def compute_wga_loss(model, inputs, beta):
    outputs = model(**inputs)
    labels = inputs["labels"]
    labels = labels.to(outputs.logits.device)

    shift_logits = outputs.logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    lm_loss = nn.CrossEntropyLoss(ignore_index=-100, reduction="none")(
        shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
    )
    weight_ce = ((-lm_loss).exp().detach()) ** beta
    forget_loss = -(weight_ce * lm_loss)[shift_labels.view(-1) != -100].mean()
    return forget_loss, outputs


def compute_satimp_loss(model, inputs, beta1, beta2):
    outputs = model(**inputs)
    labels = inputs["labels"]
    labels = labels.to(outputs.logits.device)

    shift_logits = outputs.logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    lm_loss = nn.CrossEntropyLoss(ignore_index=-100, reduction="none")(
        shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
    )
    weight_sat = ((-lm_loss).exp().detach()) ** beta1
    weight_imp = (1 - (-lm_loss).exp().detach()) ** beta2
    forget_loss = -((weight_sat * weight_imp) * lm_loss)[
        shift_labels.view(-1) != -100
    ].mean()
    return forget_loss, outputs


def compute_gradascent(model, inputs):
    forget_inputs = inputs["forget"]
    forget_inputs = {
        "input_ids": forget_inputs["input_ids"],
        "attention_mask": forget_inputs["attention_mask"],
        "labels": forget_inputs["labels"],
    }
    outputs = model(**forget_inputs)
    loss = -outputs.loss
    return loss, outputs


def compute_graddiff(model, inputs, gamma=1.0, alpha=1.0, retain_loss_type="NLL"):
    forget_inputs = inputs["forget"]
    forget_inputs = {
        "input_ids": forget_inputs["input_ids"],
        "attention_mask": forget_inputs["attention_mask"],
        "labels": forget_inputs["labels"],
    }

    forget_outputs = model(**forget_inputs)
    forget_loss = -forget_outputs.loss

    retain_inputs = inputs["retain"]
    retain_inputs = {
        "input_ids": retain_inputs["input_ids"],
        "attention_mask": retain_inputs["attention_mask"],
        "labels": retain_inputs["labels"],
    }
    retain_loss = compute_retain_loss(model=model, retain_inputs=retain_inputs, retain_loss_type=retain_loss_type)

    loss = gamma * forget_loss + alpha * retain_loss

    return loss, forget_outputs


def compute_dpo(model, inputs, alpha=1.0, beta=0.1, gamma=1.0, retain_loss_type="NLL"):
    ref_model = prepare_ref_model(model)
    forget_inputs = inputs["forget"]["original"]
    alternate_inputs = inputs["forget"]["alternate"]

    forget_loss, forget_outputs = compute_dpo_loss(
        model=model,
        ref_model=ref_model,
        win_inputs=alternate_inputs,
        lose_inputs=forget_inputs,
        beta=beta,
    )

    retain_inputs = inputs["retain"]
    retain_inputs = {
        "input_ids": retain_inputs["input_ids"],
        "attention_mask": retain_inputs["attention_mask"],
        "labels": retain_inputs["labels"],
    }
    retain_loss = compute_retain_loss(model=model, retain_inputs=retain_inputs, retain_loss_type=retain_loss_type)

    loss = gamma * forget_loss + alpha * retain_loss
    return loss, forget_outputs


def compute_npo(model, inputs, alpha=1.0, beta=0.1, gamma=1.0):
    ref_model = prepare_ref_model(model)
    forget_inputs = inputs["forget"]

    forget_loss, forget_outputs = compute_dpo_loss(
        model=model,
        ref_model=ref_model,
        win_inputs=None,
        lose_inputs=forget_inputs,
        beta=beta,
    )

    retain_inputs = inputs["retain"]
    retain_inputs = {
        "input_ids": retain_inputs["input_ids"],
        "attention_mask": retain_inputs["attention_mask"],
        "labels": retain_inputs["labels"],
    }
    retain_loss = compute_retain_loss(model=model, retain_inputs=retain_inputs)

    loss = gamma * forget_loss + alpha * retain_loss
    return loss, forget_outputs


def compute_simnpo(model, inputs, alpha=1.0, beta=4.5, delta=0.0, gamma=0.125, retain_loss_type="NLL"):
    forget_inputs = inputs["forget"]

    forget_labels = forget_inputs["labels"]
    loss_mask = forget_labels != -100
    forget_loss, forget_outputs = compute_batch_nll(model, forget_inputs)
    forget_loss = forget_loss / loss_mask.sum(-1) - delta
    forget_loss = -F.logsigmoid(beta * forget_loss).mean() * 2 / beta

    retain_inputs = inputs["retain"]
    retain_inputs = {
        "input_ids": retain_inputs["input_ids"],
        "attention_mask": retain_inputs["attention_mask"],
        "labels": retain_inputs["labels"],
    }
    retain_loss = compute_retain_loss(model=model, retain_inputs=retain_inputs, retain_loss_type=retain_loss_type)

    loss = gamma * forget_loss + alpha * retain_loss
    return loss, forget_outputs


def compute_satimp(model, inputs, alpha=1.0, beta1=5.0, beta2=1.0, gamma=0.1, retain_loss_type="NLL"):
    forget_inputs = inputs["forget"]
    forget_inputs = {
        "input_ids": forget_inputs["input_ids"],
        "attention_mask": forget_inputs["attention_mask"],
        "labels": forget_inputs["labels"],
    }
    forget_loss, forget_outputs = compute_satimp_loss(
        model=model, inputs=forget_inputs, beta1=beta1, beta2=beta2
    )

    retain_inputs = inputs["retain"]
    retain_inputs = {
        "input_ids": retain_inputs["input_ids"],
        "attention_mask": retain_inputs["attention_mask"],
        "labels": retain_inputs["labels"],
    }
    retain_loss = compute_retain_loss(model=model, retain_inputs=retain_inputs, retain_loss_type=retain_loss_type)

    loss = gamma * forget_loss + alpha * retain_loss
    return loss, forget_outputs


def compute_undial(model, inputs, alpha=0.0, beta=10.0, gamma=1.0, retain_loss_type="NLL"):
    ref_model = prepare_ref_model(model)
    forget_inputs = inputs["forget"]
    forget_loss, forget_outputs = compute_undial_loss(
        model, ref_model, forget_inputs, beta
    )

    retain_inputs = inputs["retain"]
    retain_inputs = {
        "input_ids": retain_inputs["input_ids"],
        "attention_mask": retain_inputs["attention_mask"],
        "labels": retain_inputs["labels"],
    }
    retain_loss = compute_retain_loss(model=model, retain_inputs=retain_inputs, retain_loss_type=retain_loss_type)

    loss = gamma * forget_loss + alpha * retain_loss
    return loss, forget_outputs


def compute_wga(model, inputs, alpha=1.0, beta=1.0, gamma=1.0, retain_loss_type="NLL"):
    forget_inputs = inputs["forget"]
    forget_inputs = {
        "input_ids": forget_inputs["input_ids"],
        "attention_mask": forget_inputs["attention_mask"],
        "labels": forget_inputs["labels"],
    }
    forget_loss, forget_outputs = compute_wga_loss(
        model=model, inputs=forget_inputs, beta=beta
    )

    retain_inputs = inputs["retain"]
    retain_inputs = {
        "input_ids": retain_inputs["input_ids"],
        "attention_mask": retain_inputs["attention_mask"],
        "labels": retain_inputs["labels"],
    }
    retain_loss = compute_retain_loss(model=model, retain_inputs=retain_inputs, retain_loss_type=retain_loss_type)

    loss = (
        gamma * forget_loss + alpha * retain_loss
    )  # default gamma=1.0 alpha=1.0
    return loss, forget_outputs