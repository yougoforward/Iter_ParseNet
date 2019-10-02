import math

def cosine_decay(learning_rate, global_step, warm_step, warm_lr, decay_steps, alpha=0.0001):
    # warm_step = 5 * iters_per_epoch
    # warm_lr = 0.01 * learning_rate
    # current_step = epoch * iters_per_epoch + i_iter
    if global_step < warm_step:
        lr = warm_lr
    else:
        decay_steps = decay_steps-warm_step
        global_step = min(global_step, decay_steps)-warm_step
        cosine_decay = 0.5 * (1 + math.cos(math.pi * global_step / decay_steps))
        decayed = (1 - alpha) * cosine_decay + alpha
        lr = learning_rate * decayed

    return lr


def restart_cosine_decay(learning_rate, global_step, warm_step, warm_lr, decay_steps, alpha=0.0001):
    # warm_step = 5 * iters_per_epoch
    # warm_lr = 0.01 * learning_rate
    # current_step = epoch * iters_per_epoch + i_iter
    restart_step = int((warm_step+decay_steps)/2)
    if global_step < warm_step:
        lr = warm_lr
    elif global_step <restart_step:
        end_steps = restart_step-warm_step
        cur_step = global_step-warm_step
        cosine_decay = 0.5 * (1 + math.cos(math.pi * cur_step / end_steps))
        decayed = (1 - alpha) * cosine_decay + alpha
        lr = learning_rate * decayed
    else:
        end_steps = decay_steps - restart_step
        cur_step = min(global_step, decay_steps) - restart_step
        cosine_decay = 0.5 * (1 + math.cos(math.pi * cur_step / end_steps))
        decayed = (1 - alpha) * cosine_decay + alpha
        lr = learning_rate * decayed
    return lr