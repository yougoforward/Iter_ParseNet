import math



def cosine_decay(learning_rate, global_step, decay_steps, alpha=0.0):
    global_step = min(global_step, decay_steps)
    cosine_decay = 0.5*(1+math.cos(math.pi*global_step/decay_steps))
    decayed = (1-alpha)*cosine_decay+alpha
    decayed_learning_rate = learning_rate*decayed
    return decayed_learning_rate




def cosine_decay(learning_rate, global_step, decay_steps, alpha=0.0):
    global_step = min(global_step, decay_steps)
    cosine_decay = 0.5*(1+math.cos(math.pi*global_step/decay_steps))
    decayed = (1-alpha)*cosine_decay+alpha
    decayed_learning_rate = learning_rate*decayed
    return decayed_learning_rate