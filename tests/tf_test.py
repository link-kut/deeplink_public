import tensorflow as tf

action = [[0, 1], [1, 0]]
logits = tf.Variable(initial_value=[[1.8, 0.2], [1.0, 1.0]])
policy = tf.nn.softmax(logits)

entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=policy, logits=logits)
entropy2 = tf.nn.softmax_cross_entropy_with_logits_v2(labels=policy, logits=policy)

entropy3 = -1 * tf.reduce_sum(policy * tf.log(policy + 1e-20), axis=1)

policy_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.math.argmax(action, axis=1), logits=logits)

policy_loss2 = tf.nn.softmax_cross_entropy_with_logits_v2(labels=action, logits=logits)

if __name__ == "__main__":
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    p, e, e2, e3 = sess.run([policy, entropy, entropy2, entropy3])
    p_l, p_l2 = sess.run([policy_loss, policy_loss2])

    print("policy", p, end="\n\n")
    print("entropy", e, end="\n\n")
    print("entropy2", e2, end="\n\n")
    print("entropy3", e3, end="\n\n")
    print("policy_loss", p_l, end="\n\n")
    print("policy_loss2", p_l2, end="\n\n")
