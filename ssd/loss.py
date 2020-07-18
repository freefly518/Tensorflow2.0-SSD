import tensorflow as tf
import numpy as np

def softmax(x):
    """Compute the softmax of vector x."""
    exp_x = np.exp(x)
    softmax_x = exp_x / np.sum(exp_x)
    return softmax_x

class SmoothL1Loss(object):
    def __init__(self):
        pass
    def __call__(self, y_true, y_pred, *args, **kwargs):
        absolute_value = tf.math.abs(y_true - y_pred)
        mask_boolean = tf.math.greater_equal(x=absolute_value, y=1.0)
        mask_float32 = tf.cast(x=mask_boolean, dtype=tf.float32)
        smooth_l1_loss = (1.0 - mask_float32) * 0.5 * tf.math.square(absolute_value) + mask_float32 * (absolute_value - 0.5)
        return tf.math.reduce_sum(smooth_l1_loss)

class BoxLoss(object):
    def __init__(self):
        pass

    def hard_negative_mining(self, conf_loss, num_pos, neg_ratio):
        '''Return negative indices that is 3x the number as postive indices.
        Args:
          conf_loss: (Tensor) cross entroy loss between conf_preds and conf_targets, sized [N,anchor_number,].
          pos_num: (int) postive number
          neg_ratio: (int) negative / positive ratio
        Return:
          (ndarray) negative indices, sized [N,anchor_number].
        '''
        num_neg = num_pos * neg_ratio

        rank = tf.argsort(conf_loss, axis=1, direction='DESCENDING')
        rank = tf.argsort(rank, axis=1)

        neg_idx = rank < tf.expand_dims(num_neg, 1)
        return neg_idx

    def __call__(self, loc_preds, loc_targets, conf_preds, conf_targets):
        '''Compute loss between (loc_preds, loc_targets) and (conf_preds, conf_targets).
        Args:
          loc_preds: (ndarray) predicted locations, sized [batch_size, anchor_number, 4].
          loc_targets: (ndarray) encoded target locations, sized [batch_size, anchor_number, 4].
          conf_preds: (ndarray) predicted class confidences, sized [batch_size, anchor_number, num_classes].
          conf_targets: (ndarray) encoded target classes, sized [batch_size, anchor_number].
        loss:
          (ndarray) loss = SmoothL1Loss(loc_preds, loc_targets) + CrossEntropyLoss(conf_preds, conf_targets).
        '''

        cross_entropy = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction='none')
        conf_loss = cross_entropy(
            conf_targets, conf_preds)

        # postive index
        pos_mask = conf_targets > 0 # [N,anchor_number]
        pos_num = np.sum(pos_mask)

        if pos_num == 0:
            return tf.constant([0.0]), tf.constant([0.0])

        num_pos = tf.reduce_sum(tf.dtypes.cast(pos_mask, tf.int32), axis=1)
        neg_mask = self.hard_negative_mining(conf_loss, num_pos, 3)    # [N, anchor_number]

        mask = tf.math.logical_or(pos_mask, neg_mask).numpy()
        cross_entropy = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction='sum')

        conf_loss = cross_entropy(
          conf_targets[mask], conf_preds[mask]
        )
        smooth_l1_loss = SmoothL1Loss()
        loc_loss = smooth_l1_loss(
            loc_targets[pos_mask],
            loc_preds[pos_mask])

        conf_loss = conf_loss / pos_num
        loc_loss = loc_loss / pos_num

        return conf_loss, loc_loss