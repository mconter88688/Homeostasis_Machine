
def ema(prev_data, new_unsmoothed_data, prev_data_flag, alpha):
    if not prev_data_flag:
        return new_unsmoothed_data
    return alpha * new_unsmoothed_data + (1 - alpha) * prev_data