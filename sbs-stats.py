#### Means comparisons
def t_test_means(mean_a, mean_b, std_a, std_b, n_a, n_b):
    diff = mean_a - mean_b
    ratio = mean_a / mean_b
    pooled_std = (std_a * n_a + std_b * n_b) / (n_a + n_b)
    effect_size = diff / pooled_std
    error = np.sqrt( std_a**2 / n_a + std_b**2 / n_b )
    t_score = diff / error
    p_val = st.t.sf(np.abs(t_score), df = n_a + n_b - 1)
    
    return diff, ratio, effect_size, t_score, p_val

#### Proportions comparisons
def z_test_proportions(pos_a, pos_b, n_a, n_b):
    diff = pos_a / n_a - pos_b / n_b
    ratio = (pos_a / n_a) / (pos_b / n_b)
    total_proportion = (pos_a + pos_b) / (n_a + n_b)
    error = np.sqrt( total_proportion * (1 - total_proportion) * (1 / n_a + 1 / n_b) )
    z_score = diff / error
    p_val = 1 - st.norm(loc=0, scale=1).cdf( np.abs(z_score) )
    
    effect_size = 2 * ( np.arcsin(np.sqrt(pos_a / n_a)) - np.arcsin(np.sqrt(pos_b / n_b)) )
    
    return diff, ratio, effect_size, z_score, p_val