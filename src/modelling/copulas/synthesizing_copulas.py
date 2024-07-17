def main_copulas_sampling(copulas, n_sample, df_processor):
    # sample n_sample
    sampled_data = copulas.sample(n_sample)

    # reverse prepare (post processing)
    # TO DO: check if len(sampled_data) is n_sample after reverse transform because of conditions
    sampled_data = df_processor.reverse_transform(sampled_data)

    return sampled_data
