from matplotlib import pyplot as plt

def plot_all_losses(
    loss_d: list,
    loss_g: list,
    raw_loss_g: list,
    conditional_loss: list,
    loss_info: list,
    loss_cc: list,
    loss_cg: list,
    loss_total_g: list,
):
    """
    Gets the graph of all losses evolution across epochs during training. Epochs as x axis, losses values as y axis.

    Args:
        loss_d (list): discriminator loss
        loss_g (list): identical loss as in ctgan generator loss, being the sum of raw_loss_g + conditional_g
        raw_loss_g (list): raw loss of generator. raw_loss_g = -(torch.log(y_fake + 1e-4).mean())
        conditional_loss (list): loss related to the conditional vector -> Ensure the generator generates data records with the chosen category as per the conditional vector. cross_entropy = cond_loss(faket, self.transformer.output_info, c, m)
        loss_info (list): information loss by comparing means and stds of real/fake feature representations extracted from discriminator's penultimate layer. statiscal proximity
        loss_cc (list): loss to train the classifier so that it can perform well on the real data. loss_cc = c_loss(real_pre, real_label), (cross_entropy_loss)
        loss_cg (list): loss to train the generator to improve semantic integrity between target column and rest of the data. loss_cg = c_loss(fake_pre, fake_label), (cross_entropy_loss)
        loss_total_g (list): loss_total_g = raw_loss_g + conditional_loss + loss_info + loss_cc + loss_cg.

    Returns:
        fig: figure with epochs as x axis, losses values as y values. Range of values on the right are low values in blue asssociated to all losses except loss_info and loss_total_g
    """

    # plot loss
    fig, ax1 = plt.subplots(figsize=(16, 8))

    ax1.set_xlabel("epochs")
    ax1.set_ylabel("low values", color="b", fontsize=14)

    ax1.plot(loss_d, label="DISCRIMINATOR", color="tab:blue", linewidth=5)
    ax1.plot(loss_g, label="generator = raw + cond", color="tab:brown")
    ax1.plot(raw_loss_g, label="raw generator loss", color="tab:olive")
    ax1.plot(conditional_loss, label="conditional loss", color="tab:cyan")
    ax1.plot(loss_cc, label="cc loss", color="magenta")
    ax1.plot(loss_cg, label="cg loss", color="tab:green")
    ax1.tick_params(axis="y", labelcolor="b")
    ax1.legend(fontsize=14, loc="upper left")

    ax2 = ax1.twinx()
    ax2.set_ylabel(
        "high values", color="tab:orange", fontsize=14
    )  # we already handled the x-label with ax1
    ax2.plot(loss_info, label="Info Loss", color="tab:red")
    ax2.plot(
        loss_total_g,
        label="LOSS_TOTAL_G = generator + info + cg",
        color="tab:orange",
        linewidth=5,
    )
    ax2.legend(fontsize=14, loc="upper right")
    ax2.tick_params(axis="y", labelcolor="r")

    return fig
