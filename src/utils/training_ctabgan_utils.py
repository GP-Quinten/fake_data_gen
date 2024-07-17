from src.visualizing import visualizing
from src.visualizing.ctabgan import visualizing_ctabgan


def training_losses_evolution(synthesizer):
    """
    Returns a tuple of 2 figures of graphs of convergences losses. The first is only with discriminator + generator losses. The second one is with the detailed losses composing the generator loss.

    Args:
        synthesizer (ctabgan synthesizer object): ctabgan synthesizer/model object.

    Returns:
        tuple: tuple of 2 figures
    """

    ## Retrieve losses
    loss_d = synthesizer.synthesizer.loss_d
    loss_g = synthesizer.synthesizer.loss_g
    raw_loss_g = synthesizer.synthesizer.raw_loss_g
    conditional_loss = synthesizer.synthesizer.conditional_loss
    loss_info = synthesizer.synthesizer.loss_info
    loss_cg = synthesizer.synthesizer.loss_cg
    loss_cc = synthesizer.synthesizer.loss_cc
    loss_total_g = synthesizer.synthesizer.loss_total_g

    fig_plot_losses = visualizing.plot_losses(
        loss_d,
        loss_total_g,
    )

    fig_plot_all_losses = visualizing_ctabgan.plot_all_losses(
        loss_d,
        loss_g,
        raw_loss_g,
        conditional_loss,
        loss_info,
        loss_cc,
        loss_cg,
        loss_total_g,
    )

    return fig_plot_losses, fig_plot_all_losses
