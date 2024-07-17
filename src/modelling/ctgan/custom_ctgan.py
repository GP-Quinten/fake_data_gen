import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import optim
from sdmetrics.single_column import KSComplement, TVComplement
from sdmetrics.column_pairs import ContingencySimilarity, CorrelationSimilarity
from itertools import combinations

from ctgan import CTGAN
from ctgan.data_transformer import DataTransformer
from ctgan.data_sampler import DataSampler
from ctgan.synthesizers.ctgan import Generator, Discriminator


class custom_CTGAN(CTGAN):
    def __init__(
        self,
        embedding_dim=128,
        generator_dim=(256, 256),
        discriminator_dim=(256, 256),
        generator_lr=2e-4,
        generator_decay=1e-6,
        discriminator_lr=2e-4,
        discriminator_decay=1e-6,
        batch_size=500,
        discriminator_steps=1,
        log_frequency=True,
        verbose=False,
        epochs=300,
        pac=10,
        cuda=True,
    ):

        super().__init__(
            embedding_dim=128,
            generator_dim=(256, 256),
            discriminator_dim=(256, 256),
            generator_lr=2e-4,
            generator_decay=1e-6,
            discriminator_lr=2e-4,
            discriminator_decay=1e-6,
            batch_size=500,
            discriminator_steps=1,
            log_frequency=True,
            verbose=False,
            epochs=300,
            pac=10,
            cuda=True,
        )

        self.losses_d = []
        self.losses_g = []
        self.similarity_per_column = pd.DataFrame()
        self.correlation_per_column = pd.DataFrame()

    def fit(
        self,
        train_data,
        subject_id,
        discrete_columns=(),
        epochs=None,
        n_samples_to_eval=None,
    ):
        """Fit the CTGAN Synthesizer models to the training data.

        Args:
            train_data (numpy.ndarray or pandas.DataFrame):
                Training Data. It must be a 2-dimensional numpy array or a pandas.DataFrame.
            discrete_columns (list-like):
                List of discrete columns to be used to generate the Conditional
                Vector. If ``train_data`` is a Numpy array, this list should
                contain the integer indices of the columns. Otherwise, if it is
                a ``pandas.DataFrame``, this list should contain the column names.
        """

        self._validate_discrete_columns(train_data, discrete_columns)

        if epochs is None:
            epochs = self._epochs
        else:
            warnings.warn(
                (
                    "`epochs` argument in `fit` method has been deprecated and will be removed "
                    "in a future version. Please pass `epochs` to the constructor instead"
                ),
                DeprecationWarning,
            )

        self._transformer = DataTransformer()
        self._transformer.fit(train_data, discrete_columns)

        real_data = train_data
        train_data = self._transformer.transform(train_data)

        self._data_sampler = DataSampler(
            train_data, self._transformer.output_info_list, self._log_frequency
        )

        data_dim = self._transformer.output_dimensions

        self._generator = Generator(
            self._embedding_dim + self._data_sampler.dim_cond_vec(),
            self._generator_dim,
            data_dim,
        ).to(self._device)

        discriminator = Discriminator(
            data_dim + self._data_sampler.dim_cond_vec(),
            self._discriminator_dim,
            pac=self.pac,
        ).to(self._device)

        optimizerG = optim.Adam(
            self._generator.parameters(),
            lr=self._generator_lr,
            betas=(0.5, 0.9),
            weight_decay=self._generator_decay,
        )

        optimizerD = optim.Adam(
            discriminator.parameters(),
            lr=self._discriminator_lr,
            betas=(0.5, 0.9),
            weight_decay=self._discriminator_decay,
        )

        mean = torch.zeros(self._batch_size, self._embedding_dim, device=self._device)
        std = mean + 1

        steps_per_epoch = max(len(train_data) // self._batch_size, 1)

        for i in range(epochs):
            for id_ in range(steps_per_epoch):
                for n in range(self._discriminator_steps):
                    fakez = torch.normal(mean=mean, std=std)

                    condvec = self._data_sampler.sample_condvec(self._batch_size)
                    if condvec is None:
                        c1, m1, col, opt = None, None, None, None
                        real = self._data_sampler.sample_data(
                            self._batch_size, col, opt
                        )
                    else:
                        c1, m1, col, opt = condvec
                        c1 = torch.from_numpy(c1).to(self._device)
                        m1 = torch.from_numpy(m1).to(self._device)
                        fakez = torch.cat([fakez, c1], dim=1)

                        perm = np.arange(self._batch_size)
                        np.random.shuffle(perm)
                        real = self._data_sampler.sample_data(
                            self._batch_size, col[perm], opt[perm]
                        )
                        c2 = c1[perm]

                    fake = self._generator(fakez)
                    fakeact = self._apply_activate(fake)

                    real = torch.from_numpy(real.astype("float32")).to(self._device)

                    if c1 is not None:
                        fake_cat = torch.cat([fakeact, c1], dim=1)
                        real_cat = torch.cat([real, c2], dim=1)
                    else:
                        real_cat = real
                        fake_cat = fakeact

                    y_fake = discriminator(fake_cat)
                    y_real = discriminator(real_cat)

                    pen = discriminator.calc_gradient_penalty(
                        real_cat, fake_cat, self._device, self.pac
                    )
                    loss_d = -(torch.mean(y_real) - torch.mean(y_fake))

                    optimizerD.zero_grad(set_to_none=False)
                    pen.backward(retain_graph=True)
                    loss_d.backward()
                    optimizerD.step()

                fakez = torch.normal(mean=mean, std=std)
                condvec = self._data_sampler.sample_condvec(self._batch_size)

                if condvec is None:
                    c1, m1, col, opt = None, None, None, None
                else:
                    c1, m1, col, opt = condvec
                    c1 = torch.from_numpy(c1).to(self._device)
                    m1 = torch.from_numpy(m1).to(self._device)
                    fakez = torch.cat([fakez, c1], dim=1)

                fake = self._generator(fakez)
                fakeact = self._apply_activate(fake)

                if c1 is not None:
                    y_fake = discriminator(torch.cat([fakeact, c1], dim=1))
                else:
                    y_fake = discriminator(fakeact)

                if condvec is None:
                    cross_entropy = 0
                else:
                    cross_entropy = self._cond_loss(fake, c1, m1)

                loss_g = -torch.mean(y_fake) + cross_entropy

                optimizerG.zero_grad(set_to_none=False)
                loss_g.backward()
                optimizerG.step()

            if self._verbose:
                print(
                    f"Epoch {i+1}, Loss G: {loss_g.detach().cpu(): .4f},"  # noqa: T001
                    f"Loss D: {loss_d.detach().cpu(): .4f}",
                    flush=True,
                )

            self.losses_d.append(loss_d.cpu().detach().numpy())
            self.losses_g.append(loss_g.cpu().detach().numpy())
            if n_samples_to_eval is not None:
                test_fake = self.sample(n_samples_to_eval)
                test_real = real_data.sample(n_samples_to_eval)
                columns_list = (test_real.columns).to_list()
                if subject_id in columns_list:
                    columns_list.remove(subject_id)

                for col in columns_list:
                    if col in discrete_columns:
                        score = TVComplement.compute(
                            real_data=test_real[col], synthetic_data=test_fake[col]
                        )
                    else:
                        score = KSComplement.compute(
                            real_data=test_real[col], synthetic_data=test_fake[col]
                        )
                    self.similarity_per_column.loc[i, col] = score

                col_combination = combinations(columns_list, 2)
                for (col1, col2) in col_combination:
                    if col1 in discrete_columns and col2 in discrete_columns:
                        score = ContingencySimilarity.compute(
                            real_data=test_real[[col1, col2]],
                            synthetic_data=test_fake[[col1, col2]],
                        )
                        self.correlation_per_column.loc[
                            i, f"corr_{col1}_{col2}"
                        ] = score
                    elif col1 not in discrete_columns and col2 not in discrete_columns:
                        score = CorrelationSimilarity.compute(
                            real_data=test_real[[col1, col2]],
                            synthetic_data=test_fake[[col1, col2]],
                        )
                        self.correlation_per_column.loc[
                            i, f"corr_{col1}_{col2}"
                        ] = score
                    else:
                        pass

    def return_losses(self):
        return (self.losses_d, self.losses_g)

    def return_scores(self):
        return (self.similarity_per_column, self.correlation_per_column)
