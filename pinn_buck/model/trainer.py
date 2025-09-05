from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Tuple, List, Optional

import torch

from ..model_results.history import TrainingHistory
from ..parameters.parameter_class import Parameters

from .model_param_estimator import BaseBuckEstimator
from .map_loss import MAPLoss


@dataclass
class TrainingConfigs:
    device: str = "cpu"
    patience: int = 5_000
    lr_adam: float = 1e-3
    epochs_adam: int = 20_000
    lr_reduction_factor_adam: float = 0.5
    epochs_lbfgs: int = 1500
    lr_lbfgs: float = 1e-3
    history_size_lbfgs: int = 50
    max_iter_lbfgs: int = 10
    clip_gradient_adam: float = None
    save_every_adam: int = 1000
    save_every_lbfgs: int = 100


class Trainer:
    def __init__(
        self,
        model: BaseBuckEstimator,
        map_loss: MAPLoss,
        cfg: TrainingConfigs = TrainingConfigs(),
        device="cpu",
    ):
        self.model = model.to(device)
        self.model_class = model.__class__
        self.map_loss = map_loss
        self.cfg = cfg
        self.device = device
        self._history = {"loss": [], "params": [], "lr": [], "optimizer": [], "epochs": [], "callbacks": []}
        self.callback_count = 0

    @property
    def history(self) -> TrainingHistory:
        return TrainingHistory.from_histories(
            loss_history=self._history["loss"],
            param_history=self._history["params"],
            optimizer_history=self._history["optimizer"],
            learning_rate=self._history["lr"],
            epochs=self._history["epochs"],
            callbacks=self._history["callbacks"],
        )


    @staticmethod
    def print_parameters(parameters: Parameters):
        """
        Print all parameters in a generic way.
        Scalars -> scientific notation
        Lists / sequences -> formatted comma-separated
        """
        parts = []
        for name, value in parameters.iterator():
            if isinstance(value, (int, float)):
                parts.append(f"{name}={value:.3e}")
            elif isinstance(value, torch.Tensor):
                if value.ndim == 0:  # scalar tensor
                    parts.append(f"{name}={value.item():.3e}")
                else:
                    raise ValueError("The Parameters.iterator() method should not return multi-dimensional tensors.")
            else:
                raise ValueError("The Parameters.iterator() method should not return non-numeric types.")
            #     else:  # tensor with multiple entries
            #         vals = ", ".join(f"{v.item():.3e}" for v in value.flatten())
            #         parts.append(f"{name}=[{vals}]")
            # elif isinstance(value, (list, tuple)):
            #     vals = ", ".join(f"{float(v):.3e}" for v in value)
            #     parts.append(f"{name}=[{vals}]")
            # else:
            #     # fallback: just str()
            #     parts.append(f"{name}={value}")
        print("Parameters: " + ", ".join(parts))

    def optimized_model(self, optimizer_type: Optional[str] = None) -> BaseBuckEstimator:
        best_parameters = self.history.get_best_parameters(optimizer_type)
        return self.model_class(param_init=best_parameters).to(self.device)

    def initialize_optimizer(self, optimizer_type: str, lr: float):
        """
        Initialize the optimizer based on the type and learning rate.
        Args:
            optimizer_type (str): Type of optimizer to use (e.g., 'Adam', 'LBFGS').
            lr (float): Learning rate for the optimizer.
        Returns:
            torch.optim.Optimizer: Initialized optimizer.
        """
        if optimizer_type == "Adam":
            return torch.optim.Adam(self.model.parameters(), lr=lr)
        elif optimizer_type == "LBFGS":
            return torch.optim.LBFGS(
                self.model.parameters(),
                lr=self.cfg.lr_lbfgs,
                max_iter=self.cfg.max_iter_lbfgs,  # inner line-search iterations
                history_size=self.cfg.history_size_lbfgs,  # critical for stability
            )
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")

    def log_results(self, it: int, loss: torch.Tensor, est: Parameters, opt: torch.optim.Optimizer):
        est = self.model.get_estimates()
        optimization_type = str(opt.__class__.__name__)

        # Collect gradients for scalar parameters
        scalar_param_names = ["L", "RL", "C", "RC", "Rdson", "Vin", "VF"]

        # Collect gradients for scalar parameters
        scalar_param_names = ["L", "RL", "C", "RC", "Rdson", "Vin", "VF"]
        grads = [
            getattr(self.model, f"log__{name}").grad.view(1)
            for name in scalar_param_names
            if getattr(self.model, f"log__{name}").grad is not None
        ]

        # Add gradients for Rloads
        for rload_param in self.model.log__Rloads:
            if rload_param.grad is not None:
                grads.append(rload_param.grad.view(1))

        # Compute gradient norm
        if grads:
            gradient_vector = torch.cat(grads)
            gradient_norm = gradient_vector.norm().item()
        else:
            gradient_norm = float("nan")  # no gradients found (shouldn't happen during training)

        # Print parameter estimates
        print(
            f"[{optimization_type}] Iteration {it}, gradient_norm {gradient_norm:4e}, loss {loss:4e},",
            end="\n",
        )
        self.print_parameters(est)

        est = self.model.get_estimates()
        # update the histories with the last Adam iteration
        self._history["optimizer"].append(optimization_type)
        self._history["loss"].append(loss.item())
        self._history["params"].append(est)
        self._history["lr"].append(opt.param_groups[0]["lr"])
        self._history["epochs"].append(it)
        self._history["callbacks"].append(self.callback_count)

    def print_summary(self):
        print(
            f"Best loss Adam [epoch: {self.history.get_best_epoch('Adam')}]: {self.history.get_best_loss('Adam'):.4e}"
        )
        print(f"[Adam] best ", end="")
        self.print_parameters(self.history.get_best_parameters("Adam"))
        print("-----------------------------------------")
        print(
            f"Best loss LBFGS [epoch: {self.history.get_best_epoch('LBFGS')}]: {self.history.get_best_loss('LBFGS'):.4e}"
        )
        print(f"[LBFGS] best ", end="")
        self.print_parameters(self.history.get_best_parameters("LBFGS"))

    def _check_update_loss_callback(
        self,
        optimizer_name: str,
        iteration: int,
        update_callback: Optional[
            Callable[[BaseBuckEstimator, MAPLoss, torch.Tensor], MAPLoss]
        ],
        update_every: Optional[int],
        X: torch.Tensor,
    ) -> None:
        """
        If it's time, call `update_callback` to refresh the MAP loss stored at `self.<loss_attr>`.
        Runs in eval() mode to ensure model forward pass is deterministic
        (BN/Dropout off), but *autograd is still enabled* so Jacobian-based updates work.
        """
        if update_callback is None or update_every is None:
            return
        if update_every <= 0 or (iteration % update_every) != 0:
            return

        was_training = self.model.training
        try:
            self.model.eval()
            new_loss_obj = update_callback(self.model, self.map_loss, X).clone()
            self.map_loss = new_loss_obj
            self.callback_count += 1
            print(f"[{optimizer_name}] vif {1/getattr(new_loss_obj, 'weight_likelihood_loss')} at iteration {iteration}")

        finally:
            if was_training:
                self.model.train()
            else:
                self.model.eval()

    def adam_fit(
        self,
        X: torch.Tensor,
        evaluate_initial_loss: bool = True,
        update_loss_callback: Optional[Callable[[BaseBuckEstimator, MAPLoss, torch.Tensor], MAPLoss]] = None,
        update_every: Optional[int] = None
    ):
        """
        Fit the model using Adam optimizer.

        Args:
            X (torch.Tensor): Input tensor of shape (B, T, 4) where B is the batch size and T is the number of transients.
            targets (Tuple[torch.Tensor, torch.Tensor]): Tuple of tensors containing the forward and backward targets.
        """
        opt = torch.optim.Adam(self.model.parameters(), lr=self.cfg.lr_adam)

        if evaluate_initial_loss:
            # Evaluate the initial loss before starting the LBFGS optimization
            initial_loss = self.evaluate_loss(
                X=X, loss_fn=self.map_loss
            )
            self.log_results(0, initial_loss, self.model.get_estimates(), opt)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            mode="min",
            factor=self.cfg.lr_reduction_factor_adam,
            patience=self.cfg.patience,
        )

        for it in range(1, self.cfg.epochs_adam + 1):
            # run callback function
            self._check_update_loss_callback(
                optimizer_name="Adam",
                iteration=it,
                update_callback=update_loss_callback,
                update_every=update_every,
                X=X,
            )
            self.model.train()
            opt.zero_grad()  # reset gradients

            targets = self.model.targets(X).to(self.device)
            preds = self.model(X)  # forward pass

            loss: torch.Tensor = self.map_loss(
                parameter_guess=self.model.logparams, preds=preds, targets=targets
            )

            loss.backward()  # backward pass

            if self.cfg.clip_gradient_adam is not None:
                # Clip gradients to prevent exploding gradients
                old_gr = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.cfg.clip_gradient_adam
                )

            # update parameters
            opt.step()
            scheduler.step(loss.item())

            # Print training progress every `save_every_adam` iterations
            if it % self.cfg.save_every_adam == 0:
                self.log_results(it, loss, self.model.get_estimates(), opt)

    def evaluate_loss(
        self,
        X: torch.Tensor,
        loss_fn: MAPLoss,
        parameter_guess: Optional[Parameters] = None,
    ) -> torch.Tensor:
        """
        Evaluate the loss function for the given inputs and targets.
        Args:
            X (torch.Tensor): Input tensor of shape (B, T, 4) where B is the batch size and T is the number of transients.
            targets (Tuple[torch.Tensor, torch.Tensor]): Tuple of tensors containing the forward and backward targets.
            optimizer (torch.optim.Optimizer): Optimizer used for training.
            loss_fn (Callable): Loss function to evaluate.
        Returns:
            torch.Tensor: The computed loss value.
        """
        X_ = X.clone().detach().to(self.device)
        if parameter_guess is not None:
            # copy the model so we don't modify the original model
            model_ = self.model_class(param_init=parameter_guess).to(self.device)
        else:
            model_ = self.model.to(self.device)

        # Remember and restore the original mode
        was_training = model_.training

        # I use try so that if anything happens I set the model back to .train()
        try:
            model_.eval()
            with torch.no_grad():
                preds = model_(X_) # forward pass
                targets = model_.targets(X_).to(self.device)
                loss = loss_fn(parameter_guess=model_.logparams, preds=preds, targets=targets)
        finally:
            # Restore original mode
            if was_training:
                model_.train()
            else:
                model_.eval()

        return loss

    def lbfgs_fit(
        self,
        X: torch.Tensor,
        evaluate_initial_loss: bool = True,
        update_loss_callback: Optional[Callable[[BaseBuckEstimator, MAPLoss, torch.Tensor], MAPLoss]] = None,
        update_every: Optional[int] = None
    ):
        """
        Fit the model using LBFGS optimizer.
        Args:
            X (torch.Tensor): Input tensor of shape (B, T, 4) where B is the batch size and T is the number of transients.
            targets (Tuple[torch.Tensor, torch.Tensor]): Tuple of tensors containing the forward and backward targets.
        """

        # Initialize LBFGS optimizer
        lbfgs_optim = torch.optim.LBFGS(
            self.model.parameters(),
            lr=self.cfg.lr_lbfgs,
            max_iter=self.cfg.max_iter_lbfgs,  # inner line-search iterations
            history_size=self.cfg.history_size_lbfgs,  # critical for stability
        )

        if evaluate_initial_loss:
            # Evaluate the initial loss before starting the LBFGS optimization
            initial_loss = self.evaluate_loss(
                X=X, loss_fn=self.map_loss
            )
            self.log_results(0, initial_loss, self.model.get_estimates(), lbfgs_optim)

        nan_abort = True  # raise RuntimeError on NaN/Inf

        # ------------------------------------------------------------------
        #  Closure with finite checks
        # ------------------------------------------------------------------
        def closure():
            self.model.train()
            lbfgs_optim.zero_grad()

            pred = self.model(X) # forward pass
            targets = self.model.targets(X).to(self.device)
            loss_val = self.map_loss(self.model.logparams, pred, targets)

            # 1)  finite-loss check
            if not torch.isfinite(loss_val):
                message = "[LBFGS] Non-finite loss encountered"
                if nan_abort:
                    raise RuntimeError(message)
                else:
                    print(message)
                    return loss_val
            # 2)  clip gradients to prevent exploding gradients
            loss_val.backward()

            return loss_val

        # ------------------------------------------------------------------
        #  LBFGS training loop
        # ------------------------------------------------------------------
        for it in range(1, self.cfg.epochs_lbfgs + 1):

            # run callback function
            self._check_update_loss_callback(
                optimizer_name="LBFGS",
                iteration=it,
                update_callback=update_loss_callback,
                update_every=update_every,
                X=X,
            )

            try:
                loss = lbfgs_optim.step(closure)
            except RuntimeError as err:
                print(f"[LBFGS] Stopped at outer iter {it}: {err}")
                break

            # 3)  post-step parameter sanity
            with torch.no_grad():
                if any(not torch.isfinite(p).all() for p in self.model.parameters()):
                    print("[LBFGS] Non-finite parameter detected — aborting.")
                    break

            if it % self.cfg.save_every_lbfgs == 0:
                self.log_results(it, loss, self.model.get_estimates(), lbfgs_optim)

    def fit(
        self,
        X: torch.Tensor,
        update_loss_callback: Optional[Callable[[BaseBuckEstimator, MAPLoss, torch.Tensor], MAPLoss]] = None,
        update_every_adam: Optional[int] = None,
        update_every_lbfgs: Optional[int] = None,
    ):

        ## fit with Adam
        self.adam_fit(X, evaluate_initial_loss=True, update_loss_callback=update_loss_callback, update_every=update_every_adam)

        # # → LBFGS
        # LBFGS optimization tends to find stable solutions that also minimize the gradient norm.
        # This will be useful when we want to compute the Laplace posterior, which relies on the Hessian of the loss function.
        self.lbfgs_fit(X, evaluate_initial_loss=True, update_loss_callback=update_loss_callback, update_every=update_every_lbfgs)

        # After training, we can save the history of losses and parameters
        print("Training concluded.")
