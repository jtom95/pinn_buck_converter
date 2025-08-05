from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Tuple, List, Optional

import torch

from ..io_model import TrainingHistory
from ..config import Parameters

from .model_param_estimator import BaseBuckEstimator
from .loss_function_configs import MAPLossFunction


@dataclass
class TrainingConfigs:
    savename: str = "saved_run"
    out_dir: Path = Path(".")
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
        loss_fn: MAPLossFunction,
        cfg: TrainingConfigs = TrainingConfigs(),
        lbfgs_loss_fn: Optional[MAPLossFunction] = None,
        device="cpu",
    ):
        self.model = model.to(device)
        self.model_class = model.__class__
        self.loss_fn = loss_fn
        self.lbfgs_loss_fn = lbfgs_loss_fn if lbfgs_loss_fn is not None else loss_fn
        self.cfg = cfg
        self.device = device
        self._history = {"loss": [], "params": [], "lr": [], "optimizer": [], "epochs": []}

    @property
    def history(self) -> TrainingHistory:
        return TrainingHistory.from_histories(
            loss_history=self._history["loss"],
            param_history=self._history["params"],
            optimizer_history=self._history["optimizer"],
            learning_rate=self._history["lr"],
            epochs=self._history["epochs"],
        )

    @staticmethod
    def print_parameters(parameters: Parameters):
        print(
            f"Parameters: L={parameters.L:.3e}, RL={parameters.RL:.3e}, C={parameters.C:.3e}, ",
            f"RC={parameters.RC:.3e}, Rdson={parameters.Rdson:.3e}, ",
            f"Rloads=[{', '.join(f'{r:.3e}' for r in parameters.Rloads)}], ",
            f"Vin={parameters.Vin:.3f}, VF={parameters.VF:.3e}",
        )
        
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
            getattr(self.model, f"log_{name}").grad.view(1)
            for name in scalar_param_names
            if getattr(self.model, f"log_{name}").grad is not None
        ]

        # Add gradients for Rloads
        for rload_param in self.model.log_Rloads:
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

    def adam_fit(self, X: torch.Tensor, targets: Tuple[torch.Tensor, torch.Tensor]):
        """
        Fit the model using Adam optimizer.

        Args:
            X (torch.Tensor): Input tensor of shape (B, T, 4) where B is the batch size and T is the number of transients.
            targets (Tuple[torch.Tensor, torch.Tensor]): Tuple of tensors containing the forward and backward targets.
        """
        opt = torch.optim.Adam(self.model.parameters(), lr=self.cfg.lr_adam)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            mode="min",
            factor=self.cfg.lr_reduction_factor_adam,
            patience=self.cfg.patience,
        )

        for it in range(1, self.cfg.epochs_adam + 1):
            opt.zero_grad()  # reset gradients

            preds = self.model(X)  # forward pass

            loss: torch.Tensor = self.loss_fn(
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
        loss_fn: Callable[
            [Parameters, Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]],
            torch.Tensor,
        ],
        targets: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
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
        if targets is None:
            targets = (X_[1:, :, :2].clone().detach(), X_[:-1, :, :2].clone().detach())
        if parameter_guess is not None:
            # copy the model so we don't modify the original model
            model_ = self.model_class(param_init=parameter_guess).to(self.device)
        else:
            model_ = self.model
        
        preds = model_(X_)
        loss = loss_fn(parameter_guess=model_.logparams, preds=preds, targets=targets)
        return loss

    def lbfgs_fit(
        self,
        X: torch.Tensor,
        targets: Tuple[torch.Tensor, torch.Tensor],
        evaluate_initial_loss: bool = True,
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
                X=X, targets=targets, loss_fn=self.lbfgs_loss_fn
            )
            self.log_results(0, initial_loss, self.model.get_estimates(), lbfgs_optim)

        nan_abort = True  # raise RuntimeError on NaN/Inf

        # ------------------------------------------------------------------
        #  Closure with finite checks
        # ------------------------------------------------------------------
        def closure():
            lbfgs_optim.zero_grad()

            pred = self.model(X)
            loss_val = self.lbfgs_loss_fn(self.model.logparams, pred, targets)

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

    def fit(self, X):
        X = X.detach().to(self.device)
        targets = X[1:, :, :2].clone().detach(), X[:-1, :, :2].clone().detach()

        self.adam_fit(X, targets)

        # # → LBFGS
        # LBFGS optimization tends to find stable solutions that also minimize the gradient norm.
        # This will be useful when we want to compute the Laplace posterior, which relies on the Hessian of the loss function.
        self.lbfgs_fit(X, targets, evaluate_initial_loss=True)

        # After training, we can save the history of losses and parameters
        print("Training concluded.")
        self.print_summary()

    def save_history_to_csv(self):
        # generate the output directory if it doesn't exist
        self.cfg.out_dir.mkdir(parents=True, exist_ok=True)

        # if savename doesn't end with .csv, add it
        savename = self.cfg.savename

        if not savename.endswith(".csv"):
            savename += ".csv"
        self.history.save_to_csv(self.cfg.out_dir / savename)
        print("Saved history to CSV file:", self.cfg.out_dir / savename)
