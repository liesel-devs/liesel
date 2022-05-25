"""
# Kernel sequence
"""

from dataclasses import dataclass
from typing import Sequence

import jax

from .epoch import EpochState
from .pytree import register_dataclass_as_pytree
from .types import (
    Kernel,
    KernelState,
    KeyArray,
    ModelState,
    Position,
    TransitionInfo,
    TuningInfo,
)

KernelStates = list[KernelState]
TuningInfos = dict[str, TuningInfo]
TransitionInfos = dict[str, TransitionInfo]


@register_dataclass_as_pytree
@dataclass
class KerSeqTransitionOutput:
    model_state: ModelState
    kernel_states: KernelStates
    infos: TransitionInfos


@register_dataclass_as_pytree
@dataclass
class KerSeqTuningOutput:
    kernel_states: KernelStates
    infos: TuningInfos


@register_dataclass_as_pytree
@dataclass
class KerSeqFinalizeWarmupOutput:
    kernel_states: KernelStates
    error_codes: dict[str, int]


class KernelSequence:
    def __init__(self, kernels: Sequence[Kernel]) -> None:
        identifiers = set()
        for ker in kernels:
            if not ker.identifier:
                raise RuntimeError(
                    f"Kernel identifier must be a non-empty string. "
                    f"The field is empty in {ker!r}."
                )
            identifiers.add(ker.identifier)

        if len(identifiers) < len(kernels):
            raise RuntimeError("Kernel identifier must be unique.")

        self._kernels = list(kernels)

    def get_kernels(self):
        return self._kernels

    def init_states(self, prng_key: KeyArray, model_state: ModelState) -> KernelStates:
        keys = jax.random.split(prng_key, len(self._kernels))
        states = [
            kernel.init_state(keys[i], model_state)
            for i, kernel in enumerate(self._kernels)
        ]
        return states

    def start_epoch(
        self,
        prng_key: KeyArray,
        kernel_states: KernelStates,
        model_state: ModelState,
        epoch: EpochState,
    ) -> KernelStates:
        keys = jax.random.split(prng_key, len(self._kernels))

        states = [
            kernel.start_epoch(keys[i], kernel_states[i], model_state, epoch)
            for i, kernel in enumerate(self._kernels)
        ]
        return states

    def end_epoch(
        self,
        prng_key: KeyArray,
        kernel_states: KernelStates,
        model_state: ModelState,
        epoch: EpochState,
    ) -> KernelStates:
        keys = jax.random.split(prng_key, len(self._kernels))

        states = [
            kernel.end_epoch(keys[i], kernel_states[i], model_state, epoch)
            for i, kernel in enumerate(self._kernels)
        ]
        return states

    def transition(
        self,
        prng_key: KeyArray,
        kernel_states: KernelStates,
        model_state: ModelState,
        epoch: EpochState,
    ) -> KerSeqTransitionOutput:
        keys = jax.random.split(prng_key, len(self._kernels))
        infos: TransitionInfos = {}
        kstates: KernelStates = []

        for i, kernel in enumerate(self._kernels):
            result = kernel.transition(keys[i], kernel_states[i], model_state, epoch)
            model_state = result.model_state
            kstates.append(result.kernel_state)
            infos[kernel.identifier] = result.info

        return KerSeqTransitionOutput(
            model_state=model_state, kernel_states=kstates, infos=infos
        )

    def tune(
        self,
        prng_key: KeyArray,
        kernel_states: KernelStates,
        model_state: ModelState,
        phase: EpochState,
        history: Position | None,
    ) -> KerSeqTuningOutput:
        keys = jax.random.split(prng_key, len(self._kernels))
        infos: TuningInfos = {}
        kstates = []

        # time = phase.time_before_epoch + phase.time_in_epoch

        for i, kernel in enumerate(self._kernels):
            result = kernel.tune(keys[i], kernel_states[i], model_state, phase, history)
            kstates.append(result.kernel_state)
            infos[kernel.identifier] = result.info

        return KerSeqTuningOutput(kernel_states=kstates, infos=infos)

    def end_warmup(
        self,
        prng_key: KeyArray,
        kernel_states: KernelStates,
        model_state: ModelState,
        tuning_history: None | TuningInfos,
    ) -> KerSeqFinalizeWarmupOutput:
        keys = jax.random.split(prng_key, len(self._kernels))
        new_states = []
        error_codes: dict[str, int] = {}

        for i, kernel in enumerate(self._kernels):
            if tuning_history is None:
                th = None
            else:
                th = tuning_history[kernel.identifier]

            result = kernel.end_warmup(
                keys[i],
                kernel_states[i],
                model_state,
                th,
            )
            new_states.append(result.kernel_state)
            error_codes[kernel.identifier] = result.error_code

        return KerSeqFinalizeWarmupOutput(new_states, error_codes)
