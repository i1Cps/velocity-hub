"""Small BAM adapter for the fitted Duck Mini Pro ST3025 model."""

from dataclasses import dataclass

from bam.mjlab import BamActuator, BamActuatorCfg


class DuckMiniProBamActuator(BamActuator):
    def __init__(self, cfg, entity, target_ids, target_names) -> None:
        super().__init__(cfg, entity, target_ids, target_names)

        # The fitted bridge gain is a model parameter rather than a firmware
        # setting. Apply it once to BAM's STS-family voltage control law.
        ratio = getattr(self._bam_model, "error_gain_ratio", None)
        if ratio is not None:
            self._bam_model.actuator.error_gain *= ratio.value


@dataclass(kw_only=True)
class DuckMiniProBamActuatorCfg(BamActuatorCfg):
    def build(self, entity, target_ids, target_names) -> DuckMiniProBamActuator:
        return DuckMiniProBamActuator(self, entity, target_ids, target_names)
