from pidmppo.config import apply_overrides, load_config
from pidmppo.experiment import apply_variant


def test_paper_config_has_exact_observation_dimension():
    config = load_config("configs/paper.yaml")
    assert config.env.observation_dim == 85
    assert config.ppo.sequence_length == 128
    assert config.ppo.failure_tail_steps == 20


def test_matched_and_single_auxiliary_variants_are_independent():
    base = load_config("configs/paper.yaml")
    matched = apply_variant(base, "matched_lstm")
    assert matched.model.encoder == "lstm"
    assert matched.model.twin_critic and matched.model.enable_l_head and matched.model.enable_g_head
    l_only = apply_variant(base, "l_loss_only")
    assert l_only.model.enable_l_head and not l_only.model.enable_g_head
    changed = apply_overrides(base, ["model.memory_bound=0.5", "ppo.auxiliary_coef=0.05"])
    assert changed.model.memory_bound == 0.5
    assert changed.ppo.auxiliary_coef == 0.05


def test_lstm_matched_baseline_factorial_variants():
    base = load_config("configs/paper.yaml")
    auxiliary = apply_variant(base, "lstm_auxiliary")
    dual = apply_variant(base, "lstm_dual_value")
    matched = apply_variant(base, "matched_lstm")
    assert auxiliary.model.encoder == "lstm"
    assert auxiliary.model.auxiliary_heads and not auxiliary.model.twin_critic
    assert dual.model.twin_critic and not dual.model.auxiliary_heads
    assert matched.model.twin_critic and matched.model.auxiliary_heads
