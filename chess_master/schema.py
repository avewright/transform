"""Arrow schemas and label-contract constants for chess_master_v1.

Absent metadata is null / "unknown". Never coerce missing values to 0 or "verified".
"""
from __future__ import annotations

import pyarrow as pa

DATASET_VERSION = "chess_master_v1"
VOCAB_ADAPTER_VERSION = "compact1968"
UNKNOWN = "unknown"

SOURCE_LICHESS = 1
SOURCE_SYZYGY = 2
SOURCE_MISTAKE = 3
SOURCE_SF19 = 4
SOURCE_PUZZLE = 5
SOURCE_IDS = {
    "lichess": SOURCE_LICHESS,
    "syzygy": SOURCE_SYZYGY,
    "swa_mistakes": SOURCE_MISTAKE,
    "sf19": SOURCE_SF19,
    "puzzles": SOURCE_PUZZLE,
}

ANNOTATION_ENGINE_POLICY = "engine_policy"
ANNOTATION_PUZZLE = "puzzle_solution"
ANNOTATION_TABLEBASE = "tablebase"
ANNOTATION_MODEL = "model_prediction"

PERSPECTIVE_WHITE = "white_absolute"
PERSPECTIVE_STM = "side_to_move"
PERSPECTIVE_UNKNOWN = UNKNOWN

POLICY_PROBS = "probabilities"
POLICY_ONEHOT = "onehot_solution"
POLICY_TB_SOFTMAX = "softmax_tb_wdl_mapped_cp"
POLICY_UNKNOWN = "source_probabilities_unknown_transform"

PHASE_METHOD = "non_king_20_10"
PHASE_NAME = {0: "opening", 1: "middlegame", 2: "endgame"}

# Depth sentinels are not search depth.
LICHESS_DEPTH_SENTINEL_MIN = 128
SYZYGY_DEPTH_SENTINEL = 999
SF19_IMPLAUSIBLE_DEPTH = 64
LICHESS_CP_MATE_SENTINEL = 90_000
LICHESS_MIN_DEPTH = 22
SF19_MIN_BUDGET = 100_000
SF19_MIN_DEPTH = 12

TB_WDL_VALUES = frozenset({-2, -1, 0, 1, 2})

POSITIONS_SCHEMA = pa.schema([
    ("position_id", pa.string()),
    ("legacy_hash", pa.uint64()),
    ("equivalence_key", pa.string()),
    ("equivalence_method", pa.string()),
    ("fen_4", pa.string()),
    ("fen_6", pa.string()),
    ("board_array", pa.list_(pa.int8(), 64)),
    ("turn", pa.int8()),
    ("castling", pa.int8()),
    ("ep_square", pa.int8()),
    ("halfmove", pa.int32()),
    ("fullmove", pa.int32()),
    ("rule_state_available", pa.bool_()),
    ("piece_count", pa.int8()),
    ("non_king_count", pa.int8()),
    ("mat_wp", pa.int8()),
    ("mat_wn", pa.int8()),
    ("mat_wb", pa.int8()),
    ("mat_wr", pa.int8()),
    ("mat_wq", pa.int8()),
    ("mat_bp", pa.int8()),
    ("mat_bn", pa.int8()),
    ("mat_bb", pa.int8()),
    ("mat_br", pa.int8()),
    ("mat_bq", pa.int8()),
    ("phase", pa.int8()),
    ("phase_method", pa.string()),
    ("in_check", pa.bool_()),
    ("legality_status", pa.string()),
    ("opening_eco", pa.string()),
    ("opening_name", pa.string()),
])

ANNOTATIONS_SCHEMA = pa.schema([
    ("annotation_id", pa.string()),
    ("position_id", pa.string()),
    ("search_record_id", pa.string()),
    ("annotation_type", pa.string()),
    ("source_name", pa.string()),
    ("source_revision", pa.string()),
    ("source_path", pa.string()),
    ("source_row", pa.int64()),
    ("source_license", pa.string()),
    ("engine", pa.string()),
    ("engine_version", pa.string()),
    ("network", pa.string()),
    ("nodes_requested", pa.int64()),
    ("nodes_achieved", pa.int64()),
    ("depth", pa.int32()),
    ("depth_is_sentinel", pa.bool_()),
    ("depth_sentinel_reason", pa.string()),
    ("multipv", pa.int16()),
    ("bound_skipped", pa.int16()),
    ("flags", pa.int16()),
    ("incomplete_search", pa.bool_()),
    ("policy_kind", pa.string()),
    ("policy_transform", pa.string()),
    ("tau", pa.float32()),
    ("vocab_adapter", pa.string()),
    ("best_uci", pa.string()),
    ("policy_uci", pa.list_(pa.string())),
    ("policy_indices", pa.list_(pa.int64())),
    ("policy_probs", pa.list_(pa.float32())),
    ("policy_cps", pa.list_(pa.int32())),
    ("policy_mates", pa.list_(pa.int32())),
    ("trainer_soft_indices", pa.list_(pa.int64(), 8)),
    ("trainer_soft_probs", pa.list_(pa.float32(), 8)),
    ("value_cp", pa.int32()),
    ("value_mate", pa.int32()),
    ("value_wdl", pa.list_(pa.float32())),
    ("value_perspective", pa.string()),
    ("value_valid", pa.int8()),
    ("value_eligible", pa.int8()),
    ("policy_eligible", pa.int8()),
    ("original_cp", pa.int32()),
    ("original_mate", pa.int32()),
    ("original_wdl_scalar", pa.int32()),
    ("original_perspective", pa.string()),
    ("mate_is_dtz_proxy", pa.int8()),
    ("lichess_cp_mate_sentinel", pa.int8()),
    ("puzzle_id", pa.string()),
    ("puzzle_game_id", pa.string()),
    ("puzzle_rating", pa.int32()),
    ("puzzle_rating_deviation", pa.int32()),
    ("puzzle_themes", pa.list_(pa.string())),
    ("puzzle_opening_tags", pa.list_(pa.string())),
    ("puzzle_moves", pa.string()),
    ("puzzle_setup_fen", pa.string()),
    ("puzzle_solver_fen", pa.string()),
    ("tb_wdl", pa.int8()),
    ("tb_dtz", pa.int32()),
    ("tb_n_pieces", pa.int8()),
    ("tb_rule", pa.string()),
    ("tb_probe", pa.string()),
    ("model_checkpoint", pa.string()),
    ("model_checkpoint_hash", pa.string()),
    ("model_move_uci", pa.string()),
    ("model_in_pv", pa.int8()),
    ("model_tag", pa.string()),
    ("model_drop_cp", pa.int32()),
    ("regret_cp", pa.int32()),
    ("regret_status", pa.string()),
    ("quality_status", pa.string()),
    ("reject_reason", pa.string()),
    ("source_phase", pa.int8()),
    ("source_fields_json", pa.string()),
])

MEMBERSHIP_SCHEMA = pa.schema([
    ("membership_id", pa.string()),
    ("position_id", pa.string()),
    ("annotation_id", pa.string()),
    ("legacy_hash", pa.uint64()),
    ("equivalence_key", pa.string()),
    ("pool_name", pa.string()),
    ("split", pa.string()),
    ("game_id_raw", pa.string()),
    ("game_id_ns", pa.string()),
    ("group_key", pa.string()),
    ("included_in_pool", pa.bool_()),
    ("actually_sampled", pa.string()),
    ("exposure_certainty", pa.string()),
    ("checkpoint_lineage", pa.string()),
    ("mix_row", pa.int64()),
])

QUARANTINE_SCHEMA = pa.schema([
    ("source_name", pa.string()),
    ("source_path", pa.string()),
    ("source_row", pa.int64()),
    ("reason", pa.string()),
    ("position_id", pa.string()),
    ("detail", pa.string()),
])
