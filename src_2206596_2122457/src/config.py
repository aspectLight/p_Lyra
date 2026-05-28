from dataclasses import dataclass

NUM_SIMULATIONS = 400
EXPLORATION_CONSTANT = 1.0   
GAMMA_PRUNING_THRESHOLD = 0.0001
PROGRESSIVE_BIAS_CONSTANT = 2.47

RAVE_PRIOR_COUNT = 8
RAVE_PRIOR_VALUE = 0.5
RAVE_WEIGHT_INITIAL = 2.12
RAVE_WEIGHT_FINAL = 830.0
RAVE_RANDOMIZE_FREQ = 30

# ICE Configuration
USE_ICE_PRUNING = True
ICE_FIND_PRESIMPLICIAL_PAIRS = True
ICE_FIND_ALL_PATTERN_KILLERS = True
ICE_FIND_ALL_PATTERN_SUPERIORS = True
ICE_FIND_THREE_SIDED_DEAD_REGIONS = True
ICE_ITERATIVE_DEAD_REGIONS = False
ICE_USE_CAPTURE = True
ICE_FIND_REVERSIBLE = True
ICE_USE_S_REVERSIBLE_AS_REVERSIBLE = True

@dataclass
class MCTSConfig:
    num_simulations: int = NUM_SIMULATIONS
    exploration_constant: float = EXPLORATION_CONSTANT
    use_rave: bool = True
    use_tree_reuse: bool = True    
    use_pattern_playout: bool = True
    use_pattern_priors: bool = True
    rave_prior_count: int = RAVE_PRIOR_COUNT
    rave_prior_value: float = RAVE_PRIOR_VALUE
    rave_weight_final: float = RAVE_WEIGHT_FINAL
    rave_randomize_freq: int = RAVE_RANDOMIZE_FREQ 
    gamma_pruning_threshold: float = GAMMA_PRUNING_THRESHOLD
    progressive_bias_constant: float = PROGRESSIVE_BIAS_CONSTANT
    
    use_ice_pruning: bool = USE_ICE_PRUNING
    ice_find_presimplicial_pairs: bool = ICE_FIND_PRESIMPLICIAL_PAIRS
    ice_find_all_pattern_killers: bool = ICE_FIND_ALL_PATTERN_KILLERS
    ice_find_all_pattern_superiors: bool = ICE_FIND_ALL_PATTERN_SUPERIORS
    ice_find_three_sided_dead_regions: bool = ICE_FIND_THREE_SIDED_DEAD_REGIONS
    ice_iterative_dead_regions: bool = ICE_ITERATIVE_DEAD_REGIONS
    ice_use_capture: bool = ICE_USE_CAPTURE
    ice_find_reversible: bool = ICE_FIND_REVERSIBLE
    ice_use_s_reversible_as_reversible: bool = ICE_USE_S_REVERSIBLE_AS_REVERSIBLE
    
    knowledge_threshold: int = 256
    
    use_vcs: bool = True
    vc_and_over_edge: bool = True
    vc_use_patterns: bool = True
    vc_use_non_edge_patterns: bool = True
    vc_incremental_builds: bool = True
    vc_limit_fulls: bool = False
    vc_limit_or: bool = False
    
    debug_enable: bool = False
    debug_root_only: bool = False
    debug_sample_every: int = 1000
    debug_top_k: int = 10
    detailed_iteration_logging: bool = False
    log_dir: str = "logs"