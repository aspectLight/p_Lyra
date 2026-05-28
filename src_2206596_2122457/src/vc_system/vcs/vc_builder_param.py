from dataclasses import dataclass


@dataclass
class VCBuilderParam:
    and_over_edge: bool = False
    use_patterns: bool = True
    use_non_edge_patterns: bool = True
    incremental_builds: bool = True
    limit_fulls: bool = True
    limit_or: bool = True

