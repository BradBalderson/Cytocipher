#from .activation.act_quantify import ieg_activation

#### For cluster scoring ####
from .score_and_merge.cluster_score import get_markers, \
                                  code_enrich, coexpr_enrich, giotto_page_enrich

#### For parameter optimisation ####
from .score_and_merge.group_optimisation import optimise_k

#### For cluster merging ####
from .score_and_merge.cluster_merge import merge_clusters, merge
