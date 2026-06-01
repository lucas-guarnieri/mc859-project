import yaml
from src.models.popularity          import PopularityRecommender
from src.models.cooccurrence        import CooccurrenceRecommender
from src.models.item_cf             import ItemCFRecommender
from src.models.pagerank            import PersonalizedPageRankRecommender
from src.models.multi_hop           import MultiHopRecommender
from src.models.multi_hop_penalized import MultiHopPenalizedRecommender
 
 
def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)
 
 
def build_models(config: dict) -> dict:
    """
    Instantiate all models defined in the config.
    Returns a dict {model_name: model_instance}.
    """
    cfg_models = config["models"]
    cfg_exp    = config["experiment"]
 
    models = {}
 
    # baseline models
    models["popularity"]   = PopularityRecommender()
    models["cooccurrence"] = CooccurrenceRecommender()
    models["item_cf"]      = ItemCFRecommender(
        top_n_similar=cfg_models["item_cf"]["top_n_similar"]
    )
    models["pagerank"]     = PersonalizedPageRankRecommender(
        alpha=cfg_models["pagerank"]["alpha"]
    )
 
    # multi-hop models
    for mode in cfg_models["multi_hop"]["modes"]:
        m = MultiHopRecommender(mode=mode)
        models[m.name] = m
 
    # multi-hop penalized models
    for penalty in cfg_models["multi_hop_penalized"]["penalties"]:
        for mode in cfg_models["multi_hop_penalized"]["modes"]:
            m = MultiHopPenalizedRecommender(mode=mode, penalty=penalty)
            models[m.name] = m
 
    return models