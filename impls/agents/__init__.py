from agents.crl import CRLAgent
from agents.gcbc import GCBCAgent
from agents.gciql import GCIQLAgent
from agents.gciql_ota import GCIQLOTAAgent
from agents.gcivl import GCIVLAgent
from agents.hiql import HIQLAgent
from agents.qrl import QRLAgent
from agents.mc_occ import MCOccAgent
from agents.rew import RewardAgent
from agents.sac import SACAgent
agents = dict(
    crl=CRLAgent,
    gcbc=GCBCAgent,
    gciql=GCIQLAgent,
    gcivl=GCIVLAgent,
    gciql_ota=GCIQLOTAAgent,
    hiql=HIQLAgent,
    mc_occ=MCOccAgent,
    qrl=QRLAgent,
    rew=RewardAgent,
    sac=SACAgent,
    
)
