from stable_baselines3.common.env_checker import check_env
from street_fighter_custom_wrapper import StreetFighterCustomWrapper
import retro
env = retro.make(
            game="StreetFighterIISpecialChampionEdition-Genesis", 
            state="Champion.Level12.RyuVsBison", 
            use_restricted_actions=retro.Actions.FILTERED, 
            obs_type=retro.Observations.IMAGE    
        )
env.initial_state
env = StreetFighterCustomWrapper(env)

check_env(env, warn=True)