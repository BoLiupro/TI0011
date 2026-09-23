"""Wire AppUsageDataset into AgentMove's real Agents/Agent/SocialWorld pipeline and run LLM-based
next-location prediction on the app_usage datasets. Mirrors agent.py's __main__ block, but
constructs AppUsageDataset (build_dataset.py) instead of processing/data.py's Foursquare-specific
Dataset, since app_usage has no venue/Nominatim data.
"""
import argparse
import os
import sys
import time
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
AGENTMOVE_ROOT = os.path.join(HERE, '..')
sys.path.insert(0, AGENTMOVE_ROOT)
sys.path.insert(0, HERE)

from build_dataset import AppUsageDataset
from agent import Agents
from models.world_model import SocialWorld
from config import PROCESSED_DIR
from utils import create_dir

random_seed_note = None  # agent.py already sets random.seed(100) on import


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--city', required=True, choices=['shanghai', 'nanchang'])
    parser.add_argument('--model_name', default='gpt4omini')
    parser.add_argument('--platform', default='OpenAI', choices=['SiliconFlow', 'OpenAI', 'DeepInfra', 'vllm', 'OpenRouter'])
    parser.add_argument('--prompt_type', default='agent_move_v6',
                         choices=['agent_move_v6', 'origin', 'llmmob', 'llmzs', 'llmmove'])
    parser.add_argument('--prompt_num', type=int, default=5)
    parser.add_argument('--traj_min_len', type=int, default=1)
    parser.add_argument('--traj_max_len', type=int, default=10)
    parser.add_argument('--max_sample_trajectories', type=int, default=3)
    parser.add_argument('--memory_lens', type=int, default=15)
    parser.add_argument('--max_neighbors', type=int, default=10)
    parser.add_argument('--max_explore_places', type=int, default=5)
    parser.add_argument('--social_info_type', default='address')
    parser.add_argument('--exp_name', default='app_usage')
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--skip_existing', action='store_true',
                         help='skip trajectories whose prediction JSON already exists (safe resume after interruption)')
    args = parser.parse_args()

    city_name = 'app_usage_%s' % args.city

    print('INFO START TIME:{}'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    print('args:{}'.format(args.__dict__))
    print('running experiment in city@{} model@{} samples:{} type:{}'.format(
        city_name, args.model_name, args.prompt_num, args.prompt_type))

    start_time = time.time()

    dataset = AppUsageDataset(args.city)

    social_save_dir = os.path.join(AGENTMOVE_ROOT, PROCESSED_DIR, city_name)
    create_dir(social_save_dir)
    social_world = SocialWorld(
        traj_dataset=dataset,
        save_dir=social_save_dir,
        city_name=city_name,
        khop=1,
        max_neighbors=args.max_neighbors,
    )

    agents = Agents(
        city_name=city_name,
        platform=args.platform,
        model_name=args.model_name,
        prompt_type=args.prompt_type,
        prompt_num=args.prompt_num,
        use_int_venue=True,
        dataset=dataset,
        workers=args.workers,
        exp_name=args.exp_name,
        traj_max_len=args.traj_max_len,
        traj_min_len=args.traj_min_len,
        social_world=social_world,
        social_info_type=args.social_info_type,
        memory_lens=args.memory_lens,
        max_explore_places=args.max_explore_places,
        max_sample_trajectories=args.max_sample_trajectories,
        skip_existing_is_on=args.skip_existing,
    )
    agents.get_predictions()

    print('results written to: {}'.format(agents.save_dir))
    print('running experiment within {} seconds'.format(int(time.time() - start_time)))


if __name__ == '__main__':
    main()
