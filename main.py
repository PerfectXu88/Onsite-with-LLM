import os
import yaml
import time
# import TessNG
import OnSiteReplay

from utils.ScenarioManager import select_scenario_manager
from utils.logger import logger
from planner import PLANNER

from dilu.scenario.envScenario import EnvScenario
from dilu.driver_agent.driverAgent import DriverAgent
from rich import print

def setup_env(config):
    if config['OPENAI_API_TYPE'] == 'openai':
        os.environ["OPENAI_API_TYPE"] = config['OPENAI_API_TYPE']
        os.environ["OPENAI_API_KEY"] = config['OPENAI_KEY']
        os.environ["OPENAI_CHAT_MODEL"] = config['OPENAI_CHAT_MODEL']
        proxy_url = config['PROXY_URL']
        proxy_port = config['PROXY_PORT']
        os.environ["http_proxy"] = f'{proxy_url}:{proxy_port}'
        os.environ["https_proxy"] = f'{proxy_url}:{proxy_port}'
    else:
        raise ValueError("Unknown OPENAI_API_TYPE, should be openai")


def main():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    # 网络代理设置
    config = yaml.load(open('./config/config.yaml'), Loader=yaml.FullLoader)
    setup_env(config)

    with open('./config/tasks.yaml', 'r') as f:
        tasks = yaml.safe_load(f)
    for mode, config in tasks.items():
        if mode != 'REPLAY':
            if not os.path.exists(os.path.join(BASE_DIR, 'TessNG', 'WorkSpace', 'Cert', '_cert')):
                pass
                # TessNG.run(mode, {})
        scenario_manager = select_scenario_manager(mode, config)
        while scenario_manager.next():
            try:
                # dilu模块初始化设置
                sce = EnvScenario()
                DA = DriverAgent(sce, verbose=True)
                print('DA Finished')
                tic = time.time()
                if mode == 'REPLAY':
                    scenario_manager.cur_scene.task_info['dt'] = 0.04
                    OnSiteReplay.run(config, PLANNER(sce, DA), scene_info=scenario_manager.cur_scene)
                else:
                    pass
                    # TessNG.run(mode, config, PLANNER(), scene_info=scenario_manager.cur_scene)
                toc = time.time()
                if os.path.exists(scenario_manager.cur_scene.output_path):
                    logger.info(f"[{mode:8s}-{scenario_manager.cur_scene_num+1:03d}/{len(scenario_manager.tasks):03d}] <{scenario_manager.cur_scene.name}> Test finished in {round(toc - tic, 1)}s.")
                else:
                    logger.error(f"[{mode:8s}-{scenario_manager.cur_scene_num+1:03d}/{len(scenario_manager.tasks):03d}] <{scenario_manager.cur_scene.name}> Cannot locate correct output file!")
            except Exception as e:
                logger.critical(f"[{mode:8s}-{scenario_manager.cur_scene_num+1:03d}/{len(scenario_manager.tasks):03d}] <{scenario_manager.cur_scene.name}> Test Collapse with error: {repr(e)}.")
            break



if __name__ == '__main__':
    main()