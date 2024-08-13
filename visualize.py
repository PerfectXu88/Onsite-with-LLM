from utils.visualizer import Visualizer

if __name__ == '__main__':
    result_path = r"E:\onsite\自己的数据\0320follow290\REPLAY_0_0320follow290_result.csv"
    save_path = r"E:\onsite\自己的数据\0320follow290\results.gif"
    # xodr_path = r"E:\onsite\onsite-llm-cheap\scenario\replay\0110follow103\0110follow103.xodr"

    vis = Visualizer()
    vis.replay_result(result_path=result_path, save_path=save_path)
    # vis.show_task(mode='SERIAL', task='Cyz_TJST_1')
    # vis.show_map(xodr_path=xodr_path)