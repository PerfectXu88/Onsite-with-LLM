from utils.visualizer import Visualizer

if __name__ == '__main__':
    result_path = r"E:\onsite\onsite-structured-test\outputs\REPLAY_0_0110follow103_result.csv"
    save_path = r".\outputs\REPLAY.gif"
    xodr_path = None

    vis = Visualizer()
    vis.replay_result(result_path=result_path, save_path=save_path)
    # vis.show_task(mode='SERIAL', task='Cyz_TJST_1')
    # vis.show_map(xodr_path=xodr_path)