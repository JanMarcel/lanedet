from lanedet.detect import *
import os
# print('\n'.join(os.environ['PATH']))
if __name__ == "__main__":
    home_path = os.path.expanduser("~")
    cl_args_mobilenetv2_tusimple = [
        home_path + "/dev/lanedet/configs/laneatt/mobilenetv2_tusimple_100epochs_maxlanes2.py",
        "--load_from",
        home_path + "/dev/lanedet/models/mobile_tusimple_50epochs.pth"
    ]
    
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='The path of config file')
    parser.add_argument('--show', action='store_true', help='Whether to show the image')
    parser.add_argument('--savedir', type=str, default=None, help='The root of save directory')
    parser.add_argument('--load_from', type=str, default='best.pth', help='The path of model')
    args = parser.parse_args(cl_args_mobilenetv2_tusimple)
    cfg = Config.fromfile(args.config)
    cfg.load_from = args.load_from
    processes = Process(cfg.val_process, cfg)
    net = build_net(cfg)
    net = torch.nn.parallel.DataParallel(net, device_ids = range(1)).cuda()
    net.eval()
    load_network(net, cfg.load_from)
    
    ori_cv_img = cv2.imread("vis/bla/test_0.jpg")
    img = ori_cv_img[cfg.cut_height:, :, :].astype(np.float32)

    # preprocessing
    data = {'img' : img, 'lanes' : []}
    data = processes(data)
    data['img'] = data['img'].unsqueeze(0).cuda()

    net_args = (data,)  # args fpor conversion. Must be structured for net(*net_args)
    fp = "models/mobilenet_50epochs_new_method.onnx"
    with torch.no_grad():
        torch.onnx.export(net.module, (data, {}), fp, verbose=True, opset_version=13)

    input()