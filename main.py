import os
import argparse
import time
import numpy as np
import torch
from tqdm import tqdm
import onnxruntime as ort
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_auc_score
from torch.utils.tensorboard import SummaryWriter
from datasets.mvtec import MVTecDataset
from models.resnet_backbone import modified_resnet18
from utils.util import  time_string, convert_secs2time, AverageMeter
from utils.functions import cal_anomaly_maps, cal_loss
from utils.visualization import plt_fig,plt_fig2


class STPM():
    def __init__(self, args):
        self.device = args.device
        self.data_path = args.data_path
        self.obj = args.obj
        self.img_resize = args.img_resize
        self.img_cropsize = args.img_cropsize
        self.validation_ratio = args.validation_ratio
        self.num_epochs = args.num_epochs
        self.lr = args.lr
        self.batch_size = args.batch_size
        self.vis = args.vis
        self.model_dir = args.model_dir
        self.img_dir = args.img_dir
        self.checkpoint_path = os.path.join(self.model_dir, 'checkpoint.pth')
        self.onnx_path = os.path.join(self.model_dir, 'onnx')
        self.writer = SummaryWriter(log_dir=os.path.join(self.model_dir, 'logs'))  # 初始化 TensorBoard

        self.load_model()
        self.load_dataset()

        self.optimizer = torch.optim.SGD(self.model_s.parameters(), lr=self.lr, momentum=0.9, weight_decay=0.0001)
        self.start_epoch = 1  # 默认从第1轮开始
        self.best_score = None

        self.load_checkpoint()

    def load_dataset(self):
        kwargs = {'num_workers': 12, 'pin_memory': True} if torch.cuda.is_available() else {}
        train_dataset = MVTecDataset(self.data_path, class_name=self.obj, is_train=True, resize=self.img_resize, cropsize=self.img_cropsize)
        img_nums = len(train_dataset)
        valid_num = int(img_nums * self.validation_ratio)
        train_num = img_nums - valid_num
        train_data, val_data = torch.utils.data.random_split(train_dataset, [train_num, valid_num])
        self.train_loader = torch.utils.data.DataLoader(train_data, batch_size=self.batch_size, shuffle=True, **kwargs)
        self.val_loader = torch.utils.data.DataLoader(val_data, batch_size=32, shuffle=False, **kwargs)

    def load_model(self):
        self.model_t = modified_resnet18(pretrained=True).to(self.device)
        self.model_s = modified_resnet18(pretrained=False).to(self.device)
        for param in self.model_t.parameters():
            param.requires_grad = False
        self.model_t.eval()

    def load_checkpoint(self):
        if os.path.exists(self.checkpoint_path):
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
            self.model_s.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.start_epoch = checkpoint['epoch'] + 1
            self.best_score = checkpoint.get('best_score', None)
            print(f"加载断点成功: 从第 {self.start_epoch} 轮继续训练 | 最佳验证分数: {self.best_score}")
        else:
            print("没有找到断点，从头开始训练")

    def save_checkpoint(self, epoch, best_score):
        state = {
            'epoch': epoch,
            'model_state_dict': self.model_s.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_score': best_score
        }
        torch.save(state, self.checkpoint_path)
        print(f"断点已保存: 第 {epoch} 轮 | 最佳验证分数: {best_score}")

    def train(self):
        self.model_s.train()
        epoch_time = AverageMeter()
        start_time = time.time()

        for epoch in range(self.start_epoch, self.num_epochs + 1):
            need_hour, need_mins, need_secs = convert_secs2time(epoch_time.avg * ((self.num_epochs + 1) - epoch))
            print(f'{epoch:3d}/{self.num_epochs:3d} ----- [{time_string()}] [预计剩余时间: {need_hour:02d}:{need_mins:02d}:{need_secs:02d}]')

            train_losses = AverageMeter()
            for step, (data, _, _) in enumerate(tqdm(self.train_loader)):
                data = data.to(self.device)
                self.optimizer.zero_grad()

                with torch.set_grad_enabled(True):
                    features_t = self.model_t(data)
                    features_s = self.model_s(data)
                    loss = cal_loss(features_s, features_t)

                    train_losses.update(loss.sum().item(), data.size(0))
                    loss.backward()
                    self.optimizer.step()

                    # TensorBoard: 记录训练损失
                    global_step = (epoch - 1) * len(self.train_loader) + step
                    self.writer.add_scalar('Loss/Train', loss.item(), global_step)

            print(f'Train Epoch: {epoch} | Loss: {train_losses.avg:.6f}')

            val_loss = self.val(epoch)

            if self.best_score is None or val_loss < self.best_score:
                self.best_score = val_loss
                self.save_checkpoint(epoch, self.best_score)

            epoch_time.update(time.time() - start_time)
            start_time = time.time()

        print('训练结束。')

    def val(self, epoch):
        self.model_s.eval()
        val_losses = AverageMeter()

        with torch.no_grad():
            for (data, _, _) in tqdm(self.val_loader):
                data = data.to(self.device)
                features_t = self.model_t(data)
                features_s = self.model_s(data)
                loss = cal_loss(features_s, features_t)
                val_losses.update(loss.item(), data.size(0))

        print(f'Val Epoch: {epoch} | Loss: {val_losses.avg:.6f}')
        self.writer.add_scalar('Loss/Validation', val_losses.avg, epoch)  # TensorBoard: 记录验证损失
        return val_losses.avg


    def test(self):
        # 1️⃣ 模型加载
        try:
            checkpoint = torch.load(self.checkpoint_path, map_location=torch.device('cpu'))
            self.model_s.load_state_dict(checkpoint['model_state_dict'])
        except Exception as e:
            raise Exception(f'检查模型路径或文件格式: {e}')
        self.model_s.eval()

        # 2️⃣ 数据加载
        kwargs = {'num_workers': 4, 'pin_memory': True} if torch.cuda.is_available() else {}
        test_dataset = MVTecDataset(self.data_path, class_name=self.obj, is_train=False, 
                                    resize=self.img_resize, cropsize=self.img_cropsize)
        print(f"Dataset size: {len(test_dataset)}")  # 查看数据集大小
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False, **kwargs)

        scores = []
        test_imgs = []

        print('Testing')
        for (data, _, _) in tqdm(test_loader):  # 移除 label 和 mask
            test_imgs.extend(data.cpu().numpy())
            data = data.to(self.device)
            with torch.no_grad():  # 使用更简洁的上下文管理
                features_t = self.model_t(data)
                features_s = self.model_s(data)
                score = cal_anomaly_maps(features_s, features_t, self.img_cropsize)
            scores.extend(score)

        # 3️⃣ 异常分数归一化
        scores = np.asarray(scores)
        max_anomaly_score = scores.max()
        min_anomaly_score = scores.min()
        scores = (scores - min_anomaly_score) / (max_anomaly_score - min_anomaly_score + 1e-8)

        # 4️⃣ 图像级分数提取
        img_scores = scores.reshape(scores.shape[0], -1).max(axis=1)

        # 5️⃣ 可视化 (如果启用)
        if self.vis:
            dummy_masks = [np.zeros_like(scores[i]) for i in range(len(scores))]
            # 因为没有标签，gt_mask_list 和阈值传 None 或合适默认值
            plt_fig2(test_imgs, scores, img_scores, dummy_masks, 0.5, 0.5, 
                    self.img_dir, self.obj)
            
    def onnx_test(self):
        onnx_student_path = os.path.join(self.onnx_path, f"{self.obj}_model_s.onnx")
        
        # 1️⃣ 检查ONNX模型文件是否存在，如果不存在，创建ONNX模型
        if not os.path.exists(onnx_student_path):
            print(f"ONNX 学生模型未能在以下文件夹找到 {onnx_student_path}, 创建中...")
            # 加载PyTorch模型
            try:
                checkpoint = torch.load(self.checkpoint_path, map_location=torch.device('cpu'))
                self.model_s.load_state_dict(checkpoint['model_state_dict'])
            except Exception as e:
                raise Exception(f'检查模型路径或文件格式: {e}')
            self.model_s.eval()
            
            # 确保目录存在
            os.makedirs(self.onnx_path, exist_ok=True)
            
            # 创建dummy输入进行导出
            dummy_input = torch.randn(1, 3, self.img_cropsize, self.img_cropsize)
            
            # 导出ONNX模型
            torch.onnx.export(
                self.model_s,
                dummy_input,
                onnx_student_path,
                export_params=True,
                opset_version=12,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output1', 'output2', 'output3'],  # 根据模型_s的输出调整
                dynamic_axes={
                    'input': {0: 'batch_size'},
                    'output1': {0: 'batch_size'},
                    'output2': {0: 'batch_size'},
                    'output3': {0: 'batch_size'}
                }
            )
            print(f"ONNX 学生模型创建在 {onnx_student_path}")
        
        # 同样为教师模型创建ONNX (如果需要)
        onnx_teacher_path = os.path.join(self.onnx_path, f"{self.obj}_model_t.onnx")
        if not os.path.exists(onnx_teacher_path) and hasattr(self, 'model_t'):
            print(f"ONNX 老师模型未能在以下文件夹找到 {onnx_teacher_path}, 创建中...")
            dummy_input = torch.randn(1, 3, self.img_cropsize, self.img_cropsize)
            torch.onnx.export(
                self.model_t,
                dummy_input,
                onnx_teacher_path,
                export_params=True,
                opset_version=12,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output1', 'output2', 'output3'],  # 根据模型_t的输出调整
                dynamic_axes={
                    'input': {0: 'batch_size'},
                    'output1': {0: 'batch_size'},
                    'output2': {0: 'batch_size'},
                    'output3': {0: 'batch_size'}
                }
            )
            print(f"ONNX 老师模型创建在 {onnx_teacher_path}")
        
        # 2️⃣ 创建ONNX运行时会话
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        # 创建推理会话
        student_session = ort.InferenceSession(onnx_student_path, sess_options)
        teacher_session = ort.InferenceSession(onnx_teacher_path, sess_options)
        
        # 3️⃣ 数据加载
        kwargs = {'num_workers': 4, 'pin_memory': True} if torch.cuda.is_available() else {}
        test_dataset = MVTecDataset(self.data_path, class_name=self.obj, is_train=False, 
                                    resize=self.img_resize, cropsize=self.img_cropsize)
        print(f"Dataset size: {len(test_dataset)}")
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False, **kwargs)
        
        # 4️⃣ 推理
        scores = []
        test_imgs = []
        
        print('Testing with ONNX runtime')
        for (data, _, _) in tqdm(test_loader):
            test_imgs.extend(data.cpu().numpy())
            # 转换为ONNX需要的格式
            data_numpy = data.numpy()
            
            # 获取输入名称
            input_name = student_session.get_inputs()[0].name
            
            # 学生模型推理
            student_outputs = student_session.run(None, {input_name: data_numpy})
            
            # 教师模型推理
            teacher_outputs = teacher_session.run(None, {input_name: data_numpy})
            
            # 得分计算
            score = cal_anomaly_maps(student_outputs, teacher_outputs, self.img_cropsize, is_numpy=True)
            scores.extend(score)
        
        # 5️⃣ 异常分数归一化
        scores = np.asarray(scores)
        max_anomaly_score = scores.max()
        min_anomaly_score = scores.min()
        scores = (scores - min_anomaly_score) / (max_anomaly_score - min_anomaly_score + 1e-8)
        
        # 6️⃣ 图像级分数提取
        img_scores = scores.reshape(scores.shape[0], -1).max(axis=1)
        
        # 7️⃣ 可视化
        if self.vis:
            dummy_masks = [np.zeros_like(scores[i]) for i in range(len(scores))]
            plt_fig2(test_imgs, scores, img_scores, dummy_masks, 0.5, 0.5, 
                    self.img_dir, self.obj)


def get_args():
    parser = argparse.ArgumentParser(description='STPM anomaly detection')
    parser.add_argument('--phase', default='train')
    parser.add_argument("--data_path", type=str, default=r"C:\modelproject\COMPUTERVISION\anodet\anodet\data\mvtec_dataset")
    parser.add_argument('--obj', type=str, default='zipper')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--img_resize', type=int, default=256)
    parser.add_argument('--img_cropsize', type=int, default=256) # 不进行裁切
    parser.add_argument('--validation_ratio', type=float, default=0.2)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.4)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--vis', type=eval, choices=[True, False], default=True)
    parser.add_argument("--save_path", type=str, default="./mvtec_results")
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    
    print ('Available devices ', torch.cuda.device_count())
    # print ('Current cuda device ', torch.cuda.current_device())

    args = get_args()

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # device = torch.device("cpu")
    # args.device = device

    args.model_dir = args.save_path + '/models' + '/' + args.obj
    args.img_dir = args.save_path + '/imgs' + '/' + args.obj
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.img_dir, exist_ok=True)

    stpm = STPM(args)
    if args.phase == 'train':
        stpm.train()
        stpm.test()
    elif args.phase == 'test':
        stpm.test()
    elif args.phase == 'onnx':
        stpm.onnx_test()
    else:
        print('找不到处理方法')







    

