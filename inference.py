import time
import torch
import yaml
from numpy.f2py.auxfuncs import throw_error

from dataset.custom_dataset import CustomImageDataset
from detectors.spsl_detector import SpslDetector


class AverageMeter(object):
    """Computes and stores the average and current value"""
    # Taken from deep learning assignment 2 code

    def __init__(self):
        self.count = 0
        self.avg = 0
        self.sum = 0
        self.val = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class Solver(object):
    def __init__(self, **kwargs):
        print('init solver')
        self.device = kwargs.pop("device", "cpu")
        self.annotation = kwargs.pop("annotation")
        self.image_dir = kwargs.pop('image_dir')

        self.test_loader = torch.utils.data.DataLoader(
            CustomImageDataset(self.annotation, self.image_dir), batch_size=100, shuffle=False, num_workers=2
        )

        self.model_type = kwargs.pop("model_type")
        self.model = None
        if self.model_type == "spsl":
            with open('./config/spsl.yaml', 'r') as f:
                config = yaml.safe_load(f)
            config['device'] = self.device
            self.model = SpslDetector(config)
        self.model = self.model.to(self.device)


    def inference(self):
        iter_time = AverageMeter()
        losses = AverageMeter()
        metric = AverageMeter()

        num_class = 2
        cm = torch.zeros(num_class, num_class)
        self.model.eval()
        with torch.no_grad():
            # evaluation loop
            for idx, (data, target) in enumerate(self.test_loader):
                start = time.time()
                data = data.to(self.device)
                target = target.to(self.device)
                data_dict = {'image': data, 'label': target}
                result = self.model.forward(data_dict)
                print(result['cls'])
                print(result['prob'])

                # out, loss = self._compute_loss_update_params(data, target)
                #
                # # CHange to AUC
                # batch_acc = self._check_accuracy(out, target)
                #
                # # update confusion matrix
                # _, preds = torch.max(out, 1)
                # for t, p in zip(target.view(-1), preds.view(-1)):
                #     cm[t.long(), p.long()] += 1

                # losses.update(loss.item(), out.shape[0])
                # metric.update(batch_acc, out.shape[0])
                iter_time.update(time.time() - start)

            # cm = cm / cm.sum(1)
            # per_cls_acc = cm.diag().detach().numpy().tolist()
            # for i, acc_i in enumerate(per_cls_acc):
            #     print("Accuracy of Class {}: {:.4f}".format(i, acc_i))
            #
            # print("* Prec @1: {top1.avg:.4f}".format(top1=acc))
            # return metric.avg, cm



def main():
    # load configuration
    # load dataset
    # run inference

    kwargs = {}
    kwargs['image_dir'] = 'dataset/midjourney'
    kwargs['annotation'] = 'dataset/midjourney/annotations_small.csv'
    kwargs['model_type'] = 'spsl'
    Solver(**kwargs).inference()

if __name__ == '__main__':

    # with open(config_file, "r") as read_file:
    #     config = yaml.safe_load(read_file)
    # for key in config:
    #     for k, v in config[key].items():
    #         if k != 'description':
    #             kwargs[k] = v
    main()