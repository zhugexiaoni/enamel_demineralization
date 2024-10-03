# import os
# import random
# import cv2
# import numpy as np
# import torch
#
# from ultralytics.nn.autobackend import AutoBackend
# from ultralytics.utils import ops
# from ultralytics.utils.plotting import colors
#
#
# class YOLOV8SegmentInfer:
#     def __init__(self, weights, cuda, conf_thres, iou_thres) -> None:
#         self.imgsz = 640
#         self.device = cuda
#         self.model = AutoBackend(weights, device=torch.device(cuda))
#         self.model.eval()
#         self.names = self.model.names
#         self.half = False
#         self.conf = conf_thres
#         self.iou = iou_thres
#         self.color = {"font": (255, 255, 255)}
#         self.color.update(
#             {self.names[i]: (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
#              for i in range(len(self.names))})
#
#     def infer(self, img_src, save_path):
#         img = self.precess_image(img_src)
#         preds = self.model(img)  # shape [1, 116, 6300]
#         det = ops.non_max_suppression(preds[0], self.conf, self.iou, classes=None, agnostic=False, max_det=300,
#                                       nc=len(self.names))
#
#         proto = preds[1][-1]
#         for i, pred in enumerate(det):
#             lw = max(round(sum(img_src.shape) / 2 * 0.003), 2)  # line width
#             tf = max(lw - 1, 1)  # font thickness
#             sf = lw / 3  # font scale
#
#             # 6, 640, 480
#             masks = ops.process_mask(proto[i], pred[:, 6:], pred[:, :4], img.shape[2:], upsample=True)  # HWC
#
#             pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], img_src.shape)
#             pred_bbox = pred[:, :6].cpu().detach().numpy()
#
#             self.draw_masks(img_src, masks.data, colors=[colors(x, True) for x in pred[:, 5]], im_gpu=img.squeeze(0))
#
#             for bbox in pred_bbox:
#                 self.draw_box(img_src, bbox[:4], bbox[4], self.names[bbox[5]], lw, sf, tf)
#
#         cv2.imwrite(os.path.join(save_path, os.path.split(img_path)[-1]), img_src)
#
#     def draw_box(self, img_src, box, conf, cls_name, lw, sf, tf):
#         color = self.color[cls_name]
#         label = f'{cls_name} {conf}'
#         p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
#         # 绘制矩形框
#         cv2.rectangle(img_src, p1, p2, color, thickness=lw, lineType=cv2.LINE_AA)
#         # text width, height
#         w, h = cv2.getTextSize(label, 0, fontScale=sf, thickness=tf)[0]
#         # label fits outside box
#         outside = box[1] - h - 3 >= 0
#         p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
#         # 绘制矩形框填充
#         cv2.rectangle(img_src, p1, p2, color, -1, cv2.LINE_AA)
#         # 绘制标签
#         cv2.putText(img_src, label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
#                     0, sf, self.color["font"], thickness=2, lineType=cv2.LINE_AA)
#
#     def draw_masks(self, img_src, masks, colors, im_gpu, alpha=0.5):
#         # maks [6, 640, 480]
#         if len(masks) == 0:
#             img_src[:] = im_gpu.permute(1, 2, 0).contiguous().cpu().numpy() * 255
#         if im_gpu.device != masks.device:
#             im_gpu = im_gpu.to(masks.device)
#         colors = torch.tensor(colors, device=masks.device, dtype=torch.float32) / 255.0  # shape(n,3)
#         colors = colors[:, None, None]  # shape(n,1,1,3)
#         masks = masks.unsqueeze(3)  # shape(n,h,w,1)
#         masks_color = masks * (colors * alpha)  # shape(n,h,w,3)
#
#         inv_alpha_masks = (1 - masks * alpha).cumprod(0)  # shape(n,h,w,1)
#         mcs = masks_color.max(dim=0).values  # shape(h,w,3)
#
#         im_gpu = im_gpu.flip(dims=[0])  # flip channel
#         im_gpu = im_gpu.permute(1, 2, 0).contiguous()  # shape(h,w,3)
#         im_gpu = im_gpu * inv_alpha_masks[-1] + mcs
#         im_mask = im_gpu * 255
#         im_mask_np = im_mask.byte().cpu().numpy()
#         img_src[:] = ops.scale_image(im_mask_np, img_src.shape)
#
#     @staticmethod
#     def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), scaleup=True, stride=32):
#         # Resize and pad image while meeting stride-multiple constraints
#         shape = im.shape[:2]  # current shape [height, width]
#         if isinstance(new_shape, int):
#             new_shape = (new_shape, new_shape)
#
#         # Scale ratio (new / old)
#         r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
#         if not scaleup:  # only scale down, do not scale up (for better val mAP)
#             r = min(r, 1.0)
#
#         # Compute padding
#         ratio = r, r  # width, height ratios
#         new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
#         dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
#         # minimum rectangle
#         dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
#         dw /= 2  # divide padding into 2 sides
#         dh /= 2
#
#         if shape[::-1] != new_unpad:  # resize
#             im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
#
#         top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
#         left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
#         im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
#         return im, ratio, (dw, dh)
#
#     def precess_image(self, img_src):
#         # Padded resize
#         img = self.letterbox(img_src, self.imgsz)[0]
#         img = np.expand_dims(img, axis=0)
#         # Convert
#         img = img[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW, (n, 3, h, w)
#         img = np.ascontiguousarray(img)  # contiguous
#         img = torch.from_numpy(img)
#         img = img.to(self.device)
#         img = img.half() if self.half else img.float()  # uint8 to fp16/32
#         img = img / 255  # 0 - 255 to 0.0 - 1.0
#         return img
#
#
# if __name__ == '__main__':
#     weights = r'F:\AI_tooth\enamel_demineralization\runs\weights\yolov8n-seg.pt'
#     cuda = 'cuda:0'
#     save_path = r"F:\AI_tooth\enamel_demineralization\runs\segment"
#
#     if not os.path.exists(save_path):
#         os.mkdir(save_path)
#
#     model = YOLOV8SegmentInfer(weights, cuda, 0.25, 0.7)
#
#     img_path = r"F:\AI_tooth\data\images\1-1.jpg"
#     img_src = cv2.imread(img_path)
#     model.infer(img_src, save_path)

from ultralytics import YOLO
from matplotlib import pyplot as plt
import numpy as np
import cv2

# 加载sam模型，如果没有这个框架也会自动的为你下载
model = YOLO(r"F:\AI_tooth\enamel_demineralization\runs\weights\yolov8n-seg.pt")

# 读取一张图片，
orig_img = cv2.imread(r'F:\AI_tooth\enamel_demineralization\ultralytics-main\bus.jpg')
# 这里特别注意，因为使用yolov8训练的时候默认会把图片resize成448*640的尺寸，所以这里也得改成你训练的尺寸
orig_img = cv2.resize(orig_img, (640, 448))  # 注意OpenCV中尺寸是先宽度后高度

# 使用模型进行推理， 后面save=True的参数可以输出测试分割的图片
results = model(orig_img, save=True)

# 这里是我获取mask编码部分的。
mask = results[0].masks.data[0].cpu().numpy().astype(np.bool_)

# 应用掩码到原始图像
# 如果你想要背景是黑色的
masked_image = np.zeros_like(orig_img)
masked_image[mask] = orig_img[mask]

# 如果你想要背景透明（假设原始图像是RGB格式）
# 创建一个RGBA图像，其中背景是透明的
alpha_channel = np.ones(mask.shape, dtype=orig_img.dtype) * 255  # 创建alpha通道
masked_image_rgba = np.dstack((masked_image, alpha_channel))  # 将alpha通道添加到RGB通道
masked_image_rgba[~mask] = (0, 0, 0, 0)  # 将非掩码区域设置为透明

# 保存图片，黑色背景
cv2.imwrite('masked_image_test.jpg', masked_image)

# 如果背景是透明的
cv2.imwrite('masked_image_test.png', masked_image_rgba)

# label-studio-ml start sam --port 8003 --with sam_config=vit_h sam_checkpoint_file=sam_vit_h_4b8939.pth out_mask=True out_bbox=True device=cuda:0