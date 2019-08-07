import cv2
import numpy as np
import PIL.Image as Image
from matplotlib import gridspec
from matplotlib import pyplot as plt
import torch
from torch.nn import functional as F


def vis_segmentation(image, map_list):
  """Visualizes input image, segmentation map and overlay view."""
  plt.figure(figsize=(15, 5))
  grid_spec = gridspec.GridSpec(2, 5, width_ratios=[6, 6, 6, 6, 6])

  plt.subplot(grid_spec[0,0])
  plt.imshow(image)
  plt.axis('off')
  plt.title('input image')

  plt.subplot(grid_spec[0,1])
  plt.imshow((map_list[0]*255).astype(np.uint8))
  plt.axis('off')
  plt.title('upper body decomp att map')

  plt.subplot(grid_spec[0,2])
  plt.imshow((map_list[1]*255).astype(np.uint8))
  plt.axis('off')
  plt.title('lower body decomp att map')

  plt.subplot(grid_spec[0,3])
  plt.imshow((map_list[2]*255).astype(np.uint8))
  plt.axis('off')
  plt.title('att_aprt1 map')

  plt.subplot(grid_spec[0,4])
  plt.imshow((map_list[3]*255).astype(np.uint8))
  plt.axis('off')
  plt.title('att_aprt2 map')

  plt.subplot(grid_spec[1, 1])
  plt.imshow((map_list[4] * 255).astype(np.uint8))
  plt.axis('off')
  plt.title('att_aprt3 map')

  plt.subplot(grid_spec[1, 2])
  plt.imshow((map_list[5] * 255).astype(np.uint8))
  plt.axis('off')
  plt.title('att_aprt4 att map')

  plt.subplot(grid_spec[1, 3])
  plt.imshow((map_list[6] * 255).astype(np.uint8))
  plt.axis('off')
  plt.title('att_aprt5 map')

  plt.subplot(grid_spec[1, 4])
  plt.imshow((map_list[7] * 255).astype(np.uint8))
  plt.axis('off')
  plt.title('att_aprt6 map')

  # unique_labels = np.unique(seg_map)
  # ax = plt.subplot(grid_spec[3])
  # plt.imshow(
  #     FULL_COLOR_MAP[unique_labels].astype(np.uint8), interpolation='nearest')
  # ax.yaxis.tick_right()
  # plt.yticks(range(len(unique_labels)), LABEL_NAMES[unique_labels])
  # plt.xticks([], [])
  # ax.tick_params(width=0.0)
  plt.grid('off')
  plt.show()

img = Image.open("/home/hlzhu/hlzhu/Iter_ParseNet_final/data/Person/JPEGImages/2008_002829.jpg")
# image = torch.load("/home/hlzhu/hlzhu/Iter_ParseNet_final/vis/image.pth")
# img = torch.tensor(image[0]).permute(1,2,0).cpu().numpy()
# xfh_att_list = torch.load("/home/hlzhu/hlzhu/Iter_ParseNet_final/vis/2008_002829/xfh_att_list.pth")
att_upper = torch.load("/home/hlzhu/hlzhu/Iter_ParseNet_final/vis/2008_002829/att_upper.pth")
att_lower = torch.load("/home/hlzhu/hlzhu/Iter_ParseNet_final/vis/2008_002829/att_lower.pth")
att_part1 = torch.load("/home/hlzhu/hlzhu/Iter_ParseNet_final/vis/2008_002829/att_part_1.pth")
att_part2 = torch.load("/home/hlzhu/hlzhu/Iter_ParseNet_final/vis/2008_002829/att_part_2.pth")
att_part3 = torch.load("/home/hlzhu/hlzhu/Iter_ParseNet_final/vis/2008_002829/att_part_3.pth")
att_part4 = torch.load("/home/hlzhu/hlzhu/Iter_ParseNet_final/vis/2008_002829/att_part_4.pth")
att_part5 = torch.load("/home/hlzhu/hlzhu/Iter_ParseNet_final/vis/2008_002829/att_part_5.pth")
att_part6 = torch.load("/home/hlzhu/hlzhu/Iter_ParseNet_final/vis/2008_002829/att_part_6.pth")

# xfh_att_map = [torch.tensor(xfh_map[0][0]).cpu().numpy() for xfh_map in xfh_att_list]
h = img.height
w = img.width

att_upper = F.interpolate(att_upper, size=(h, w), mode='bilinear', align_corners=True)
att_upper = torch.tensor(att_upper[0][0]).cpu().numpy()

att_lower = F.interpolate(att_lower, size=(h, w), mode='bilinear', align_corners=True)
att_lower = torch.tensor(att_lower[0][0]).cpu().numpy()

att_part1 = F.interpolate(att_part1, size=(h, w), mode='bilinear', align_corners=True)
att_part1 = torch.tensor(att_part1[0][0]).cpu().numpy()

att_part2 = F.interpolate(att_part2, size=(h, w), mode='bilinear', align_corners=True)
att_part2 = torch.tensor(att_part2[0][0]).cpu().numpy()

att_part3 = F.interpolate(att_part3, size=(h, w), mode='bilinear', align_corners=True)
att_part3 = torch.tensor(att_part3[0][0]).cpu().numpy()

att_part4 = F.interpolate(att_part4, size=(h, w), mode='bilinear', align_corners=True)
att_part4 = torch.tensor(att_part4[0][0]).cpu().numpy()

att_part5 = F.interpolate(att_part5, size=(h, w), mode='bilinear', align_corners=True)
att_part5 = torch.tensor(att_part5[0][0]).cpu().numpy()

att_part6 = F.interpolate(att_part6, size=(h, w), mode='bilinear', align_corners=True)
att_part6 = torch.tensor(att_part6[0][0]).cpu().numpy()

vis_segmentation(img, [att_upper,att_lower, att_part1, att_part2, att_part3, att_part4, att_part5, att_part6])




