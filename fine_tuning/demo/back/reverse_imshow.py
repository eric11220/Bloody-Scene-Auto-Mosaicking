import sys
import cv2
import math
import matplotlib.pyplot as plt
color_num = int(sys.argv[1])

step = float(1)/color_num
sqrt = int(math.sqrt(float(color_num)))

img = [[0 for x in range(0,sqrt)] for x in range(0,sqrt)]
for i in range(0,sqrt):
    for j in range(0,sqrt):
        img[i][j] = step*(i*sqrt+j)

print sqrt
fig = plt.figure(figsize=(sqrt, sqrt), frameon=False, dpi=80)
ax = plt.Axes(fig, [0, 0, 1, 1])
ax.set_axis_off()
fig.add_axes(ax)
plt.imshow(img, aspect='normal', interpolation='nearest')
plt.savefig('reverse.png', dpi=80)

mask = cv2.imread('reverse.png')
height, width, depth = mask.shape
print str(height) + ' ' + str(width) + ' ' + str(depth)

for i in range(0,sqrt):
    for j in range(0, sqrt):
        print mask[i*80][j*80]


