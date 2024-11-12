import cv2
import matplotlib.pyplot as plt
import numpy as np
import pyrealsense2 as rs
from matplotlib.patches import Circle, Rectangle
from PIL import Image

# fix the random seed for deterministic generation
np.random.seed(500)

# number of positions to generate
test_case = 15

# define target objects
# format: [type, params, x_range, y_range]
# each element in params is either a constant or a list [min, max]
#   constant: the value is fixed
#   [min, max]: the value is randomly generated between min and max
# support type:
#   circle: [r]
#   rect: [w, h, [angle_min, angle_max]]
objs = [
    ['circle', [0.13], [0.2, 0.8], [0.2, 0.8]],
    ['rect', [0.3, 0.33, [0, 180]], [0.15, 0.85], [0.15, 0.85]]
]

obj_params = []
for obj in objs:
    x_min, x_max = obj[2]
    y_min, y_max = obj[3]
    # evenly spilt the space
    grid_num_x = int(np.ceil(np.sqrt(test_case * (x_max-x_min)/(y_max-y_min))))
    grid_num_y = int(np.ceil((y_max-y_min)/(x_max-x_min) * grid_num_x))

    grid_size_x = (x_max-x_min) / grid_num_x
    grid_size_y = (y_max-y_min) / grid_num_y

    X, Y = np.meshgrid(np.linspace(x_min, x_max, grid_num_x+1)[:-1], np.linspace(y_min, y_max, grid_num_y+1)[:-1])

    grid_pos = np.stack([X.flatten(), Y.flatten()], axis=1)
    grid_num = len(grid_pos)
    print(f'Total {grid_num} grids for {obj[0]}')
    
    params = np.zeros((grid_num, 2+len(obj[1])))
    # randomly generate positions
    params[:, 0] = np.random.uniform(size=grid_num) * grid_size_x
    params[:, 1] = np.random.uniform(size=grid_num) * grid_size_y
    params[:, :2] += grid_pos
    # determine other params
    for i, p in enumerate(obj[1]):
        if isinstance(p, list):
            params[:, 2+i] = np.random.uniform(low=p[0], high=p[1], size=grid_num)
        else:
            params[:, 2+i] = p
    np.random.shuffle(params)
    obj_params.append([obj[0], params])

def distance(obj_1, obj_2):
    x1 = obj_1[1][0]
    y1 = obj_1[1][1]
    if obj_1[0] == 'circle':
        r1 = obj_1[1][2]
    elif obj_1[0] == 'rect':
        r1 = np.linalg.norm(obj_1[1][2:4]/2)

    x2 = obj_2[1][0]
    y2 = obj_2[1][1]
    if obj_2[0] == 'circle':
        r2 = obj_2[1][2]
    elif obj_2[0] == 'rect':
        r2 = np.linalg.norm(obj_2[1][2:4]/2)

    return np.linalg.norm([x2-x1, y2-y1]) - r1 - r2

def compatible(group, obj_param, i):
    for obj in group:
        d = distance(obj, (obj_param[0], obj_param[1][i]))
        if d < 0:
            return d
    return d

# generate positions
trial_cnt = 0
while trial_cnt < 1000:
    pos = []
    retry = False
    for i in range(test_case):
        # select the first object
        group = [(obj_params[0][0], obj_params[0][1][i])]
        # choose other objects from non-overlapping candidates
        for obj_param in obj_params[1:]:
            p = np.array([compatible(group, obj_param, j) for j in range(i, len(obj_param[1]))])
            p[p>=0] += 0.1
            p[p<0] = 0
            if sum(p) > 0:
                p /= sum(p)
                j = np.random.choice(range(i, len(obj_param[1])), p=p)
                temp = obj_param[1][i].copy()
                obj_param[1][i] = obj_param[1][j]
                obj_param[1][j] = temp
                group.append((obj_param[0], obj_param[1][i]))
            else:
                print('error when matching', i)
                retry = True
                break
        if retry:
            break
        pos.append(group)
    trial_cnt += 1
    if not retry:
        print('success!')
        break

# visualize the generated positions on the image
frame_rate = 30
resolution = (1280, 720)
pipeline = rs.pipeline()
config = rs.config()
# TODO: change the camera serial
serial = '038522063145'

config.enable_device(serial)
config.enable_stream(rs.stream.color, resolution[0], resolution[1], rs.format.rgb8, frame_rate)
align = rs.align(rs.stream.color)

pipeline.start(config)

img_size = 720
# define the perspective transform matrix to simulate top-down view
# TODO: change the four anchor points according to your camera position
M = cv2.getPerspectiveTransform(
    np.array([(345, 690), (953, 713), (430, 250), (850, 270)],dtype=np.float32),
    np.array([(0, img_size), (img_size, img_size), (0, 0), (img_size, 0)],dtype=np.float32),
)

def on_press(event):
    global r, idx
    if event.key == 'r':
        r = range(test_case)
        idx = -1
    elif event.key == 'n':
        idx += 1
        if idx >= test_case:
            idx = 0
        r = [idx]
    elif event.key == 'p':
        idx -= 1
        if idx < 0:
            idx = test_case-1
        r = [idx]
    elif event.key == 'q':
        pipeline.stop()
        exit()

fig = plt.figure(figsize=(12,12))
fig.canvas.mpl_connect('key_press_event', on_press)

r = range(test_case)
idx = -1

plt.ion()
while True:
    frameset = align.process(pipeline.wait_for_frames())
    img = Image.fromarray(np.asanyarray(frameset.get_color_frame().get_data()).astype(np.uint8))
    img = cv2.warpPerspective(np.array(img), M, (img_size, img_size))

    ax = plt.gca()
    ax.imshow(img)
    ax.set_title(f'{idx+1}th Case' if idx >= 0 else 'All Cases')
    for i in r:
        for obj in pos[i]:
            if obj[0] == 'circle':
                ax.add_patch(
                    Circle(
                        xy=obj[1][:2]*img_size,
                        radius=obj[1][2]*img_size,
                        fill=False,
                    )
                )
            elif obj[0] == 'rect':
                ax.add_patch(
                    Rectangle(
                        xy=(obj[1][:2] - obj[1][2:4]/2)*img_size,
                        width=obj[1][2]*img_size,
                        height=obj[1][3]*img_size,
                        angle=obj[1][4],
                        rotation_point='center',
                        fill=False,
                    )
            )
    plt.show()
    plt.pause(0.1)
    plt.clf()
