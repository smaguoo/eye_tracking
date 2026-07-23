# Online EllSeg 单目纯几何视线估计

本项目提供一条基于笔记本前置摄像头的实时单目视线估计流程。直接用瞳孔轴与屏幕平面的几何交点显示 gaze dot。

```text
camera frame
-> 单眼 ROI
-> EllSeg/DenseElNet 分割
-> pupil ellipse
-> quality filter + EMA
-> virtual pupil axis
-> axis candidate auto disambiguation
-> real pupil axis, 角膜折射修正
-> 几何模型: 眼轴射线与屏幕平面求交
-> 屏幕 gaze dot + CSV
```

## 代码入口

当前主入口是：

```text
online_ellseg.run_gaze_display
```

它只保留单目纯几何流程。`--geometry` 参数仍可写在命令里，目前是兼容参数；该入口始终使用纯几何模型。

## 外部 EllSeg 的依赖

本项目复用外部 EllSeg/DenseElNet 仓库。默认路径目前是本机开发路径：

```text
E:\Users\guocong1.wen\PycharmProjects\eye_test\ellseg_denseelnet
```

默认权重路径是：

```text
E:\Users\guocong1.wen\PycharmProjects\eye_test\ellseg_denseelnet\weights\all.git_ok
```

运行时也可以用参数覆盖：

```powershell
--ellseg-root <EllSeg 仓库路径> --weights <权重文件路径>
```


## 环境检查

使用当前 `eye_uss` Python 环境：

```powershell
& E:\Users\guocong1.wen\AppData\Local\anaconda3\envs\eye_uss\python.exe -m online_ellseg.check_env
```

依赖可参考：

```text
requirements-eye_uss.txt
```

主要依赖包括：

```text
numpy
opencv-python
torch
torchvision
scipy
scikit-image
tqdm
```

## 相机标定

几何模型依赖相机内参。运行分辨率必须和标定文件匹配。

当前常用配置是 `1280x720 @ 30fps`，对应：

```text
outputs\camera_calibration_1280x720.json
```

使用 9x6 内角点、25 mm 方格棋盘进行标定：

```powershell
cd E:\Users\guocong1.wen\PycharmProjects\eye_sys

& E:\Users\guocong1.wen\AppData\Local\anaconda3\envs\eye_uss\python.exe -m online_ellseg.calibrate_camera `
  --camera 0 `
  --backend dshow `
  --width 1280 `
  --height 720 `
  --fps 30 `
  --auto `
  --output-json outputs\camera_calibration_1280x720.json `
  --output-npz outputs\camera_calibration_1280x720.npz
```



## 坐标系

几何部分使用 OpenCV 相机坐标系：

```text
+x: 图像右侧
+y: 图像下方
+z: 从摄像头指向人脸
```

屏幕建模为一个与相机图像平面平行的平面：

```text
screen z = --screen-z-mm
```

单目模式下，`--eye-midpoint-x-mm/y-mm/z-mm` 表示当前追踪的这只眼睛的三维原点，而不是双眼中点。

例如当前左眼固定在摄像头正前方 150 mm、下方 50 mm：

```powershell
--eye-midpoint-x-mm 0 --eye-midpoint-y-mm 50 --eye-midpoint-z-mm 150
```

## 实时运行

第一次运行手动框选 ROI：

```powershell
$py="E:\Users\guocong1.wen\AppData\Local\anaconda3\envs\eye_uss\python.exe"; cd "E:\Users\guocong1.wen\PycharmProjects\eye_sys"; & $py -m online_ellseg.run_gaze_display --device cuda --camera 0 --backend dshow --width 1280 --height 720 --fps 30 --select-roi --geometry --calibration outputs\camera_calibration_1280x720.json --screen-width-mm 346 --screen-height-mm 217 --screen-center-x-mm 0 --screen-center-y-mm 110 --screen-z-mm 0 --eye-midpoint-x-mm 0 --eye-midpoint-y-mm 50 --eye-midpoint-z-mm 150 --kappa-yaw-deg 5.2 --kappa-pitch-deg 1.5 --invert-gaze-y --axis-candidate auto --axis-disambiguation-window 8 --axis-disambiguation-smoothness 1 --smooth-alpha 1.0 --windowed --csv outputs\live_gaze_geometry_mono.csv
```

程序会打印类似：

```text
selected_roi: 334,308,154,84
```

之后如果头位和摄像头画面固定，可以把 `--select-roi` 换成固定 ROI：

```powershell
$py="E:\Users\guocong1.wen\AppData\Local\anaconda3\envs\eye_uss\python.exe"; cd "E:\Users\guocong1.wen\PycharmProjects\eye_sys"; & $py -m online_ellseg.run_gaze_display --device cuda --camera 0 --backend dshow --width 1280 --height 720 --fps 30 --roi 334,308,154,84 --geometry --calibration outputs\camera_calibration_1280x720.json --screen-width-mm 346 --screen-height-mm 217 --screen-center-x-mm 0 --screen-center-y-mm 110 --screen-z-mm 0 --eye-midpoint-x-mm 0 --eye-midpoint-y-mm 50 --eye-midpoint-z-mm 150 --kappa-yaw-deg 5.2 --kappa-pitch-deg 1.5 --invert-gaze-y --axis-candidate auto --axis-disambiguation-window 8 --axis-disambiguation-smoothness 1 --smooth-alpha 1.0 --windowed --csv outputs\live_gaze_geometry_mono.csv
```

## 参数

`--calibration`

相机标定 JSON。必须与当前 `--width/--height` 匹配。

`--roi`

固定单眼区域，格式为 `x,y,w,h`，单位是相机图像像素。

`--axis-candidate auto`

非固定椭圆反投影的两个候选法线分支。目前程序会用最近若干帧自动选择更符合眼球模型和时间连续性的分支。

`--axis-disambiguation-window 8`

用于分支消歧的滑动窗口长度。每帧椭圆反投影有两个候选法线，窗口为 8 时会在最近 8 帧里选择整体最合理的候选路径。

`--kappa-yaw-deg` / `--kappa-pitch-deg`

视觉轴与瞳孔轴之间的常数角度修正。它们用于修正系统性偏移，但不能替代真实个体眼模型标定。

`--smooth-alpha`

gaze dot 的屏幕位置平滑系数。`1.0` 表示不平滑，响应最快但更抖；较小值更平滑但有延迟。

`--filter-alpha`

pupil ellipse 的 EMA 平滑系数。

## CSV 输出

默认输出：

```text
outputs\live_gaze_geometry_mono.csv
```

字段包括：

```text
timestamp
valid
raw_x, raw_y
smooth_x, smooth_y
yaw_deg, pitch_deg
axis_reason
```

`raw_x/raw_y` 是几何求交得到的原始屏幕坐标，`smooth_x/smooth_y` 是 gaze dot 平滑后的坐标。

## 当前保留的核心文件

单目纯几何主流程需要：

```text
online_ellseg\run_gaze_display.py
online_ellseg\camera.py
online_ellseg\ellseg_adapter.py
online_ellseg\pipeline.py
online_ellseg\quality.py
online_ellseg\filtering.py
online_ellseg\roi.py
online_ellseg\types.py
online_ellseg\virtual_axis.py
online_ellseg\axis_disambiguation.py
online_ellseg\real_pupil_axis.py
online_ellseg\geometry_gaze.py
online_ellseg\calibrate_camera.py
online_ellseg\check_env.py
online_ellseg\__init__.py
```


## 常见误差来源

当前模型中，主要误差通常来自：

```text
眼睛三维原点不准
屏幕-相机几何参数不准
kappa 常数修正不适合全屏
角膜折射模型使用通用参数
EllSeg pupil ellipse 检测误差
axis candidate 分支选择错误
相机标定残差或分辨率不匹配
```

如果 gaze dot 有趋势但偏移明显，优先检查：

```text
--eye-midpoint-x/y/z-mm
--screen-center-x/y-mm
--screen-width-mm / --screen-height-mm
--kappa-yaw-deg / --kappa-pitch-deg
```

```

EllSeg/DenseElNet 的获取方式、权重下载方式和 `--ellseg-root/--weights` 配置
