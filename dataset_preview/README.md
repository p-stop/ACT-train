# LeRobot 数据集预览（上传 Hugging Face 前检查）

在 bimanual 上启动（只读，不改数据集）：

    cd <ROBOT_PROJECT_DIR>
    bash tasks/dataset_preview.sh                    # 列出 dataset/lerobot 和 ~/.cache/huggingface/lerobot 下的全部数据集
    bash tasks/dataset_preview.sh /某个/数据集目录     # 只看指定的数据集（或它的上级目录）
    bash tasks/dataset_preview.sh --port 8900        # 换端口

然后在同一局域网的电脑上用浏览器打开 http://<ROBOT_IP>:8899/ （启动时会打印地址），Ctrl+C 停止。

- 「概览与检查」：格式版本、回合数、时长、任务描述、特征表；逐帧检查（回合编号、帧数、index、时间戳、NaN、
  文件缺失或多余、视频时长、stats.json 是否过期、录制残留的 images/ 目录、特别短或特别长的回合、关节读数不变、动作跳变、重复回合）。
- 「回合浏览」：三路视频同步播放，14 个关节的 state/action 曲线，点曲线跳帧；
  快捷键：空格 播放，←/→ 逐帧（Shift 跳 1 秒），↑/↓ 换回合，X 标记不要。
- 标记存在 ~/.cache/lerobot_preview/marks/，不写进数据集；概览页会给出用 lerobot-edit-dataset 删掉这些回合、另存为新数据集的命令（只给命令，不执行）。

文件：tools/dataset_preview/server.py（Python 标准库 + numpy + pyarrow，用 PICO-ACT/.venv）、index.html（没有外部依赖）、tasks/dataset_preview.sh（启动脚本）。

## 标分界线（双臂合成数据 01_ACT_fused 用）

打开 `01_put_blocks_into_plate`，在「回合浏览」里场景画面会放大占满一行，下面多一栏「分界线」：

- 在场景画面上按住鼠标拖动，画一条线（自动延长到画面上下边）；拖两端的圆点微调，拖线身平移。松开鼠标自动保存。
- 线左边归左臂、右边归右臂。合成时左边的像素取自这一条的左臂阶段，右边取自右臂阶段。
- 「手臂活动区域」（快捷键 L）：蓝色是这一条里左臂到过的地方和木块杯垫、绿盘，橙色是右臂到过的地方、停靠位置和海绵杯垫、蓝盘。线要从两种颜色中间穿过；下面会显示两边离线还有多少像素，负数表示越线，线会变红。
- 两只手的活动区域在画面里重叠、怎么画都分不开时，勾「这条不参与合成」，左侧列表里显示 ⊘。
- 没标的回合显示白色虚线（自动线），可以点「用自动线」直接采用。左侧列表的「未标线」只列还没标的回合，已标的回合后面有 ⟋。
- 标注存在 `~/.cache/lerobot_preview/seams/01_put_blocks_into_plate_<哈希>.json`，不写进数据集。生成数据时用 `--seams` 指向这个文件。

手臂活动区域来自 `dataset/aug_work_01/analysis`（`tools/aug01/analyze.py` 生成），启动脚本发现这个目录就会自动带上。
