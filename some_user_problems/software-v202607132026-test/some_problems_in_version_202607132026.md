# some_problems_in_version_202607132026

## 测试 202607132026

- 软件版本：v202607132026
- nose 中心线夹角：`seita3 = 80°`
- 线圈匝数：`N = 8`
- 三维模型：逐匝精细模型
- 鼻端内弯半径：`RD = 15 mm`
- 鼻端中心线圆弧半径：`Rc = RD + WA/2 = 19.25 mm`
- 槽内匝间节距：`HAD = 3.65 mm`
- 鼻端匝间节距：`HBD = 3.65 mm`
- 不含四段直臂的固定中心线长度：`L0 = 3448.254224555620 mm`
- 目标中心线长度：`LLM = 3505.783309849714 mm`
- 四段等长鼻端直臂：`Larm = (LLM - L0)/4 = 14.382271323523 mm`
- 配置文件：`test 202607132026\1\config.txt`
- STEP 文件：`test 202607132026\1\output\coil_3d.step`
- SolidWorks 装配：`test 202607132026\1\output\coil_3d.SLDASM`

## 鼻端形态总结

- 单个 nose 的主体位于用户确认的 B 立面内；鼻端多匝沿电机轴向按 `HBD` 嵌套排列，形成有厚度的立面墙。
- 两侧肩部圆弧 `rd2` 的终点 `Q` 固定，鼻冠连接点 `P` 随目标总长移动；`Q → P` 是有限长度的直鼻臂，不再把所需余长挤成中心尖峰。
- 鼻冠是中心线半径为 `Rc` 的连续 180° 圆冠；中心线连接顺序为 `rd2 → Q → 直鼻臂 → P → Rc 鼻冠`，各段保持 G1 相切。
- 非出线侧连接同序轴向位置；出线侧在整个 U 形鼻端范围内用 smoothstep 光顺接续相邻位置，不再出现独立台阶或突兀换匝坡道。
- 参数 `F` 使整个鼻端从径向朝槽底方向抬高，不改变上述鼻端立面逻辑。
- 结果中没有中心尖峰、M 形折返、额外接头或鼻冠台阶。

![](C:\Users\19144\Desktop\CoilDrawing\some_user_problems\software-v202607132026-test\pics\鼻端角度80度的线圈三维模型.png)

![](C:\Users\19144\Desktop\CoilDrawing\some_user_problems\software-v202607132026-test\pics\鼻端角度80度的线圈鼻端立面特写.png)

## 验证结果

- 完整自动化测试：`76 passed`
- 打包后 EXE 冒烟测试：UI、参数计算、二维图和详细三维 STEP 全部通过；详细三维共 39 个零件。
- SolidWorks 2023 SP5.0 审计：41 个树节点、39 个叶节点、39 个实体体、0 个曲面体、0 个线体、0 个缺失体。
- SolidWorks 装配中的 40 个组件均已转为虚拟组件并内置到 SLDASM；只复制单个 SLDASM 到隔离目录后仍可静默重开，复核结果仍为 39 个实体体、0 个曲面体、0 个线体、0 个缺失体。
- 三维安全限制：当前精细 3D 要求 `T1=T3` 且 `T2=T4`；不相等时参数计算和二维图仍可使用，但软件会阻止三维导出，避免绝缘层干涉或厚度含义错误。

## 发布包

- 文件：`dist\CoilDrawing-v202607132026.zip`
- 大小：`186271396 bytes`
- ZIP SHA-256：`C624D3733CD82BE9BFC559B27856C9492BDA732AAEE3ABFF1F1DD498078C994B`
- EXE SHA-256：`85CD247E5A0D142EFF88A11A59E1890243704BCEA80D2E9B2C7B75F53634703C`
- STEP SHA-256：`F5D16726352ABAA39A1D3B6534AC5B30B7F14E3FC1ACD8888232F16B41FD7182`
- SLDASM SHA-256：`807E88C68887299852203E0C2D53FA6349A25A4B274C75310912A5F0B9323F1D`

## 待记录问题

由用户查看和测试后填写。

## 用户看到的问题

真实线圈的形态：
下层线圈的靠近槽口的半匝 和 上层线圈靠近槽底的半匝，共同构成原始梭形线圈的最内侧匝。
下层线圈的靠近槽底的半匝 和 上层线圈靠近槽口的半匝，共同构成原始梭形线圈的最外侧匝。
从电机轴向看，会看到“人”字形；从电机切向看，会看到两个nose，每个nose都有一个“圆环”，关于这一点，可以参考![](C:\Users\19144\Desktop\CoilDrawing\some_user_problems\Reference-Pictures\完成涨形的筒形线圈.png)。

而 GPT 写的代码，做出来了这样一个东西![](C:\Users\19144\Desktop\CoilDrawing\some_user_problems\software-v202607132026-test\pics\鼻端角度80度的线圈鼻端立面特写.png)

从轴向看，看到一个“凸”形，如图![](C:\Users\19144\Desktop\CoilDrawing\some_user_problems\software-v202607132026-test\pics\轴向凸形.png)

从切向看，看到“小船”的形状，如下图所示![](C:\Users\19144\Desktop\CoilDrawing\some_user_problems\software-v202607132026-test\pics\切向船形.png)

GPT把“下层线圈的靠近槽口的半匝 和 上层线圈靠近槽底的半匝，共同构成原始梭形线圈的最内侧匝；下层线圈的靠近槽底的半匝 和 上层线圈靠近槽口的半匝，共同构成原始梭形线圈的最外侧匝” 这两句话的拓扑关系搞对了，

却没有做对“从电机轴向看，会看到‘人’字形；从电机切向看，会看到两个nose，每个nose都有一个‘圆环’”这两点。
