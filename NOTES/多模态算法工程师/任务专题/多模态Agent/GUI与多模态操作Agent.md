---
learned: false
---

# GUI与多模态操作Agent：让模型亲手操作屏幕

> 本模块索引见 [多模态Agent详解](多模态Agent详解.md)

## 一、GUI Agent 任务

GUI Agent（图形界面智能体）的目标：**给定自然语言指令，模型像人一样在电脑/手机屏幕上完成操作**。

| 操作类型 | 示例 | 底层动作 |
| --- | --- | --- |
| 点击 | "点右上角的登录按钮" | `click(x, y)` |
| 输入 | "在搜索框输入'瑜伽垫'" | `type(text)` |
| 拖拽 | "把文件拖到回收站" | `drag(from_x, from_y, to_x, to_y)` |
| 滚动 | "往下翻看评论区" | `scroll(dx, dy)` |
| 打开/关闭 | "打开设置"、"关掉弹窗" | `open(app)` / `keyboard` |
| 多步任务 | "帮我查今天到北京的机票" | 上述动作的组合 |

任务本质：**把"目标状态"翻译成"一连串屏幕动作"**。难点在于：

1. 屏幕状态是**非结构化图像**——没有 DOM、没有控件句柄，模型只能"看"；
2. 动作是**有副作用且不可逆**的——点错一步可能下单、删文件；
3. 环境是**动态**的——弹窗、加载、网络延迟让每一步之后的世界都在变。

## 二、视觉定位与操作闭环

### 2.1 闭环：截图 → 定位 → 动作 → 观察新截图

```
            ┌──────────────────────────────────────────────┐
            │                操作闭环（循环）                   │
            │                                                │
            ▼                                                │
   ┌────────────────┐    ┌────────────────────┐    ┌────────┐
   │ 1. 截图         │    │ 2. 视觉定位           │    │ 3. 动作 │
   │ 获取当前屏幕像素   │───▶│ OCR/UI检测/点预测     │───▶│ click  │
   │ (截图 + 无障碍树) │    │ 找到目标元素坐标       │    │ type   │
   └────────────────┘    └────────────────────┘    │ scroll │
            ▲                                       └────────┘
            │                                            │
            └────────────── 4. 观察新截图（回到 1）─────────┘
```

每一轮循环的输入是"最新截图 + 指令 + 历史"，输出是一个动作；执行动作后拿到**新截图**，判断任务是否完成（或是否需要修正上一步）。

### 2.2 坐标系统

屏幕坐标通常以**左上角为原点**，像素为单位；模型输出坐标常做归一化（与分辨率解耦）：

$$x_{norm} = \frac{x}{W}, \quad y_{norm} = \frac{y}{H}$$

点预测的常见建模：模型输出归一化坐标，或输出一张"点击热力图"，取 argmax：

$$(\hat{x}, \hat{y}) = \arg\max_{(x,y)} f_{\theta}(x, y)$$

**为什么用归一化坐标**：同一页面在不同分辨率设备上像素坐标完全不同，归一化后模型学到的"位置语义"（右上角=0.9, 0.1 附近）跨设备可迁移。

## 三、关键方法

### 3.1 屏幕解析：OCR + UI 元素检测

模型要先"看懂"屏幕。两条路线：

| 路线 | 做法 | 优点 | 缺点 |
| --- | --- | --- | --- |
| OCR + 元素检测 | 独立小模型提取文字框、控件框，喂给 LLM/VLM | 结构清晰、可解释、便宜 | 额外延迟，检测漏检时模型"瞎" |
| VLM 直接读图 | 截图直接进 VLM，端到端理解 | 无中间损失、原生多模态 | 贵、分辨率/token 成本高 |

主流 GUI Agent 用混合路线：**截图整体送 VLM + 用 OCR/检测补充文字与控件框**（类似 [工具调用与ReAct.md](工具调用与ReAct.md) 的工具链思想）。

### 3.2 坐标预测：绝对坐标 vs 相对坐标

| 方式 | 含义 | 例子 | 优缺点 |
| --- | --- | --- | --- |
| 绝对坐标 | 直接预测目标点在屏幕的位置 | `click(0.62, 0.35)` | 简单；模型需学习"哪里是什么" |
| 相对坐标 | 预测"相对于某个锚点元素"的位置 | `click(锚点=搜索框, 偏移=(0, +8))` | 更稳（锚点由检测保证），需维护元素树 |
| 元素引用 | 直接指认已解析元素 ID | `click(uid=37)` | 最稳，但依赖解析器不漏检 |

实际系统中常组合：**点预测给初值，OCR/检测结果做锚定修正**（点击坐标取"目标元素框中心"，而不是模型裸预测的点）。

### 3.3 动作空间定义（结构化 JSON）

动作空间 = 模型允许输出的全部动作类型的集合，必须**封闭、可执行、可校验**：

```json
{
  "action": "click",
  "x": 0.62,
  "y": 0.35
}
```

```json
{"action": "type", "text": "瑜伽垫"}
```

```json
{"action": "scroll", "dx": 0, "dy": -300}
```

| 设计原则 | 说明 |
| --- | --- |
| 动作集合要小 | 6~10 个原语足够（click/type/scroll/drag/open/finish），动作越多越难学 |
| 必须含 `finish` | 模型判断任务完成时的终止动作，是循环的出口 |
| 坐标用归一化 | 与分辨率解耦（见 2.2） |
| 可校验 | 非法动作（点出屏幕、type 超长）应在执行层拦截并返回错误 Observation |

## 四、代表工作

### 4.1 Anthropic Computer Use（Claude 3.5/3.7）

- 2024 年 10 月发布的**操控真实电脑**的 API 能力；
- 原理：模型接收**截图**，输出**鼠标/键盘动作**（坐标 + 点击/输入），通过虚拟机安全执行；
- 关键设计：截图分辨率有上限、动作间自动"暂停思考"，用 system prompt 约束（"不能点击任何已提示之外的按钮"）；
- 意义：第一个把"视觉 Agent 操作真实桌面"产品化的工作，安全问题（误点击、钓鱼）引发广泛讨论。

### 4.2 Qwen2.5-VL 的 GUI 能力

- Qwen2.5-VL 系列针对 GUI 专门做了坐标能力增强（72B 以上版本支持）；
- 输出形式：**UI 定位框/点 + 绝对坐标 JSON**，例如识别"登录按钮在 (0.31, 0.42)"；
- 配套 Agent 生态：Qwen-Agent（多工具编排）、UI-TARS（端到端 GUI Agent 模型），开源生态完整；
- 与 Computer Use 的对比：Qwen 系开箱即用、可本地部署；Computer Use 是托管 API。

### 4.3 OSWorld benchmark

- 首个可复现的**通用计算机操作基准**：真实 Ubuntu 系统上 369 个任务（办公、网页、文件管理等）；
- 评测方式：完成任务后的**屏幕截图对比**（成功率），不是简单的步骤匹配；
- 数据结论：2024 年底 SOTA（GPT-4o + 相关框架）成功率约 20% 级别，说明 GUI Agent 仍远未成熟——面试可以引用这个数字说明"难点在评测和鲁棒性"。

| 工作 | 类型 | 输出 | 定位 |
| --- | --- | --- | --- |
| Anthropic Computer Use | 产品/API | 鼠标键盘动作 | 商业化桌面操作 |
| Qwen2.5-VL GUI | 模型能力 | UI 框/点 + 绝对坐标 JSON | 开源、可部署 |
| UI-TARS | 端到端模型 | 动作序列 | 开源 GUI Agent 模型 |
| OSWorld | 评测基准 | 成功率指标 | 公平对比的标尺 |

## 五、源码实现：最小 GUI 动作解析器 + 截图→OCR→点击模拟循环

```python
import json

VALID_ACTIONS = {"click", "type", "scroll", "wait", "finish"}


def parse_action(raw: str) -> dict:
    """最小 GUI 动作解析器：模型输出 JSON → 校验 → 动作指令"""
    try:
        act = json.loads(raw)
    except json.JSONDecodeError:
        raise ValueError(f"非法 JSON: {raw}")
    assert act["action"] in VALID_ACTIONS, f"非法动作类型: {act['action']}"
    if act["action"] == "click":
        assert isinstance(act.get("x"), (int, float)) and isinstance(act.get("y"), (int, float)), \
            "click 必须带数值坐标"
    return act


def to_pixel(x_norm: float, y_norm: float, W: int, H: int) -> tuple:
    """归一化坐标 → 屏幕像素坐标：x = x_norm * W"""
    return round(x_norm * W), round(y_norm * H)


def hit(button_center: tuple, x: int, y: int, radius: int = 40) -> bool:
    """点击是否命中按钮：欧氏距离小于半径"""
    bx, by = button_center
    return ((bx - x) ** 2 + (by - y) ** 2) ** 0.5 < radius


PAGES = {
    "登录页": {"登录": (120, 200), "注册": (360, 200)},
    "首页": {"提交订单": (240, 420)},
}


def screenshot(page: str) -> str:
    """模拟系统截图：返回当前页面名"""
    return page


def ocr_button_centers(page: str) -> dict:
    """模拟 OCR + UI 元素检测：返回 {文字: 中心像素坐标}"""
    return dict(PAGES[page])


def gui_model(page: str) -> str:
    """模拟 GUI 模型：截图 + 指令 → 输出动作 JSON 字符串"""
    target = "登录" if page == "登录页" else "提交订单"
    x, y = ocr_button_centers(page)[target]
    return json.dumps({"action": "click", "x": x, "y": y}, ensure_ascii=False)


def execute(page: str, act: dict) -> str:
    """模拟系统执行动作，返回执行后的新截图"""
    if act["action"] != "click":
        return page
    for name, center in PAGES[page].items():
        if hit(center, act["x"], act["y"]):
            print(f"  命中按钮『{name}』")
            return {"登录": "首页", "提交订单": "完成页", "注册": "注册页"}[name]
    print("  未命中任何按钮")
    return page


def gui_loop(max_steps: int = 3):
    """主循环：截图 → OCR → 模型输出 → 解析 → 执行 → 观察新截图"""
    page = "登录页"
    for step in range(1, max_steps + 1):
        print(f"--- Step {step} ---")
        shot = screenshot(page)
        print(f"截图：{shot}")
        print(f"OCR 结果：{ocr_button_centers(shot)}")
        raw = gui_model(shot)
        print(f"模型输出：{raw}")
        act = parse_action(raw)
        print(f"解析动作：{act}")
        new_page = execute(page, act)
        print(f"观察新截图：{new_page}")
        page = new_page
        if page == "完成页":
            print("目标达成，任务完成")
            return
    print("未达成目标")


if __name__ == "__main__":
    gui_loop()
    print("\n--- 坐标换算小实验 ---")
    print(to_pixel(0.5, 0.25, 1280, 720))   # (640, 180)
    print(hit((640, 180), 630, 185))        # True
    print(hit((640, 180), 100, 100))        # False
```

运行输出（Python 3.10+）：

```
--- Step 1 ---
截图：登录页
OCR 结果：{'登录': (120, 200), '注册': (360, 200)}
模型输出：{"action": "click", "x": 120, "y": 200}
解析动作：{'action': 'click', 'x': 120, 'y': 200}
  命中按钮『登录』
观察新截图：首页
--- Step 2 ---
截图：首页
OCR 结果：{'提交订单': (240, 420)}
模型输出：{"action": "click", "x": 240, "y": 420}
解析动作：{'action': 'click', 'x': 240, 'y': 420}
  命中按钮『提交订单』
观察新截图：完成页
目标达成，任务完成

--- 坐标换算小实验 ---
(640, 180)
True
False
```

代码与真实系统的对应关系：

| 代码位置 | 对应真实系统 |
| --- | --- |
| `screenshot` | 系统截屏 API（macOS `screencapture` / Android `screencap`） |
| `ocr_button_centers` | OCR（PaddleOCR/Tesseract）+ UI 检测模型 |
| `gui_model` | 支持 GUI 的 VLM（Qwen2.5-VL / Claude Computer Use） |
| `parse_action` | 动作解析与校验层（白名单 + 参数校验） |
| `execute` / `hit` | 系统自动化执行层（pyautogui / ADB / Playwright） |

## 六、风险与安全

| 风险 | 示例 | 缓解措施 |
| --- | --- | --- |
| 误操作（不可逆副作用） | 误点"确认付款"、"删除文件" | 高风险动作二次确认；沙箱/虚拟机执行；操作审计日志 |
| 权限控制 | 模型越权操作系统设置、访问敏感数据 | 最小权限运行（专用账户/容器）；动作白名单 |
| 屏幕信息泄露 | 截图包含密码、隐私页面 | 截图脱敏（遮罩敏感区域）；禁止截取密码框 |
| 恶意指令注入 | 网页弹窗引导模型执行恶意动作（prompt injection） | 区分"用户指令"与"屏幕内容"两种输入来源，屏幕内容不视为指令 |
| 循环失控 | 死循环反复点击 | `max_steps` 上限 + 每步成本配额 + 异常动作熔断 |

**行业做法**：Anthropic Computer Use 强制在受控虚拟机中运行；国内 Agent 产品（钉钉/飞书/腾讯 ima 等）普遍要求关键动作二次确认。面试被问"你的 Agent 敢不敢让它自己下单"——标准答案：**沙箱 + 权限最小化 + 关键动作人工确认**。

## 七、面试问答

**Q1：GUI Agent 怎么定位元素？**
主流是"截图 → OCR 提文字 + UI 检测提控件框 → 取目标元素框中心坐标"；能力强的 VLM 可直接端到端预测点/框（Qwen2.5-VL 输出 UI 定位框 + 绝对坐标 JSON）。坐标常归一化以跨分辨率迁移。

**Q2：动作空间怎么设计？**
小而封闭：click/type/scroll/drag/open/finish 等 6~10 个原语；必须带 `finish` 作为终止动作；坐标归一化；执行层做合法性校验，非法动作返回错误 Observation 而不是静默执行。

**Q3：OSWorld 是什么？**
可复现的通用计算机操作基准：真实 Ubuntu 上 369 个任务，用任务完成后截图对比算成功率。2024 年底 GPT-4o 系 SOTA 约 20% 左右，说明 GUI Agent 离成熟还很远。

**Q4：绝对坐标和相对坐标选哪个？**
绝对坐标简单但依赖模型"记住布局"；相对坐标以检测到的锚点元素为参照更稳但需元素树。生产常用组合：预测点 + 最近元素框中心做锚定修正。

**Q5：怎么防止 GUI Agent 误操作？**
沙箱/虚拟机、最小权限账户、动作白名单与二次确认、审计日志、max_steps 熔断；高风险动作（付款、删除）强制人工确认。

**Q6：GUI Agent 和 RPA 的区别？**
RPA 是"录制/编写好的固定流程"，按脚本走，不感知页面变化；GUI Agent 靠视觉实时理解屏幕、动态决策，能处理没见过的页面，但更慢、更不可控。两者常结合：稳定环节用 RPA，变化环节用 Agent。

**Q7：屏幕上的 prompt injection 怎么防？**
把输入来源分级：用户指令是"可信指令"，屏幕内容（网页文字、弹窗）是"不可信数据"，模型中明确区分两类输入；屏幕内容只能作为观察材料，不能改变任务目标。

**Q8：GUI Agent 的评测难点？**
环境状态难复现（动态页面、网络延迟）→ 用固定虚拟机镜像；结果判定难（"任务完成"要对比截图语义）→ OSWorld 用截图对比；过程质量（走弯路/误点）难自动标注 → 需要人工逐步标注。

## 八、自我检验

- [ ] 能说出 GUI Agent 的完整闭环：截图 → 定位 → 动作 → 观察新截图
- [ ] 能写出归一化坐标公式并能口算 0.5,0.25 → (640,180) @1280x720
- [ ] 能说出绝对坐标 / 相对坐标 / 元素引用的取舍
- [ ] 能设计一个含 click/type/scroll/finish 的动作空间 JSON
- [ ] 能说出 Computer Use、Qwen2.5-VL GUI、OSWorld 各自定位
- [ ] 能独立运行第五节代码并复述输出（登录 → 提交订单 → 完成）
- [ ] 能说出至少 3 条 GUI Agent 安全措施
- [ ] 能回答 8 个面试追问
