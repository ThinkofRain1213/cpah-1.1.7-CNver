# CPAH：赛博朋克 2077 自动黑客工具
已更新以兼容《赛博朋克 2077》v2.3 版本 /《幻影自由》DLC

原项目/作者致谢：https://gitlab.com/jkchen2/cpah
更新项目：https://gitlab.com/Dreded/cpah

### 我几乎不会写代码和使用github，我想要使用这个程序，但是它有问题。在AI的帮助下，我花了两天的时间解决了问题，顺便做了中文本地化翻译，并重绘了应用图标（原来的真的太丑了）。希望能帮助到大家。我什么也不会，如果你们懂得更多，请指导我，谢谢。

### [点击此处查看完整文档](https://dreded.gitlab.io/cpah/)

这是一款帮助简化《赛博朋克 2077》中“破解协议”黑客小游戏的工具。它会截取游戏画面，分析并决定破解所有已选守护程序的最佳顺序，甚至可以自动输入解决方案。

## 默认快捷键：
* CTRL + SHIFT + H = 分析破解协议矩阵（如果在选项中启用了自动破解功能，且所有已选守护程序均可破解，则会自动输入解决方案）
* CTRL + SHIFT + K = 自动破解（自动移动鼠标并输入解决方案）
* CTRL + SHIFT + [1-9] = 启用/禁用对应的守护程序（1-9号）
## 视频演示：
![demo](docs/media/demo.mp4)

### “分析”之后的效果
![screenshot](docs/media/screenshot_solved.png)

### 禁用守护程序 1 和 2
![screenshot](docs/media/screenshot_daemons_disabled.png)

### 解决方案长度超出缓冲区限制（缓冲区大小 = 3）
![screenshot](docs/media/screenshot_too_long.png)

### “分析”之前的状态（刚打开界面）
![screenshot](docs/media/screenshot.png)

### 游戏窗口与 CPAH 界面叠加显示
![Game Window](docs/media/cpah_game.png)
