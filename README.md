# 风电场布局优化工具

这个项目用于估算风电场的年发电量，并比较不同风机布局和尾流模型的结果。项目包含风机与风资源模型、场地边界和间距约束、遗传算法与粒子群优化、经济性分析以及无界面图表输出。

## 安装

建议使用 Python 3.10 或更新版本，并在虚拟环境中安装依赖：

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
```

Windows PowerShell 可以使用 `.venv\\Scripts\\Activate.ps1` 激活环境。

## 快速验证

```bash
python quick_test.py
```

快速验证会覆盖模型、约束、年发电量、优化、经济性和图表生成，并在 `test_output/` 写入临时图片。该目录不会纳入版本控制。

## 完整分析

```bash
python -m wind_farm_opt --help
python -m wind_farm_opt --n-turbines 15 --iterations 100 --population 50 --output-dir output
```

也可以先生成配置文件，再通过 `--config` 运行：

```bash
python -m wind_farm_opt --generate-config my_config.json
python -m wind_farm_opt --config my_config.json
```

所有运行结果默认写入 `output/`，可以用 `--no-plots` 跳过图表生成。命令行使用无界面绘图后端，适合容器和服务器环境。
