# 法律文书纠错系统


ai生成的初步项目结构

├─ 数据预处理模块（清洗、标注、格式转换）

├─ 基础模型加载模块（LegalBERT/LaWGPT等）

├─ 子任务模块

│  ├─ 语法错误检测与修正

│  ├─ 法条引用错误检测

│  └─ 逻辑矛盾检测

└─ 结果融合与可视化模块

1、数据集
关于数据集，ai总是推荐CAIL的，但是我看了一下，从数据集简介看和ai说的内容货不对板，而且要下载还要先注册，我两个手机号都尝试了，注册不给发验证码，白天可以再试试
数据集大概率还是要从网上零零散散的找一找。

https://www.modelscope.cn/datasets/Weaxrcks/csc/files
偶然发现上面那个网站，里面挑挑拣拣能找到一点有用的数据集，但是还要找更多的数据集...，我就先找了一点法律拼写纠错的数据集先做着。

2、法条拼写错误修正实现

项目

├── data_loader.py

├── model.py

├── trainer.py

├── predictor.py

├── main.py

├── law.train    # 你的训练集

└── law.test     # 你的测试集

需要的包
torch pandas numpy scikit-learn transformers tqdm

代码还没debug完，白天看有没有时间再debug
