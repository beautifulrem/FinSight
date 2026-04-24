# Manual Test

人工测试入口。

运行：

```bash
python manual_test/run_manual_query.py
```

或者直接带问题：

```bash
python manual_test/run_manual_query.py --query "中国平安最近为什么跌？"
```

输出目录：

- `manual_test/output/<timestamp>-<slug>/query.txt`
- `manual_test/output/<timestamp>-<slug>/nlu_result.json`
- `manual_test/output/<timestamp>-<slug>/retrieval_result.json`

说明：

- `nlu_result.json` 对应第一个模块
- `retrieval_result.json` 对应第二个模块
- 每次运行都会新建一个输出目录，不会覆盖上一次结果
