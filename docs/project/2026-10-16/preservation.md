# MindLM 资料保全与恢复

2026-10-16 分别保存本地与 60 的原始工作树资料，随后新增项目台账。两端共享 HEAD，但未提交资料不同，档案不互相覆盖。源码、报告、配置和小型文本资料有内容副本；模型、二进制数据和大文件只有原位清单，不能用本档案恢复全部训练资产。

## 档案位置

本机档案根目录为 `/Users/mac/research/MindLM-project-archive/20261016/20261016T024604Z`。60 的档案根目录为 `/mnt/data2/runke.zhong.srv/MindLM/archives/20261016T024604Z`。

| 来源与用途 | 已保存位置 | 内容与校验结果 |
|---|---|---|
| 本地原始工作树 | 本机 [local-worktree](../../../../MindLM-project-archive/20261016/20261016T024604Z/local-worktree/SUMMARY.json) 与 [local-worktree.tar.gz](../../../../MindLM-project-archive/20261016/20261016T024604Z/local-worktree.tar.gz) | 372 个常规文件，40,915,363 字节；包 7,459,475 字节；逐文件 SHA256 通过，Git status/index 不变 |
| 60 原始工作树 | 服务器 `ssh60:/mnt/data2/runke.zhong.srv/MindLM/archives/20261016T024604Z/server60` 与同根 `server60.tar.gz`；本机 [server60.tar.gz 副本](../../../../MindLM-project-archive/20261016/20261016T024604Z/server60.tar.gz) | 1,021 个常规文件，277,230,416 字节，另保留 30 个符号链接；覆盖全部 13 个 run；包 49,551,291 字节；源文件与复制内容 SHA256 一致，Git status/index 不变 |
| 60 档案元数据 | 本机 [server60-metadata](../../../../MindLM-project-archive/20261016/20261016T024604Z/server60-metadata/SUMMARY.json) | 原始文件清单、排除理由、Git 身份和前后状态、复制校验；完整资料内容在 tar 中 |
| 外部小型数据补充 | 本机 [external-data-supplement.tar.gz](../../../../MindLM-project-archive/20261016/20261016T024604Z/external-data-supplement.tar.gz) 与 [逐文件校验](../../../../MindLM-project-archive/20261016/20261016T024604Z/external-data-supplement/verification.json) | 单独补存 format pilot 的 `train.raw.jsonl`、960 pilot 的 `raw.jsonl` 和 `supervision_recheck.json`，共 8,848,291 字节；3 文件源端与副本哈希均一致 |
| 双源差异 | 本机 [source-comparison.json](../../../../MindLM-project-archive/20261016/20261016T024604Z/source-comparison.json) | 比较已复制的小文件及路径；含不同报告层级与检查点元信息，不是独有源码数量统计 |
| 封包传输校验 | 本机 [server60-transfer-verification.json](../../../../MindLM-project-archive/20261016/20261016T024604Z/server60-transfer-verification.json) | 整包 SHA256 一致，包内 1,056 个常规成员哈希与 30 个链接目标验证通过 |
| 本轮新增台账 | 本机 [project-register.tar.gz](../../../../MindLM-project-archive/20261016/20261016T024604Z/project-register.tar.gz)，相邻 `.sha256`；包内 `project-register/` | 本目录 4 文件与独立校验记录；不改动原始工作树快照 |

表中 Markdown 链接仅供本机访问，服务器路径以 `ssh60:` 标记。每份原始档案包含 `MANIFEST.jsonl`、`SUMMARY.json`、`SHA256SUMS`、`verification.jsonl`、Git 证据及 `snapshot/worktree/`。相邻 `.tar.gz.sha256` 文件校验压缩包，包内 `SHA256SUMS` 校验所收录的常规文件。

完整范围见本机 [MASTER.json](../../../../MindLM-project-archive/20261016/20261016T024604Z/MASTER.json)、[新增文件清单](../../../../MindLM-project-archive/20261016/20261016T024604Z/ADDED-FILES.jsonl) 和 [归档总校验](../../../../MindLM-project-archive/20261016/20261016T024604Z/ARCHIVE-SHA256SUMS)。外部小数据、验证证据与新台账是独立补充包，不改变原始双源快照。

## 保存范围与排除内容

保存范围包括模型与训练源码、数据处理器、评测与测试、配置、tokenizer/模板的小型文件、依赖说明、文档、实验命令、日志、manifest、选择记录及原始评测。未跟踪源码和负面报告也保留；Git HEAD、近期提交信息、index 明细、staged/unstaged patch 与归档前后状态独立记录。

以下内容不包含完整数据副本，具体条目与 `action` 见各源 `MANIFEST.jsonl`：

- `.pt`、`.pth`、`.safetensors`、`.bin`、`.parquet` 等模型或二进制数据，仅记录路径、大小、时间和权限；其他超过 50MiB 的文件同样只做元数据清单。模型内容未统一计算哈希，完整恢复仍依赖原位文件。
- `.git`、`.cache`、`.pytest_cache`、`__pycache__`、虚拟环境等不复制。`.git` pack 约 83.69GiB；本档案保存工作树与补丁，不是完整 Git 历史备份。
- 符号链接保留链接本身与目标字符串，不跟随。60 的 `full_data_v2` 指向 `/mnt/data2/reexen_datasets/llm_training/MindLM_full_data_v2`；外部数据有元数据清单、所选小型元数据及单独补存的两份小型 raw JSONL，大型 raw 与二进制不在档案中。`full_data_v2.migration-backup/` 保持原位，未作为可删除内容处理。
- 根权限所有的可再生缓存未变更权限。早期只读探针与封包校验的链接跟随问题记录在执行证据中；修正后原始资料复制与封包校验均通过，归档 summary 的 `errors` 和 `copy_integrity_errors` 为空。

本轮原始快照发生在新台账创建之前；新台账另行保全，避免混入历史源状态。已授权的前一轮检查点清理不在本次恢复范围内，被删除的旧模型不能从小资料档案中找回。

## 验证记录

本轮在 60 的既有训练环境执行 CPU 测试，持久日志位于本机 `verification-evidence/`：

- [完整测试日志](../../../../MindLM-project-archive/20261016/20261016T024604Z/verification-evidence/mindlm-review-full-20261016.log)：退出码 2，4.82 秒；收集 `tests/test_tool_loop.py` 时缺少 `eval.eval_sft_tool.build_summary`。
- [排除闭环文件的测试日志](../../../../MindLM-project-archive/20261016/20261016T024604Z/verification-evidence/mindlm-review-excluding-tool-loop-20261016.log)：退出码 0，9.87 秒；53 passed、27 subtests passed、16 warnings。
- [执行记录](../../../../MindLM-project-archive/20261016/20261016T024604Z/verification-evidence/execution-record.json)：归档计划、退出码、探针异常与修正范围。该目录独立于原始工作树档案。
- [测试命令](../../../../MindLM-project-archive/20261016/20261016T024604Z/verification-evidence/test-commands.json)：容器、Python 环境、完整 argv、退出码及 CPU/缓存控制参数。
- [Needle 研究依赖副本](../../../../MindLM-project-archive/20261016/20261016T024604Z/verification-evidence/needle-模型格式与训练目标分析.md) 与 [来源校验](../../../../MindLM-project-archive/20261016/20261016T024604Z/verification-evidence/research-dependency.json)：保存 10 月 3 日计划引用的本地研究资料，源文件与副本 SHA256 一致。该研究判断不等于 MindLM 实验证明。

历史闭环使用的版本与当前 HEAD 不同，历史分数不能代替当前入口可运行的证明；本轮未执行 GPU 训练、吞吐或数值复测。

## 恢复步骤

1. 从相应源端取得 tar 与同名 `.sha256`，先核对来源标签和 HEAD。两个源的档案分别解包，不合并覆盖。
2. 在新目录执行包校验后解包，例如在本机：

   ```bash
   cd /Users/mac/research/MindLM-project-archive/20261016/20261016T024604Z
   shasum -a 256 -c server60.tar.gz.sha256
   mkdir /tmp/MindLM-server60-review-20261016
   tar -xzf server60.tar.gz -C /tmp/MindLM-server60-review-20261016
   cd /tmp/MindLM-server60-review-20261016/server60
   shasum -a 256 -c SHA256SUMS
   ```

3. 从 `snapshot/worktree/` 查看所需文件，对照 manifest 的 `action` 和 Git 证据决定恢复范围。已保存的工作树文件已经包含未提交修改，不要再次盲目套用相同 patch。
4. 从独立、完整的仓库取得所记录 HEAD；本档案不提供全部 Git 对象。确认模型与外部数据原位文件仍存在，再恢复链接和运行环境。只靠元数据不能重建模型或大数据内容。
5. 在独立目录完成文件差异审查、入口检查和关联测试后，再确定是否回写开发目录。

## 遗留风险

60 的归档副本与外部数据均在 data2，同盘故障仍可能同时影响它们；服务器小资料包另有本机副本，模型和大数据仍未形成独立备份。repair v2 生成器、旧临时评测权重及部分被引用文档缺失，归档不能补齐原本不存在的内容。符号链接可能指向服务器外部路径，离线解包后不可直接运行。
