Reading First:
1. 这个子目录的目的是进行Anomaly 这一个子类的专项测试，把各种模型在ReAct架构下的表现测试出来。
实验主要的目的是验证 给予了Oracle Table Visual Image（人工筛选出来的表格派生的可视化图像）之后，原有模型能否有显著提升？以此来验证 Agentic workflow adding Multi-modal Tools 的动机Motivation

2. 经过人工的生成，我会将 images/ 上传到该目录下，里面是*.png 图片文件，命名的格式为 {id}.png，这里的id与题目的id，甚至与csv的id一致，可以与题目进行匹配。
3. 当前这一版本，大部分评测代码，仍然是沿用上一级的评测代码，但是可能要做一些针对性的修改，主要涉及到的是第一轮的PROMPT的输入，可能需要输入新的内容。因此可以考虑在当前目录下把上一级的主要核心评测代码:'/nas-files/ziyi/projects/proj_dsagent/repro_datamind/DataMind/datamind/eval/python'目录下的代码先复制过来，然后在基础上更改。

4. 代码需要更改的地方和原因：
考虑到这次我们加入了多模态图片内容，因此在PROMPT上需要加入新的内容：特别的，本次实验的基础模型是纯文本LLM, 因此我会使用VLM先将图片进行一些caption，之后我会试图保存这些Text 模态的caption 在某个JSON文件中，在eval中，组成PROMPT的时候需要附带上这个caption。
同时，可能也需要复制llmjudge的相关代码：'/nas-files/ziyi/projects/proj_dsagent/repro_datamind/scripts/judge_tablebench_eval.py'
届时我会修改llm judge的PROMPT。

5. 我会把只有Anomaly Detection 的题目，生成一个新的JSON，当然格式与之前接受输入的JSON格式一致。然后同时，我也建议你将../scripts里面的shell脚本都看一遍，复制过来到本地，然后撰写脚本内容，以能够做到在repro这一目录层级就可以通过shell来运行实验。outputs等等其他的内容都要放到'./'下面，即都放在本级，而不是上一级，因为我们这次是一个重要实验。

6. 如果有没有涉及到的地方，或者没有指示的地方，请按照常规的方便程度来即可。比如，似乎可能要联系csv？你可以选择通过修改代码里面的地址，使得能够正确的在运行中读入到？也可以想其他办法。同时，如果有什么需要预处理的地方，请你也可以在后续指出来，比如在未来你可能要在本级目录写的README？

---
得到了data/vlm 后的更新：
1. 现在已经有了一些caption，那么后面，需要在真正的eval的时候，在PROMPT里面进行加入。从/nas-files/ziyi/projects/proj_dsagent/repro_datamind/anomaly_experiments/data/vlm/vlm_captions.jsonl 里面读取id和以下JSON字段：
{"salient_observations": [...],
  "visual_summary": "...",
  "uncertainties": [...]}  
然后不妨以 字符串的形式直接 介入到PROMPT里面。

2. 
认可你的回答：
建议注入形式：
An auxiliary visual-perception summary derived from a chart of this table is
provided below. Use it to guide your analysis, but verify exact values against
the CSV because the visual summary may contain approximate or uncertain
observations.

Visual evidence:
{
  "salient_observations": [...],
  "visual_summary": "...",
  "uncertainties": [...]
}
关于eval PROMPT的更改问题，请参照我的思路：
首先在SYSTEM里面的适当位置加入（转化为English）：
你可以参考 <vision></vision>里面的视觉内容，作为你判断的依据，但是，视觉内容只是参考。

然后在User那里，Question结束那里加一句：如果是Anomaly Detection问题，也有可能不存在异常点，如果不存在，可以直接输出不存在异常点及原因。
然后，记得把视觉信息用vision 这一对<></>包裹起来。

