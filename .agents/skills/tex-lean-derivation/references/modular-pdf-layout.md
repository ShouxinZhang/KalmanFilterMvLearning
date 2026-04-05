# Modular PDF Layout

适用于“主文档简洁，细节推导独立”的写法。

推荐目录：

```text
tex-derivation-note/
├── main.tex
├── preamble.tex
├── derivations/
│   ├── covariance_prediction.tex
│   ├── kalman_gain.tex
│   └── joseph_form.tex
├── pdf/
│   ├── main.pdf
│   └── derivations/
│       ├── covariance_prediction.pdf
│       ├── kalman_gain.pdf
│       └── joseph_form.pdf
└── build/
```

约定：

- `main.tex` 负责主线叙述、目录、超链接入口。
- `preamble.tex` 放公共宏、包、颜色、步骤环境。
- `derivations/*.tex` 每个文件只讲一个推导。
- 每个子推导从新页开始；如果独立编译成 PDF，就让它只依赖共享导言区。
- `pdf/` 放最终产物；`build/` 放中间文件。

建议：

- 主文档里只放“结论版”推导，细节版用超链接跳到子 PDF。
- 子 PDF 标题直接写清“在证明什么”，不要重复主文档的大段背景。
- 同一套颜色宏、步骤宏在 `main` 和子推导里保持一致。
