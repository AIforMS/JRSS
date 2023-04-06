# highcharts - https://www.highcharts.com


- 先运行 `get_boxplot_value.py` 得到箱线图数据；
  - 根据保存的 log 提取 dsc 数值，可根据具体情况获取。

- 在 `data-js` 目录下新增一个 `xx.js` 文件，复制数据进去；

- 在 `boxplots.html` 中替换 `xx.js`；

- 在浏览器中运行 `boxplots.html` 即可看到绘制的箱线图；
  - 推荐使用 VS Code 安装 Live Server 插件来运行 HTML 文件，更改数据保存即可刷新。
