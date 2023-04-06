/**
 * highcharts - https://www.highcharts.com.cn/demo/highcharts/box-plot
 * edit online - https://jshare.com.cn/demos/hhhhiQ
 * API doc - https://api.highcharts.com.cn/highcharts/index.html
 *
 * 箱形图是一种通过五个数字汇总来描述数据组的便捷方法:
 * 最小观测值 (样本最小值)，下四分位数 (Q1)，中位数 (Q2)，上四分位数 (Q3) 和最大观测值 (样本最大值)
*/
Highcharts.chart('container', {
	chart: {
		type: 'boxplot'
	},
	title: {
		text: 'TCIA Segmentation'
	},
	legend: {
		itemStyle: {
			color: 'black',
			fontWeight: '0',
			fontSize: '11px'
		},
		enabled: true,
		// floating: true,
		layout: 'vertical',
		floating: true,
		align: 'left',
		verticalAlign: 'top',
		borderWidth: 1,
		borderColor: '#A9A9A9',
		backgroundColor: '#FFFFFF',
		x: 65,
		y: 118,
	},
	xAxis: {
		tickColor: 'black',
		tickWidth: 1,
		tickLength: 5,
		tickmarkPlacement: 'on',
		labels: {
			style: {
				color: 'black',
				fontSize: '11px'
			}
		},
		lineColor: 'black',
		categories: ['肝脏', '左肾', '脾脏', '胃', '胰腺', '胆囊', '食道', '十二指肠'],
		// ['Liver', 'Left Kidney', 'Spleen', 'Stomach', 'Pancreas', 'Gallbladder', 'Esophagus', 'Duodenum'],
		// title: {
		// 	text: 'Experiment No.'
		// }
		height: '60%',
	},
	yAxis: {
		max: 1,
		tickColor: 'black',
		tickWidth: 1,
		tickLength: 5,
		labels: {
			style: {
				color: 'black',
				fontSize: '13px'
			}
		},
		lineColor: 'black',
		lineWidth: 1,
		// ceiling: 1.0,  // 天花板
		height: '60%',
		min: 0,
		tickInterval: 0.2,  // 刻度间距
		title: {
			style: {
				color: 'black',
				fontSize: '13px'
			},
			text: 'Seg-DSC (N=5)'
		}
	},
	plotOptions: {
		boxplot: {
			// pointRange: 1,  // 每组的间距，默认1
			// pointPadding: 0.15,  // 每个箱子的间距，默认0.1
			// // groupPadding: 0.9,  // 每个的间距默认0.5
			// pointWidth: 4,  // 每个箱子的宽度，单位为px，默认 undefined
			// // fillColor: '#F0F0E0',
			lineColor: 'black',  // 箱子框线颜色
			lineWidth: 0.5,  // 箱子框线粗细
			medianColor: 'black',  // 中位数横线颜色
			medianWidth: 1,
			stemColor: '#A63400',  // 虚线长条颜色
			stemDashStyle: 'dot',
			stemWidth: 1,
			whiskerColor: 'blue',  // 上下小横线颜色
			whiskerLength: '80%',
			whiskerWidth: 0.8
		}
	},
	series: [
		// 每个字典对象是一个 baseline，data 属性里是每个数组对应一个器官的 DSC 值的 box 分布
		{
			fillColor: Highcharts.getOptions().colors[0],
			name: 'VM-UNet',
			data: [  // 样本最小值，下四分位数 (Q1)，中位数 (Q2)，上四分位数 (Q3) 和样本最大值
				[0.76, 0.82, 0.86, 0.91, 0.95],
				[0.73, 0.79, 0.85, 0.90, 0.95],
				[0.72, 0.78, 0.84, 0.88, 0.921],
				[0.40, 0.51, 0.63, 0.73, 0.835],
				[0.45, 0.555, 0.615, 0.675, 0.71],
				[0.42, 0.50, 0.57, 0.64, 0.68],
				[0.40, 0.46, 0.525, 0.56, 0.59],
				[0.29, 0.38, 0.44, 0.52, 0.55],
			]
		}, {
			color: '#FFD700',  // label颜色
			fillColor: '#FFD700',  // 箱线图颜色
			name: "Mono-Net",
			data: [
				[0.77, 0.81, 0.88, 0.92, 0.94],
				[0.70, 0.76, 0.84, 0.89, 0.943],
				[0.72, 0.78, 0.84, 0.88, 0.92],
				[0.47, 0.56, 0.66, 0.74, 0.84],
				[0.42, 0.52, 0.58, 0.65, 0.69],
				[0.40, 0.48, 0.56, 0.63, 0.70],
				[0.38, 0.45, 0.535, 0.59, 0.63],
				[0.30, 0.37, 0.45, 0.50, 0.54],
			]
		}, {
			color: '#FFA07A',  // label颜色
			fillColor: '#FFA07A',  // 箱线图颜色
			name: "DeepAtlas",
			data: [
				[0.81 , 0.85 , 0.89  , 0.92 , 0.94 ],
				[0.72 , 0.78 , 0.86 , 0.91 , 0.963],
				[0.74 , 0.8  , 0.86 , 0.9  , 0.94 ],
				[0.49 , 0.58 , 0.68 , 0.76 , 0.86 ],
				[0.45 , 0.54 , 0.6  , 0.67 , 0.71 ],
				[0.42 , 0.5  , 0.58 , 0.65 , 0.7  ],
				[0.4  , 0.47 , 0.555, 0.61 , 0.65 ],
				[0.32 , 0.39 , 0.47 , 0.52 , 0.56 ]
			]
		}, {
			color: '#B0C4DE',
			fillColor: '#B0C4DE',
			name: "RSegNet",
			data: [
				[0.80, 0.85, 0.90, 0.935, 0.955],
				[0.745, 0.80, 0.86, 0.91, 0.94],
				[0.73, 0.80, 0.85, 0.895, 0.94],
				[0.51, 0.60, 0.70, 0.77, 0.865],
				[0.51, 0.58, 0.65, 0.70, 0.75],
				[0.44, 0.52, 0.59, 0.66, 0.72],
				[0.37, 0.47, 0.54, 0.60, 0.66],
				[0.30, 0.405, 0.47, 0.53, 0.56],
			]
		}, {
			color: '#32CD32',
			fillColor: '#32CD32',
			name: "(Our) JRSS",
			data: [
				[0.82, 0.87, 0.91, 0.94, 0.96],
				[0.77, 0.81, 0.86, 0.92, 0.95],
				[0.78, 0.82, 0.86, 0.90, 0.94],
				[0.60, 0.68, 0.77, 0.84, 0.88],
				[0.55, 0.61, 0.65, 0.71, 0.77],
				[0.52, 0.59, 0.65, 0.73, 0.79],
				[0.40, 0.50, 0.57, 0.63, 0.66],
				[0.31, 0.40, 0.47, 0.52, 0.55],
			]
		},
	]
});
