/**
 * highcharts - https://www.highcharts.com
 * edit online - https://jshare.com.cn/demos/hhhhiQ
 *
 * 箱形图是一种通过五个数字汇总来描述数据组的便捷方法:
 * 最小观测值 (样本最小值)，下四分位数 (Q1)，中位数 (Q2)，上四分位数 (Q3) 和最大观测值 (样本最大值)
*/
Highcharts.chart('container', {
	chart: {
		type: 'boxplot'
	},

	title: {
		text: 'MR -> CT',
		style: {
            fontSize: '13px',
        }
	},

	legend: {  		  // 图例
		itemStyle: {  // 字体
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
		x: 65,  // 位置
		y: 130,
	},

	xAxis: {  // x 轴属性
		tickColor: 'black',
		tickWidth: 1,
		tickLength: 5,
		tickmarkPlacement: 'on',
		labels: {
			style: {
				color: 'black',
				fontSize: '13px',
			}
		},
		lineColor: 'black',
		categories: ['肝脏', '脾脏', '右肾', '左肾'],
		// title: {
		// 	text: 'Experiment No.'
		// },
		height: '60%',  // x 轴线上下位置
	},

	yAxis: {  // y 轴属性
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
		height: '60%',  // y 轴线长度
		min: 0,
		tickInterval: 0.2,  // 刻度间距
		title: {
			style: {
				color: 'black',
				fontSize: '13px'
			},
			text: 'DSC'
		}
	},

	plotOptions: {  // 箱线图属性
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
		/**
		 * baselines 数据
		 * 每个字典对象是一个 baseline，data 属性里是每个数组对应一个器官的 DSC 值的 box 分布
		 *
		*/
		{  // data 0
			fillColor: Highcharts.getOptions().colors[0],
			name: 'DeedsBCV',
			data: [
				/**
				 * 首先获取每个测试图中同一器官的 DSC 数据得到数组，然后计算该数组的以下数据：
				 * 样本最小值，下四分位数 (Q1)，中位数 (Q2)，上四分位数 (Q3) 和样本最大值
				 *
				*/
				[0.814, 0.866, 0.884, 0.908, 0.934],  // Liver  deeds
				[0.69, 0.757, 0.851, 0.873, 0.90],  // Spleen
				[0.745, 0.791, 0.837, 0.854, 0.888],  // Right-kidney
				[0.707, 0.784, 0.816, 0.836, 0.861],  // Letf-kidney
			]
		}, {  // data 1
			color: '#FFD700',  // label颜色
			fillColor: '#FFD700',  // 箱线图颜色
			name: "VM-2",
			data: [
				[0.696, 0.812, 0.873, 0.902, 0.916],  // Liver  744
				[0.573, 0.684, 0.765, 0.825, 0.84],  // Spleen
				[0.539, 0.666, 0.701, 0.78, 0.825],  // Right-kidney
				[0.743, 0.758, 0.784, 0.807, 0.817],  // Letf-kidney
			]
		}, {  // data 2
			color: '#FFA07A',  // label颜色
			fillColor: '#FFA07A',  // 箱线图颜色
			name: "SUITS",
			data: [
				[0.684, 0.808, 0.88, 0.894, 0.909],  // Liver  737
				[0.560, 0.707, 0.776, 0.829, 0.846],  // Spleen
				[0.542, 0.65, 0.709, 0.76, 0.787],  // Right-kidney
				[0.71, 0.73, 0.75, 0.794, 0.839],  // Letf-kidney
			]
		}, {  // data 3
			color: '#B0C4DE',
			fillColor: '#B0C4DE',
			name: "TransMorph",
			data: [
				[0.685, 0.805, 0.856, 0.901, 0.923],  // Liver  754
				[0.573, 0.749, 0.835, 0.846, 0.871],  // Spleen
				[0.645, 0.73, 0.774, 0.786, 0.846],  // Right-kidney
				[0.71, 0.75, 0.79, 0.81, 0.84],  // Letf-kidney
			]
		}, {  // data 4
			color: '#32CD32',
			fillColor: '#32CD32',
			name: "(Our) MRSS",
			data: [
				[0.745, 0.823, 0.874, 0.917, 0.933],  // Liver  763
				[0.67, 0.78, 0.828, 0.854, 0.866],  // Spleen
				[0.729, 0.772, 0.799, 0.84, 0.875],  // Right-kidney
				[0.747, 0.773, 0.79, 0.813, 0.841],  // Letf-kidney
			]
		}
	]
});
