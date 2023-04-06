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
		text: 'CT -> MR',
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
				[0.794, 0.84, 0.875, 0.9, 0.934],  // Liver  deeds
				[0.562, 0.747, 0.789, 0.843, 0.885],  // Spleen
				[0.676, 0.719, 0.776, 0.829, 0.848],  // Right-kidney
				[0.581, 0.703, 0.764, 0.82, 0.831],  // Letf-kidney
			]
		}, {  // data 1
			color: '#FFD700',  // label颜色
			fillColor: '#FFD700',  // 箱线图颜色
			name: "VM-2",
			data: [
				[0.651, 0.782, 0.843, 0.873, 0.898],  // Liver  771
				[0.51, 0.708, 0.765, 0.817, 0.846],  // Spleen
				[0.614, 0.67, 0.728, 0.745, 0.798],  // Right-kidney
				[0.643, 0.7, 0.738, 0.754, 0.809],  // Letf-kidney
			]
		}, {  // data 2
			color: '#FFA07A',  // label颜色
			fillColor: '#FFA07A',  // 箱线图颜色
			name: "SUITS",
			data: [
				[0.65, 0.774, 0.839, 0.86, 0.889],  // Liver  757
				[0.484, 0.708, 0.742, 0.809, 0.831],  // Spleen
				[0.581, 0.658, 0.695, 0.718, 0.781],  // Right-kidney
				[0.608, 0.686, 0.722, 0.736, 0.804],  // Letf-kidney
			]
		}, {  // data 3
			color: '#B0C4DE',
			fillColor: '#B0C4DE',
			name: "TransMorph",
			data: [
				[0.655, 0.778, 0.852, 0.87, 0.891],  // Liver  772
				[0.593, 0.713, 0.77, 0.814, 0.85],  // Spleen
				[0.64, 0.694, 0.732, 0.749, 0.803],  // Right-kidney
				[0.6, 0.697, 0.709, 0.77, 0.783],  // Letf-kidney
			]
		}, {  // data 4
			color: '#32CD32',
			fillColor: '#32CD32',
			name: "(Our) MRSS",
			data: [
				[0.721, 0.797, 0.841, 0.883, 0.916],  // Liver  785
				[0.648, 0.721, 0.766, 0.832, 0.855],  // Spleen
				[0.679, 0.698, 0.748, 0.782, 0.836],  // Right-kidney
				[0.702, 0.722, 0.771, 0.792, 0.816],  // Letf-kidney
			]
		}
	]
});
