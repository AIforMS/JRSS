/**
* edit online - https://jshare.com.cn/public/S0G6jk
* API doc - https://api.highcharts.com.cn/highcharts/index.html
*/
Highcharts.chart('container', {
	chart: {
		type: 'boxplot'
	},
	title: {
		text: 'TCIA Registration'
	},
	legend: {
		itemStyle: {
			color: 'black',
			fontWeight: '0',
			fontSize: '10px'
		},
		enabled: true,
		// floating: true,
		layout: 'vertical',  // proximate，vertical，horizontal
		floating: true,
		align: 'left',
		verticalAlign: 'top',
		borderWidth: 1,
		borderColor: '#A9A9A9',
		backgroundColor: '#FFFFFF',
		x: 62,
		y: 131,
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
		// title: {
		// 	text: 'Experiment No.'
		// }
		height: '60%',
	},
	yAxis: {
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
			text: 'Reg-DSC (N=5)'
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
			whiskerWidth: 0.8,
			enableMouseTracking: false,
		}
	},
	series: [
		// 每个字典对象是一个 baseline，data 属性里是每个数组对应一个器官的 DSC 值的 box 分布
		{
			color: Highcharts.getOptions().colors[6],
			fillColor: Highcharts.getOptions().colors[6],
			name: 'ANTs (SyN)',
			data: [
				[0.696, 0.756, 0.836, 0.886, 0.916],
				[0.556, 0.626, 0.696, 0.786, 0.836],
				[0.576, 0.636, 0.676, 0.776, 0.816],
				[0.266, 0.416, 0.506, 0.606, 0.706],
				[0.196, 0.261, 0.331, 0.401, 0.441],
				[0.206, 0.256, 0.296, 0.336, 0.376],
				[0.116, 0.206, 0.251, 0.296, 0.326],
				[0.076, 0.106, 0.156, 0.186, 0.236]
			]
		}, {
			color: Highcharts.getOptions().colors[0],
			fillColor: Highcharts.getOptions().colors[0],
			name: 'VoxelMorph',
			data: [
				[0.69, 0.74, 0.78, 0.87, 0.90],
				[0.54, 0.61, 0.65, 0.75, 0.80],
				[0.62, 0.68, 0.72, 0.76, 0.80],
				[0.25, 0.36, 0.45, 0.55, 0.65],
				[0.21, 0.245, 0.305, 0.365, 0.405],
				[0.17, 0.22, 0.26, 0.30, 0.34],
				[0.11, 0.16, 0.205, 0.26, 0.29],
				[0.04, 0.07, 0.12, 0.15, 0.20],
			]
		}, {
			color: '#FFD700',  // label颜色
			fillColor: '#FFD700',  // 箱线图颜色
			name: "Mono-Net",
			data: [
				[0.71, 0.76, 0.80, 0.87, 0.89],
				[0.55, 0.63, 0.68, 0.77, 0.82],
				[0.64, 0.70, 0.74, 0.78, 0.82],
				[0.32, 0.41, 0.50, 0.58, 0.69],
				[0.19, 0.24, 0.30, 0.36, 0.39],
				[0.18, 0.23, 0.29, 0.33, 0.38],
				[0.07, 0.15, 0.225, 0.27, 0.32],
				[0.05, 0.08, 0.11, 0.14, 0.19],
			]
		}, {
			color: '#FFA07A',
			fillColor: '#FFA07A',
			name: 'DeepAtlas',
			data: [
				[0.703, 0.753, 0.833, 0.883, 0.913],
				[0.553, 0.623, 0.693, 0.763, 0.813],
				[0.633, 0.693, 0.733, 0.773, 0.813],
				[0.263, 0.373, 0.443, 0.563, 0.663],
				[0.223, 0.258, 0.318, 0.378, 0.418],
				[0.183, 0.233, 0.273, 0.313, 0.353],
				[0.123, 0.173, 0.218, 0.273, 0.303],
				[0.053, 0.083, 0.133, 0.163, 0.213],
			]
		}, {
			color: '#B0C4DE',
			fillColor: '#B0C4DE',
			name: "RSegNet",
			data: [
				[0.70, 0.78, 0.84, 0.88, 0.92],
				[0.58, 0.66, 0.72, 0.77, 0.81],
				[0.65, 0.699, 0.74, 0.78, 0.82],
				[0.335, 0.40, 0.47, 0.56, 0.70],
				[0.20, 0.24, 0.31, 0.36, 0.41],
				[0.21, 0.25, 0.31, 0.35, 0.385],
				[0.095, 0.175, 0.23, 0.27, 0.32],
				[0.06, 0.11, 0.15, 0.17, 0.22],
			]
		}, {
			color: '#32CD32',
			fillColor: '#32CD32',
			name: "(Our) JRSS",
			data: [
				[0.71, 0.78, 0.83, 0.88, 0.91],
				[0.60, 0.67, 0.72, 0.79, 0.81],
				[0.70, 0.74, 0.77, 0.80, 0.84],
				[0.38, 0.46, 0.56, 0.62, 0.72],
				[0.25, 0.30, 0.37, 0.40, 0.43],
				[0.22, 0.26, 0.30, 0.35, 0.38],
				[0.10, 0.17, 0.245, 0.28, 0.33],
				[0.08, 0.10, 0.13, 0.16, 0.21],
			]
		},
	]
});
