/**
* edit online - https://jshare.com.cn/public/S0G6jk
* API doc - https://api.highcharts.com.cn/highcharts/index.html
*/
// Highcharts.setOptions({
// 	// 全局设置一系列颜色，series 中的颜色依次排列
// 	colors: ['#058DC7', '#50B432', '#ED561B', '#DDDF00', '#24CBE5', '#64E572', '#FF9655', '#FFF263', '#6AF9C4']
// });
Highcharts.chart('container', {
	chart: {
		type: 'boxplot'
	},
	title: {
		text: 'LPBA40 Segmentation'
	},
	legend: {
		itemStyle: {
			color: 'black',
			fontWeight: '0',
			fontSize: '10px'
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
		x: 63,
		y: 130,
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
		// ['Frontal', 'Parietal', 'Limbic', 'Occipital', 'Temporal', 'Brainstem', 'Putamen', 'Caudate', 'Insular Cortex', 'Cerebellum'],
		categories: ['额叶', '顶叶', '边缘', '枕叶', '颞叶', '脑干', '壳核', '尾状核', '岛状皮质', '小脑'],
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
		min: 0.2,
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
			// color:
			fillColor: Highcharts.getOptions().colors[0],
			// color: Highcharts.getOptions().colors[0],
			// marker: {  // 离群点
			// 	fillColor: 'blue',
			// 	lineWidth: 1,
			// 	lineColor: Highcharts.getOptions().colors[0]
			// },
			name: 'VM-UNet',
			data: [
				[0.76, 0.81, 0.85, 0.87, 0.90],
				[0.60, 0.70, 0.76, 0.821, 0.88],
				[0.68, 0.73, 0.77, 0.821, 0.86],
				[0.65, 0.70, 0.74, 0.78, 0.83],
				[0.66, 0.75, 0.80, 0.86, 0.883],
				[0.81, 0.855, 0.875, 0.895, 0.92],
				[0.70, 0.76, 0.79, 0.82, 0.86],
				[0.69, 0.73, 0.75, 0.81, 0.85],
				[0.65, 0.70, 0.73, 0.77, 0.80],
				[0.60, 0.70, 0.74, 0.821, 0.86],
			]
		}, {
			color: '#FFD700',  // label颜色
			fillColor: '#FFD700',  // 箱线图颜色
			name: "Mono-Net",
			data: [
				[0.75, 0.80, 0.84, 0.87, 0.89],
				[0.62, 0.70, 0.76, 0.811, 0.87],
				[0.70, 0.75, 0.79, 0.841, 0.88],
				[0.67, 0.71, 0.753, 0.793, 0.833],
				[0.65, 0.74, 0.80, 0.86, 0.89],
				[0.81, 0.850, 0.87, 0.89, 0.91],
				[0.72, 0.75, 0.78, 0.81, 0.85],
				[0.68, 0.73, 0.77, 0.82, 0.86],
				[0.62, 0.68, 0.73, 0.77, 0.813],
				[0.65, 0.70, 0.76, 0.811, 0.86],
			]
		}, {
			color: '#FFA07A',  // label颜色
			fillColor: '#FFA07A',  // 箱线图颜色
			name: "DeepAtlas",
			data: [
				[0.76, 0.81, 0.85, 0.88, 0.91],
				[0.63, 0.71, 0.77, 0.831, 0.88],
				[0.71, 0.76, 0.80, 0.851, 0.89],
				[0.675, 0.715, 0.758, 0.798, 0.838],
				[0.655, 0.745, 0.805, 0.865, 0.895],
				[0.815, 0.8505, 0.875, 0.895, 0.915],
				[0.68, 0.76, 0.79, 0.82, 0.86],
				[0.69, 0.75, 0.78, 0.83, 0.88],
				[0.63, 0.69, 0.74, 0.78, 0.823],
				[0.64, 0.71, 0.77, 0.821, 0.87],
			]
		}, {
			color: '#B0C4DE',
			fillColor: '#B0C4DE',
			name: "RSegNet",
			data: [
				[0.80, 0.83, 0.86, 0.89, 0.92],
				[0.635, 0.71, 0.775, 0.83, 0.89],
				[0.73, 0.79, 0.83, 0.87, 0.905],
				[0.65, 0.71, 0.75, 0.80, 0.84],
				[0.68, 0.77, 0.83, 0.875, 0.895],
				[0.83, 0.86, 0.89, 0.91, 0.925],
				[0.72, 0.765, 0.80, 0.845, 0.885],
				[0.70, 0.75, 0.81, 0.84, 0.89],
				[0.665, 0.71 , 0.76, 0.795, 0.84],
				[0.68, 0.75, 0.79, 0.84, 0.88],
			]
		}, {
			color: '#32CD32',
			fillColor: '#32CD32',
			name: "(Our) JRSS",
			data: [
				[0.82, 0.85, 0.88, 0.91, 0.93],
				[0.67, 0.73, 0.79, 0.84, 0.89],
				[0.74, 0.81, 0.83, 0.875, 0.90],
				[0.69, 0.74, 0.77, 0.81, 0.85],
				[0.69, 0.78, 0.83, 0.88, 0.90],
				[0.84, 0.869, 0.89, 0.91, 0.93],
				[0.73, 0.77, 0.81, 0.84, 0.88],
				[0.71, 0.76, 0.80, 0.85, 0.892],
				[0.68, 0.73 , 0.79, 0.81, 0.85],
				[0.69, 0.757, 0.80, 0.84, 0.90],
			]
		},
	],
});