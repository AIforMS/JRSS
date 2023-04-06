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
		text: 'LPBA40 Registration'
	},
	legend: {
		itemStyle: {
			color: 'black',
			fontWeight: '0',
			fontSize: '10px'
		},
		enabled: true,
		// floating: true,
		// layout: 'vertical',
		floating: true,
		align: 'left',
		verticalAlign: 'top',
		borderWidth: 1,
		borderColor: '#A9A9A9',
		backgroundColor: '#FFFFFF',
		x: 80,
		y: 170,
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
		// ['Frontal', 'Parietal', 'Limbic', 'Occipital', 'Temporal', 'Brainstem', 'Putamen', 'Caudate', 'Insular Cortex', 'Cerebellum'],,
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
			whiskerLength: '60%',
			whiskerWidth: 0.8,
			enableMouseTracking: false,
		}
	},
	series: [
		// 每个字典对象是一个 baseline，data 属性里是每个数组对应一个器官的 DSC 值的 box 分布
		{
			color: Highcharts.getOptions().colors[6],
			fillColor: Highcharts.getOptions().colors[6],
			// color: Highcharts.getOptions().colors[0],
			name: 'ANTs (SyN)',
			data: [
				[0.73 , 0.8  , 0.84 , 0.87 , 0.9  ],
				[0.61 , 0.68 , 0.74 , 0.795, 0.87 ],
				[0.61 , 0.671, 0.701, 0.74 , 0.785],
				[0.64 , 0.69 , 0.73 , 0.77 , 0.81 ],
				[0.68 , 0.76 , 0.8  , 0.86 , 0.903],
				[0.81 , 0.845, 0.865, 0.885, 0.91 ],
				[0.61 , 0.67 , 0.73 , 0.77 , 0.83 ],
				[0.56 , 0.68 , 0.73 , 0.8  , 0.86 ],
				[0.41 , 0.59 , 0.68 , 0.76 , 0.87 ],
				[0.52 , 0.66 , 0.73 , 0.778, 0.83 ]
			],
		}, {
			color: Highcharts.getOptions().colors[0],
			fillColor: Highcharts.getOptions().colors[0],
			name: 'VoxelMorph',
			data: [
				[0.71, 0.77, 0.81, 0.84, 0.888],
				[0.56, 0.64, 0.70, 0.755, 0.82],
				[0.57, 0.641, 0.671, 0.720, 0.77],
				[0.62, 0.67, 0.71, 0.75, 0.79],
				[0.62, 0.71, 0.78, 0.84, 0.863],
				[0.77, 0.815, 0.835, 0.855, 0.88],
				[0.56, 0.63, 0.71, 0.76, 0.82],
				[0.55, 0.66, 0.70, 0.78, 0.84],
				[0.41, 0.62 , 0.68, 0.76, 0.86],
				[0.47, 0.61, 0.70, 0.758, 0.81],
			],
		}, {
			color: '#FFA07A',
			fillColor: '#FFA07A',
			name: 'DeepAtlas',
			data: [
				[0.697, 0.757, 0.797, 0.827, 0.875],
				[0.547, 0.627, 0.687, 0.742, 0.807],
				[0.557, 0.628, 0.658, 0.707, 0.757],
				[0.607, 0.657, 0.697, 0.737, 0.777],
				[0.607, 0.697, 0.767, 0.827, 0.85 ],
				[0.757, 0.802, 0.822, 0.842, 0.867],
				[0.547, 0.617, 0.697, 0.747, 0.807],
				[0.537, 0.647, 0.687, 0.767, 0.827],
				[0.397, 0.607, 0.667, 0.747, 0.847],
				[0.457, 0.597, 0.687, 0.745, 0.797]
			],
		}, {
			color: '#FFD700',  // label颜色
			fillColor: '#FFD700',  // 箱线图颜色
			name: "Mono-Net",
			data: [
				[0.74, 0.79, 0.83, 0.86, 0.88],
				[0.55, 0.63, 0.70, 0.76, 0.82],
				[0.581, 0.645, 0.675, 0.720, 0.79],
				[0.62, 0.66, 0.72, 0.76, 0.80],
				[0.65, 0.74, 0.80, 0.86, 0.88],
				[0.78, 0.820, 0.84, 0.86, 0.885],
				[0.534, 0.61, 0.69, 0.74, 0.82],
				[0.60, 0.70, 0.75, 0.81, 0.833],
				[0.45, 0.63 , 0.69, 0.77, 0.87],
				[0.51, 0.65, 0.71, 0.78, 0.84],
			]
		}, {
			color: '#B0C4DE',
			fillColor: '#B0C4DE',
			name: "RSegNet",
			data: [
				[0.72, 0.78, 0.84, 0.87, 0.91],
				[0.58, 0.65, 0.73, 0.77, 0.82],
				[0.57, 0.63, 0.67, 0.72, 0.795],
				[0.61, 0.67, 0.71, 0.76, 0.80],
				[0.64, 0.73, 0.80, 0.85, 0.885],
				[0.79, 0.82, 0.84, 0.875, 0.895],
				[0.58, 0.66, 0.70, 0.75, 0.80],
				[0.63, 0.72, 0.78, 0.82, 0.84],
				[0.45, 0.65 , 0.74, 0.80, 0.88],
				[0.58, 0.68, 0.73, 0.79, 0.84],
			]
		}, {
			color: '#32CD32',
			fillColor: '#32CD32',
			name: "(Our) JRSS",
			data: [
				[0.78, 0.82, 0.86, 0.88, 0.90],
				[0.62, 0.68, 0.72, 0.78, 0.84],
				[0.59, 0.65, 0.68, 0.725, 0.79],
				[0.65, 0.70, 0.73, 0.77, 0.81],
				[0.67, 0.76, 0.81, 0.86, 0.88],
				[0.80, 0.829, 0.85, 0.87, 0.89],
				[0.60, 0.67, 0.71, 0.76, 0.81],
				[0.64, 0.74, 0.78, 0.83, 0.852],
				[0.47, 0.66 , 0.72, 0.81, 0.87],
				[0.64, 0.707, 0.75, 0.79, 0.85],
			]
		}
	],
});