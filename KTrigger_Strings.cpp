/*	
*   本土化区域				Localization
*	可在这里补充本地化语言	 You can add your local language here
*/

#include "KTrigger_Strings.h"

typedef struct {
	A_u_long	index;
	A_char		str[380];
} TableString;

// 简体中文 zh_CN
TableString		zh_strs[StrID_NUMTYPES] = {
	StrID_NONE,						"",
	StrID_Name,						PLUGIN_ID,
	StrID_Description,				"通过关键帧触发以克隆图层的效果, Made by 坏枪.\n\n项目开源在 https://github.com/MixBadGun/Keyframe-Trigger",
	StrID_Gain_Param_Name,			"控制器",
	StrID_Count_Param_Name,			"数量",
	StrID_Start_Time_Param_Name,	"起始时间",
	StrID_Playrate_Param_Name,		"播放速率",
	StrID_Is_Control_Param_Name,	"控制器影响偏移",
	StrID_Is_Frozen_Name,			"生成后冻结状态",
	StrID_Anchor_Param_Name,		"锚点",
	StrID_Scale_Size_Name,			"缩放大小",
	StrID_Is_Flip_Param_Name,		"水平翻转",
	StrID_Switch_Param_Name,		"偏移模式",
	StrID_Switch_Popup_Choices,		"位置|"
									"旋转",
	StrID_Return_Group_Name,		"序列属性",
	StrID_Return_Param_Name,		"折返来回",
	StrID_Return2_Param_Name,		"（折返）遇首尾不重复",
	StrID_Offset_Group_Name,		"位置属性",
	StrID_Offset_Param_Name,		"位置偏移",
	StrID_Offset_Count_Param_Name,	"序列数量",
	StrID_Rotate_Group_Name,		"旋转属性",
	StrID_Rotate_Param_Name,		"旋转角度",
	StrID_Rotate_Scale_Param_Name,	"旋转缩放",
	StrID_Single_Scale_Group_Name,	"累积缩放属性",
	StrID_Single_Scale_Param_Name,	"累积缩放",
	StrID_Max_Dur_Name,				"最大持续时间 (秒)",
	StrID_Xfer_Mode_Name,			"混合模式",
	StrID_Xfer_Mode_Choices,		"向前覆盖|"
									"向后覆盖|"
									"(-|"
									"溶解|"
									"相加|"
									"相乘|"
									"屏幕|"
									"叠加|"
									"柔光|"
									"强光|"
									"变暗|"
									"变亮|"
									"差异|"
									"色相|"
									"饱和度|"
									"颜色|"
									"发光度|"
									"Alpha 添加|"
									"排除|"
									"相减|"
									"相除",
	StrID_Second_Layer_Name,		"第二图层",
	StrID_Random_Name,				"用来刷新渲染的笨笨滑条",
};

// English en_US
TableString		en_strs[StrID_NUMTYPES] = {
	StrID_NONE,						"",
	StrID_Name,						PLUGIN_ID,
	StrID_Description,				"Clone layers triggered by keyframes, Made by BadGun.\n\nOpen source at https://github.com/MixBadGun/Keyframe-Trigger",
	StrID_Gain_Param_Name,			"Controller",
	StrID_Count_Param_Name,			"Count",
	StrID_Start_Time_Param_Name,	"Start Time",
	StrID_Playrate_Param_Name,		"Play Rate",
	StrID_Is_Control_Param_Name,	"Controller Affects Offset",
	StrID_Is_Frozen_Name,			"Freeze State After Generation",
	StrID_Anchor_Param_Name,		"Anchor Point",
	StrID_Scale_Size_Name,			"Scale Size",
	StrID_Is_Flip_Param_Name,		"Flip Horizontal",
	StrID_Switch_Param_Name,		"Offset Mode",
	StrID_Switch_Popup_Choices,		"Position|"
									"Rotation",
	StrID_Return_Group_Name,		"Sequence Properties",
	StrID_Return_Param_Name,		"Ping Pong",
	StrID_Return2_Param_Name,		"(Ping Pong) No Repeat at Ends",
	StrID_Offset_Group_Name,		"Position Properties",
	StrID_Offset_Param_Name,		"Position Offset",
	StrID_Offset_Count_Param_Name,	"Sequence Count",
	StrID_Rotate_Group_Name,		"Rotation Properties",
	StrID_Rotate_Param_Name,		"Rotation Angle",
	StrID_Rotate_Scale_Param_Name,	"Rotation Scale",
	StrID_Single_Scale_Group_Name,	"Cumulative Scale Properties",
	StrID_Single_Scale_Param_Name,	"Cumulative Scale",
	StrID_Max_Dur_Name,				"Max Duration (seconds)",
	StrID_Xfer_Mode_Name,			"Blend Mode",
	StrID_Xfer_Mode_Choices,		"Foreground Over|"
									"Background Over|"
									"(-|"
									"Dissolve|"
									"Add|"
									"Multiply|"
									"Screen|"
									"Overlay|"
									"Soft Light|"
									"Hard Light|"
									"Darken|"
									"Lighten|"
									"Difference|"
									"Hue|"
									"Saturation|"
									"Color|"
									"Luminosity|"
									"Alpha Add|"
									"Exclusion|"
									"Subtract|"
									"Divide",
	StrID_Second_Layer_Name,		"Second Layer",
	StrID_Random_Name,				"Dummy Slider for Refresh Rendering",
};


/*
* ############################
* 此处需要翻译。
* This part needs translation.
* ############################
*/ 
// 한국어 ko_KR
TableString		ko_strs[StrID_NUMTYPES] = {
	StrID_NONE,						"",
	StrID_Name,						PLUGIN_ID,
	StrID_Description,				"通过关键帧触发以克隆图层的效果, Made by 坏枪.\n\n项目开源在 https://github.com/MixBadGun/Keyframe-Trigger",
	StrID_Gain_Param_Name,			"控制器",
	StrID_Count_Param_Name,			"数量",
	StrID_Start_Time_Param_Name,	"起始时间",
	StrID_Playrate_Param_Name,		"播放速率",
	StrID_Is_Control_Param_Name,	"控制器影响偏移",
	StrID_Is_Frozen_Name,			"生成后冻结状态",
	StrID_Anchor_Param_Name,		"锚点",
	StrID_Scale_Size_Name,			"缩放大小",
	StrID_Is_Flip_Param_Name,		"水平翻转",
	StrID_Switch_Param_Name,		"偏移模式",
	StrID_Switch_Popup_Choices,		"位置|"
									"旋转",
	StrID_Return_Group_Name,		"序列属性",
	StrID_Return_Param_Name,		"折返来回",
	StrID_Return2_Param_Name,		"（折返）遇首尾不重复",
	StrID_Offset_Group_Name,		"位置属性",
	StrID_Offset_Param_Name,		"位置偏移",
	StrID_Offset_Count_Param_Name,	"序列数量",
	StrID_Rotate_Group_Name,		"旋转属性",
	StrID_Rotate_Param_Name,		"旋转角度",
	StrID_Rotate_Scale_Param_Name,	"旋转缩放",
	StrID_Single_Scale_Group_Name,	"累积缩放属性",
	StrID_Single_Scale_Param_Name,	"累积缩放",
	StrID_Max_Dur_Name,				"最大持续时间 (秒)",
	StrID_Xfer_Mode_Name,			"混合模式",
	StrID_Xfer_Mode_Choices,		"向前覆盖|"
									"向后覆盖|"
									"(-|"
									"溶解|"
									"相加|"
									"相乘|"
									"屏幕|"
									"叠加|"
									"柔光|"
									"强光|"
									"变暗|"
									"变亮|"
									"差异|"
									"色相|"
									"饱和度|"
									"颜色|"
									"发光度|"
									"Alpha 添加|"
									"排除|"
									"相减|"
									"相除",
	StrID_Second_Layer_Name,		"第二图层",
	StrID_Random_Name,				"用来刷新渲染的笨笨滑条",
};

/*
* ############################
* 此处需要翻译。
* This part needs translation.
* ############################
*/ 
// 日本語 ja_JP
TableString		ja_strs[StrID_NUMTYPES] = {
	StrID_NONE,						"",
	StrID_Name,						PLUGIN_ID,
	StrID_Description,				"通过关键帧触发以克隆图层的效果, Made by 坏枪.\n\n项目开源在 https://github.com/MixBadGun/Keyframe-Trigger",
	StrID_Gain_Param_Name,			"控制器",
	StrID_Count_Param_Name,			"数量",
	StrID_Start_Time_Param_Name,	"起始时间",
	StrID_Playrate_Param_Name,		"播放速率",
	StrID_Is_Control_Param_Name,	"控制器影响偏移",
	StrID_Is_Frozen_Name,			"生成后冻结状态",
	StrID_Anchor_Param_Name,		"锚点",
	StrID_Scale_Size_Name,			"缩放大小",
	StrID_Is_Flip_Param_Name,		"水平翻转",
	StrID_Switch_Param_Name,		"偏移模式",
	StrID_Switch_Popup_Choices,		"位置|"
									"旋转",
	StrID_Return_Group_Name,		"序列属性",
	StrID_Return_Param_Name,		"折返来回",
	StrID_Return2_Param_Name,		"（折返）遇首尾不重复",
	StrID_Offset_Group_Name,		"位置属性",
	StrID_Offset_Param_Name,		"位置偏移",
	StrID_Offset_Count_Param_Name,	"序列数量",
	StrID_Rotate_Group_Name,		"旋转属性",
	StrID_Rotate_Param_Name,		"旋转角度",
	StrID_Rotate_Scale_Param_Name,	"旋转缩放",
	StrID_Single_Scale_Group_Name,	"累积缩放属性",
	StrID_Single_Scale_Param_Name,	"累积缩放",
	StrID_Max_Dur_Name,				"最大持续时间 (秒)",
	StrID_Xfer_Mode_Name,			"混合模式",
	StrID_Xfer_Mode_Choices,		"向前覆盖|"
									"向后覆盖|"
									"(-|"
									"溶解|"
									"相加|"
									"相乘|"
									"屏幕|"
									"叠加|"
									"柔光|"
									"强光|"
									"变暗|"
									"变亮|"
									"差异|"
									"色相|"
									"饱和度|"
									"颜色|"
									"发光度|"
									"Alpha 添加|"
									"排除|"
									"相减|"
									"相除",
	StrID_Second_Layer_Name,		"第二图层",
	StrID_Random_Name,				"用来刷新渲染的笨笨滑条",
};


char	*GetStringPtr(int strNum, A_char* lang_tagZ = nullptr)
{
	if (lang_tagZ) {
		if (strcmp(lang_tagZ, "zh_CN") == 0){
			return zh_strs[strNum].str;
		}
		if (strcmp(lang_tagZ, "en_US") == 0) {
			return en_strs[strNum].str;
		}
		if (strcmp(lang_tagZ, "fr_FR") == 0) {
			// TODO...
		}
		if (strcmp(lang_tagZ, "de_DE") == 0) {
			// TODO...
		}
		if (strcmp(lang_tagZ, "it_IT") == 0) {
			// TODO...
		}
		if (strcmp(lang_tagZ, "ja_JP") == 0) {
			// TODO...
		}
		if (strcmp(lang_tagZ, "ko_KR") == 0) {
			// TODO...
		}
		if (strcmp(lang_tagZ, "es_ES") == 0) {
			// TODO...
		}
	}
	return en_strs[strNum].str;
}