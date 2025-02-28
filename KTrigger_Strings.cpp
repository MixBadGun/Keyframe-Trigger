/*******************************************************************/
/*                                                                 */
/*                      ADOBE CONFIDENTIAL                         */
/*                   _ _ _ _ _ _ _ _ _ _ _ _ _                     */
/*                                                                 */
/* Copyright 2007-2023 Adobe Inc.                                  */
/* All Rights Reserved.                                            */
/*                                                                 */
/* NOTICE:  All information contained herein is, and remains the   */
/* property of Adobe Inc. and its suppliers, if                    */
/* any.  The intellectual and technical concepts contained         */
/* herein are proprietary to Adobe Inc. and its                    */
/* suppliers and may be covered by U.S. and Foreign Patents,       */
/* patents in process, and are protected by trade secret or        */
/* copyright law.  Dissemination of this information or            */
/* reproduction of this material is strictly forbidden unless      */
/* prior written permission is obtained from Adobe Inc.            */
/* Incorporated.                                                   */
/*                                                                 */
/*******************************************************************/

#include "KTrigger.h"

typedef struct {
	A_u_long	index;
	A_char		str[380];
} TableString;



TableString		g_strs[StrID_NUMTYPES] = {
	StrID_NONE,						"",
	StrID_Name,						PLUGIN_ID,
	StrID_Description,				"通过关键帧触发以克隆图层的效果, Made by 坏枪.",
	StrID_Gain_Param_Name,			"控制器",
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


char	*GetStringPtr(int strNum)
{
	return g_strs[strNum].str;
}