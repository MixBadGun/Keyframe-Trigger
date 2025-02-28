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
	StrID_Description,				"ͨ���ؼ�֡�����Կ�¡ͼ���Ч��, Made by ��ǹ.",
	StrID_Gain_Param_Name,			"������",
	StrID_Is_Control_Param_Name,	"������Ӱ��ƫ��",
	StrID_Is_Frozen_Name,			"���ɺ󶳽�״̬",
	StrID_Anchor_Param_Name,		"ê��",
	StrID_Scale_Size_Name,			"���Ŵ�С",
	StrID_Is_Flip_Param_Name,		"ˮƽ��ת",
	StrID_Switch_Param_Name,		"ƫ��ģʽ",
	StrID_Switch_Popup_Choices,		"λ��|"
									"��ת",
	StrID_Return_Group_Name,		"��������",
	StrID_Return_Param_Name,		"�۷�����",
	StrID_Return2_Param_Name,		"���۷�������β���ظ�",
	StrID_Offset_Group_Name,		"λ������",
	StrID_Offset_Param_Name,		"λ��ƫ��",
	StrID_Offset_Count_Param_Name,	"��������",
	StrID_Rotate_Group_Name,		"��ת����",
	StrID_Rotate_Param_Name,		"��ת�Ƕ�",
	StrID_Rotate_Scale_Param_Name,	"��ת����",
	StrID_Max_Dur_Name,				"������ʱ�� (��)",
	StrID_Xfer_Mode_Name,			"���ģʽ",
	StrID_Xfer_Mode_Choices,		"��ǰ����|"
									"��󸲸�|"
									"(-|"
									"�ܽ�|"
									"���|"
									"���|"
									"��Ļ|"
									"����|"
									"���|"
									"ǿ��|"
									"�䰵|"
									"����|"
									"����|"
									"ɫ��|"
									"���Ͷ�|"
									"��ɫ|"
									"�����|"
									"Alpha ���|"
									"�ų�|"
									"���|"
									"���",
	StrID_Second_Layer_Name,		"�ڶ�ͼ��",
	StrID_Random_Name,				"����ˢ����Ⱦ�ı�������",
};


char	*GetStringPtr(int strNum)
{
	return g_strs[strNum].str;
}